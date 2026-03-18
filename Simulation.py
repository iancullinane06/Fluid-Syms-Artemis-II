from Mechanisms.Classes import RocketProfile
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.widgets import Slider
from Mechanisms.FluidSimulation import FluidSimulation
from Mechanisms.RocketDynamics import RocketDynamics
from Mechanisms.Graphing import format_duration, update_progress_bar

# Physics and display cadence are decoupled:
# - small physics dt keeps advection/projection stable
# - larger frame interval keeps animation length manageable
# - compressible mode may substep internally with a CFL-limited dt
USE_COMPRESSIBLE = False
time_step = 0.02 if USE_COMPRESSIBLE else 0.1
sim_time = 80.0 if USE_COMPRESSIBLE else 250.0
frame_interval = 0.5 if USE_COMPRESSIBLE else 1.0
num_frames = int(sim_time / frame_interval)

SIM_SPEED_SCALE = 1.0 if USE_COMPRESSIBLE else 1.0 / 80.0
COMPRESSIBLE_BASE_MACH = 1.45
COMPRESSIBLE_GUST_MACH = 0.025
COMPRESSIBLE_CFL_NUMBER = 0.70
COMPRESSIBLE_FLUX_SCHEME = "hllc"

EXPORT_VIDEO = True
SHOW_PLOT = True
VIDEO_FPS = 20
VIDEO_DPI = 140
VIDEO_BASENAME = "rocket_compressible"
EXPORT_GIF_COPY = True
VIDEO_CASE_TAG = "baseline"
FIGURE_HEIGHT_SCALE = 1.20
INLET_MACH = COMPRESSIBLE_BASE_MACH

# Artemis-II-inspired slender profile proportions for the 2D silhouette.
artemis_diameter_m = 8.4
artemis_length_m = 98.0
rocket_height_cells = 300  # Example value, adjust as needed
rocket_width_cells = round(rocket_height_cells *
                           (artemis_diameter_m / artemis_length_m), 1)
rocket_body_fraction = 0.84
rocket_nose_points = 96
rocket_fore_spike_fraction = 0.08
rocket_fore_spike_half_width_fraction = 0.12

rocket = RocketProfile(
    name="Artemis-II-like",
    mass=2_600_000.0,
    thrust=39_000_000.0,
    burn_time=510.0,
    width=rocket_width_cells,
    height=rocket_height_cells,
)


def thrust_profile_n(time_s: float) -> float:
    # Approximate Artemis II / SLS Block-1 thrust history.
    time_knots_s = np.array(
        [0.0, 8.0, 40.0, 80.0, 120.0, 126.0, 320.0, 500.0, 510.0, 700.0], dtype=float)
    thrust_knots_n = np.array(
        [39.0e6, 38.9e6, 38.5e6, 38.2e6, 37.8e6, 9.0e6, 8.9e6, 8.6e6, 0.0, 0.0], dtype=float)
    return float(np.interp(float(time_s), time_knots_s, thrust_knots_n))


dynamics = RocketDynamics(
    mass_kg=rocket.mass,
    thrust_profile=thrust_profile_n,
    sim_speed_scale=SIM_SPEED_SCALE,
    drag_force_scale_n=18000.0,
    flight_direction=(1.0, 0.0),
    dry_mass_kg=900_000.0,
    specific_impulse_s=330.0,
)

SEA_LEVEL_ATMOSPHERE = dynamics.atmosphere_at_altitude(0.0)


def ambient_velocity_profile(time_s: float) -> np.ndarray:
    if USE_COMPRESSIBLE:
        base_wind_mps = COMPRESSIBLE_BASE_MACH * \
            SEA_LEVEL_ATMOSPHERE.speed_of_sound_m_s
        gust_mps = COMPRESSIBLE_GUST_MACH * SEA_LEVEL_ATMOSPHERE.speed_of_sound_m_s * (
            0.8 * np.sin(0.28 * time_s) + 0.4 * np.sin(0.92 * time_s + 0.35)
        )
    else:
        # Ambient wind in rocket frame should be small at liftoff; keep this mild so
        # initial relative speed does not start artificially high.
        base_wind_mps = 4.0
        gust_mps = 1.5 * np.sin(0.28 * time_s) + 0.8 * np.sin(0.92 * time_s + 0.35)
    wind_solver = (base_wind_mps + gust_mps) * SIM_SPEED_SCALE
    return np.array([-wind_solver, 0.0], dtype=float)


def rocket_velocity_profile(time_s: float) -> np.ndarray:
    return dynamics.rocket_velocity_profile(time_s)


def compute_diagnostics(sim: FluidSimulation):
    speed = np.hypot(sim.u, sim.v)

    # Vorticity: dvdx - dudy
    dvdx = np.zeros_like(sim.v)
    dudy = np.zeros_like(sim.u)
    dvdx[1:-1, 1:-1] = (sim.v[1:-1, 2:] - sim.v[1:-1, :-2]) * 0.5
    dudy[1:-1, 1:-1] = (sim.u[2:, 1:-1] - sim.u[:-2, 1:-1]) * 0.5
    vorticity = dvdx - dudy

    # Pressure field (already computed during projection step)
    pressure = sim.p.copy()
    if sim.compressible:
        pressure = pressure - float(getattr(sim, "freestream_pressure", 0.0))

    # Streamwise velocity component projected onto current freestream direction.
    streamwise_velocity = (
        sim.u * float(sim.freestream_direction[0])
        + sim.v * float(sim.freestream_direction[1])
    )

    total_drag_signed_n, pressure_drag_signed_n, shear_drag_signed_n, shear_stress = dynamics.compute_drag_components_n(
        sim)
    total_drag_mag_n, pressure_drag_mag_n, shear_drag_mag_n, _ = dynamics.compute_drag_components_n_magnitude(
        sim)

    diagnostics = {
        "speed": speed,
        "pressure": pressure,
        "streamwise_velocity": streamwise_velocity,
        "vorticity": vorticity,
        "density": sim.rho.copy() if sim.compressible else None,
        "temperature": sim.temperature.copy() if sim.compressible else None,
        "mach": sim.mach.copy() if sim.compressible else None,
        "shear_stress": shear_stress,
        "drag_total": total_drag_mag_n,
        "drag_pressure": pressure_drag_mag_n,
        "drag_shear": shear_drag_mag_n,
        "drag_total_signed": total_drag_signed_n,
        "drag_pressure_signed": pressure_drag_signed_n,
        "drag_shear_signed": shear_drag_signed_n,
    }
    return diagnostics


# Initialise the fluid simulation
grid_size = (140, 460) if USE_COMPRESSIBLE else (200, 650)  # Height x Width
viscosity = (
    SEA_LEVEL_ATMOSPHERE.dynamic_viscosity_pa_s /
    SEA_LEVEL_ATMOSPHERE.density_kg_m3
    if USE_COMPRESSIBLE
    else 0.0000018
)  # Kinematic viscosity (m^2/s)
density = SEA_LEVEL_ATMOSPHERE.density_kg_m3 if USE_COMPRESSIBLE else 1.225
initial_relative_velocity = ambient_velocity_profile(
    0.0) - rocket_velocity_profile(0.0)
initial_speed = float(np.linalg.norm(initial_relative_velocity))
freestream_speed = max(initial_speed, 1e-6)
freestream_direction = tuple(
    (initial_relative_velocity / freestream_speed).tolist())
simulation = FluidSimulation(
    grid_size,
    viscosity,
    time_step,
    density,
    rocket_profile=rocket,
    acceleration=0.0,
    acceleration_direction=freestream_direction,
    edge_speed=freestream_speed,
    assume_laminar_edges=True,
    edge_relaxation=0.985,
    wall_penalty=22.0,         # reduce over-damping; still no-slip near wall
    wall_shear_layers=6,
    turbulence_strength=0.8,  # lower SGS damping so wake structures survive
    vorticity_confinement=0.32,
    ambient_velocity_profile=ambient_velocity_profile,
    rocket_velocity_profile=rocket_velocity_profile,
    inflow_blend=0.02,
    compressible=USE_COMPRESSIBLE,
    reference_temperature=SEA_LEVEL_ATMOSPHERE.temperature_k,
    cfl_number=COMPRESSIBLE_CFL_NUMBER,
    compressible_flux_scheme=COMPRESSIBLE_FLUX_SCHEME,
    compressible_velocity_diffusion=0.0,
)

if USE_COMPRESSIBLE:
    simulation.set_freestream_thermodynamics(
        density=SEA_LEVEL_ATMOSPHERE.density_kg_m3,
        temperature_k=SEA_LEVEL_ATMOSPHERE.temperature_k,
        pressure_pa=SEA_LEVEL_ATMOSPHERE.pressure_pa,
    )
simulation.set_uniform_flow(freestream_speed, freestream_direction)

rocket_center_x = grid_size[1] // 2
rocket_center_y = grid_size[0] // 2
simulation.add_rocket_profile(
    rocket_profile=rocket,
    center=(rocket_center_x, rocket_center_y),
    body_fraction=rocket_body_fraction,
    nose_points=rocket_nose_points,
    fore_spike_fraction=rocket_fore_spike_fraction,
    fore_spike_half_width_fraction=rocket_fore_spike_half_width_fraction,
)

rows, cols = grid_size

profile_x, profile_y = rocket.get_2d_profile_polygon(
    center_x=rocket_center_x,
    center_y=rocket_center_y,
    body_fraction=rocket_body_fraction,
    nose_points=rocket_nose_points,
    fore_spike_fraction=rocket_fore_spike_fraction,
    fore_spike_half_width_fraction=rocket_fore_spike_half_width_fraction,
)

x = np.arange(grid_size[1], dtype=float)
y = np.arange(grid_size[0], dtype=float)
X, Y = np.meshgrid(x, y)

# stride=16 on 400×400 keeps quiver density readable (~25×25 arrows)
quiver_stride = 16
Xq = X[::quiver_stride, ::quiver_stride]
Yq = Y[::quiver_stride, ::quiver_stride]

# Precompute flow frames for slider-based time navigation.
total_time = num_frames * frame_interval
obstacle = simulation.obstacle_mask
frames = []
drag_history = []

global_min_speed = np.inf
global_max_speed = -np.inf
global_min_pressure = np.inf
global_max_pressure = -np.inf
global_min_density = np.inf
global_max_density = -np.inf
global_min_temperature = np.inf
global_max_temperature = -np.inf
global_min_mach = np.inf
global_max_mach = -np.inf
global_max_shear = -np.inf

precompute_start_time = time.perf_counter()
update_progress_bar(0, num_frames, precompute_start_time)

for frame_index in range(num_frames):
    target_time = min((frame_index + 1) * frame_interval, sim_time)
    while simulation.simulation_time < target_time - 1e-12:
        simulation.step_coupled(dynamics)

    dx = simulation.u.copy()
    dy = simulation.v.copy()
    diagnostics = compute_diagnostics(simulation)
    frame_time = simulation.simulation_time
    frame_speed = simulation.edge_speed
    frame_thrust_n = float(dynamics.state.thrust_n)
    frame_net_force_n = float(dynamics.state.net_force_n)
    frame_rocket_speed_mps = float(dynamics.state.velocity_mps)
    frame_altitude_m = float(dynamics.state.altitude_m)
    frame_freestream_mach = 0.0
    if simulation.compressible:
        local_atmosphere = dynamics.atmosphere_at_altitude(frame_altitude_m)
        frame_freestream_mach = frame_speed / \
            max(local_atmosphere.speed_of_sound_m_s, 1e-6)

    dx[obstacle] = 0.0
    dy[obstacle] = 0.0
    speed_plot = diagnostics["speed"].copy()
    speed_plot[obstacle] = np.nan
    pressure_plot = diagnostics["pressure"].copy()
    pressure_plot[obstacle] = np.nan

    finite_speed = speed_plot[np.isfinite(speed_plot)]
    if finite_speed.size > 0:
        global_min_speed = min(global_min_speed, float(np.min(finite_speed)))
        global_max_speed = max(global_max_speed, float(np.max(finite_speed)))

    finite_pressure = pressure_plot[np.isfinite(pressure_plot)]
    if finite_pressure.size > 0:
        global_min_pressure = min(
            global_min_pressure, float(np.min(finite_pressure)))
        global_max_pressure = max(
            global_max_pressure, float(np.max(finite_pressure)))

    frame_data = {
        "dx": dx,
        "dy": dy,
        "speed_plot": speed_plot,
        "pressure_plot": pressure_plot,
        "frame_time": frame_time,
        "frame_speed": frame_speed,
        "frame_mach": frame_freestream_mach,
        "drag_total": diagnostics["drag_total"],
        "drag_pressure": diagnostics["drag_pressure"],
        "drag_shear": diagnostics["drag_shear"],
        "drag_total_signed": diagnostics["drag_total_signed"],
        "drag_pressure_signed": diagnostics["drag_pressure_signed"],
        "drag_shear_signed": diagnostics["drag_shear_signed"],
        "frame_thrust_n": frame_thrust_n,
        "frame_net_force_n": frame_net_force_n,
        "frame_rocket_speed_mps": frame_rocket_speed_mps,
        "frame_altitude_m": frame_altitude_m,
    }

    if simulation.compressible:
        density_plot = diagnostics["density"].copy()
        density_plot[obstacle] = np.nan
        temperature_plot = diagnostics["temperature"].copy()
        temperature_plot[obstacle] = np.nan
        mach_plot = diagnostics["mach"].copy()
        mach_plot[obstacle] = np.nan

        finite_density = density_plot[np.isfinite(density_plot)]
        if finite_density.size > 0:
            global_min_density = min(global_min_density, float(np.min(finite_density)))
            global_max_density = max(global_max_density, float(np.max(finite_density)))

        finite_temperature = temperature_plot[np.isfinite(temperature_plot)]
        if finite_temperature.size > 0:
            global_min_temperature = min(
                global_min_temperature, float(np.min(finite_temperature)))
            global_max_temperature = max(
                global_max_temperature, float(np.max(finite_temperature)))

        finite_mach = mach_plot[np.isfinite(mach_plot)]
        if finite_mach.size > 0:
            global_min_mach = min(global_min_mach, float(np.min(finite_mach)))
            global_max_mach = max(global_max_mach, float(np.max(finite_mach)))

        frame_data["density_plot"] = density_plot
        frame_data["temperature_plot"] = temperature_plot
        frame_data["mach_plot"] = mach_plot
    else:
        streamwise_plot = diagnostics["streamwise_velocity"].copy()
        streamwise_plot[obstacle] = np.nan
        vorticity_plot = np.abs(diagnostics["vorticity"])
        vorticity_plot[obstacle] = np.nan
        shear_plot = diagnostics["shear_stress"].copy()
        shear_plot[obstacle] = np.nan

        finite_shear = shear_plot[np.isfinite(shear_plot)]
        if finite_shear.size > 0:
            global_max_shear = max(global_max_shear, float(np.max(finite_shear)))

        frame_data["streamwise_plot"] = streamwise_plot
        frame_data["vorticity_plot"] = vorticity_plot
        frame_data["shear_plot"] = shear_plot

    frames.append(frame_data)
    drag_history.append((
        frame_time,
        diagnostics["drag_total"],
        diagnostics["drag_pressure"],
        diagnostics["drag_shear"],
        diagnostics["drag_total_signed"],
        diagnostics["drag_pressure_signed"],
        diagnostics["drag_shear_signed"],
        frame_thrust_n,
        frame_net_force_n,
        frame_altitude_m,
    ))
    update_progress_bar(frame_index + 1, num_frames, precompute_start_time)

if not np.isfinite(global_min_speed) or not np.isfinite(global_max_speed):
    global_min_speed, global_max_speed = 0.0, 1.0
if not np.isfinite(global_min_pressure) or not np.isfinite(global_max_pressure):
    global_min_pressure, global_max_pressure = -1.0, 1.0
if not np.isfinite(global_min_density) or not np.isfinite(global_max_density):
    global_min_density, global_max_density = 0.0, 1.0
if not np.isfinite(global_min_temperature) or not np.isfinite(global_max_temperature):
    global_min_temperature, global_max_temperature = 200.0, 400.0
if not np.isfinite(global_min_mach) or not np.isfinite(global_max_mach):
    global_min_mach, global_max_mach = 0.0, 1.0
if not np.isfinite(global_max_shear):
    global_max_shear = 1.0

max_dim = max(rows, cols)
fig_width = 16.0 * cols / max_dim
fig_height = 10.8 * rows / max_dim * FIGURE_HEIGHT_SCALE
fig = plt.figure(figsize=(fig_width, fig_height))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                      bottom=0.12, top=0.95, left=0.08, right=0.96)

ax_speed = fig.add_subplot(gs[0, 0])
ax_pressure = fig.add_subplot(gs[0, 1])
ax_streamwise = fig.add_subplot(gs[0, 2])
ax_vortex = fig.add_subplot(gs[1, 0])
ax_drag = fig.add_subplot(gs[1, 1])
ax_drag_altitude = ax_drag.twinx()
ax_shear = fig.add_subplot(gs[1, 2])

colorbars = {}
field_images = {}


def draw_frame(frame_index):
    global colorbars, field_images
    for ax in [ax_speed, ax_pressure, ax_streamwise, ax_vortex, ax_shear]:
        ax.clear()

    ax_drag.clear()
    ax_drag_altitude.clear()

    frame = frames[frame_index]
    speed_plot = frame["speed_plot"]
    pressure_plot = frame["pressure_plot"]
    frame_time = frame["frame_time"]
    frame_speed = frame["frame_speed"]
    frame_mach = frame["frame_mach"]
    drag_total = frame["drag_total"]
    drag_pressure = frame["drag_pressure"]
    drag_shear = frame["drag_shear"]
    drag_total_signed = frame["drag_total_signed"]
    frame_thrust_n = frame["frame_thrust_n"]
    frame_net_force_n = frame["frame_net_force_n"]
    frame_rocket_speed_mps = frame["frame_rocket_speed_mps"]
    frame_altitude_m = frame["frame_altitude_m"]

    # Top-left: Velocity magnitude
    ax_speed.set_facecolor("white")
    im_speed = ax_speed.imshow(
        speed_plot,
        origin="lower",
        cmap="viridis",
        vmin=global_min_speed,
        vmax=global_max_speed,
        extent=(0, cols - 1, 0, rows - 1),
        alpha=0.75,
        interpolation="bilinear",
    )
    ax_speed.fill(profile_x, profile_y,
                  color="cornflowerblue", alpha=0.35, zorder=6)
    ax_speed.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
    ax_speed.set(aspect=1, title="Velocity Magnitude")
    ax_speed.set_xlabel("X", fontsize=9)
    ax_speed.set_ylabel("Y", fontsize=9)
    ax_speed.set_xlim(0, cols - 1)
    ax_speed.set_ylim(0, rows - 1)
    if "speed" not in colorbars:
        colorbars["speed"] = plt.colorbar(
            im_speed, ax=ax_speed, fraction=0.046, pad=0.04)
        colorbars["speed"].set_label("m/s", fontsize=8)
    else:
        colorbars["speed"].update_normal(im_speed)

    # Top-middle: Pressure
    ax_pressure.set_facecolor("white")
    pressure_vmin = global_min_pressure
    pressure_vmax = global_max_pressure
    pressure_title = "Pressure Field"
    if simulation.compressible:
        pressure_min = np.nanpercentile(pressure_plot, 1)
        pressure_max = np.nanpercentile(pressure_plot, 99)
        pressure_lim = max(abs(pressure_min), abs(pressure_max), 1.0)
        pressure_vmin = -pressure_lim
        pressure_vmax = pressure_lim
        pressure_title = "Gauge Pressure Field"

    im_pressure = ax_pressure.imshow(
        pressure_plot,
        origin="lower",
        cmap="RdBu_r",
        vmin=pressure_vmin,
        vmax=pressure_vmax,
        extent=(0, cols - 1, 0, rows - 1),
        alpha=0.75,
        interpolation="bilinear",
    )
    ax_pressure.fill(profile_x, profile_y,
                     color="cornflowerblue", alpha=0.35, zorder=6)
    ax_pressure.plot(profile_x, profile_y, color="blue",
                     linewidth=1.5, zorder=7)
    ax_pressure.set(aspect=1, title=pressure_title)
    ax_pressure.set_xlabel("X", fontsize=9)
    ax_pressure.set_ylabel("Y", fontsize=9)
    ax_pressure.set_xlim(0, cols - 1)
    ax_pressure.set_ylim(0, rows - 1)
    if "pressure" not in colorbars:
        colorbars["pressure"] = plt.colorbar(
            im_pressure, ax=ax_pressure, fraction=0.046, pad=0.04)
        colorbars["pressure"].set_label("Pa", fontsize=8)
    else:
        colorbars["pressure"].update_normal(im_pressure)

    if simulation.compressible:
        mach_plot = frame["mach_plot"]
        density_plot = frame["density_plot"]
        temperature_plot = frame["temperature_plot"]

        mach_vmin = np.nanpercentile(mach_plot, 1)
        mach_vmax = np.nanpercentile(mach_plot, 99)
        if not np.isfinite(mach_vmin) or not np.isfinite(mach_vmax) or mach_vmax <= mach_vmin:
            mach_vmin = global_min_mach
            mach_vmax = global_max_mach
        if mach_vmax <= mach_vmin:
            mach_vmax = mach_vmin + 1e-6

        # Top-right: Mach number
        ax_streamwise.set_facecolor("white")
        im_streamwise = ax_streamwise.imshow(
            mach_plot,
            origin="lower",
            cmap="plasma",
            vmin=mach_vmin,
            vmax=mach_vmax,
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_streamwise.fill(profile_x, profile_y,
                           color="cornflowerblue", alpha=0.35, zorder=6)
        ax_streamwise.plot(profile_x, profile_y, color="blue",
                           linewidth=1.5, zorder=7)
        ax_streamwise.set(aspect=1, title="Mach Number")
        ax_streamwise.set_xlabel("X", fontsize=9)
        ax_streamwise.set_ylabel("Y", fontsize=9)
        ax_streamwise.set_xlim(0, cols - 1)
        ax_streamwise.set_ylim(0, rows - 1)
        if "streamwise" not in colorbars:
            colorbars["streamwise"] = plt.colorbar(
                im_streamwise, ax=ax_streamwise, fraction=0.046, pad=0.04)
            colorbars["streamwise"].set_label("M", fontsize=8)
        else:
            colorbars["streamwise"].update_normal(im_streamwise)

        # Bottom-left: Density
        ax_vortex.set_facecolor("white")
        density_vmin = np.nanpercentile(density_plot, 1)
        density_vmax = np.nanpercentile(density_plot, 99)
        if not np.isfinite(density_vmin) or not np.isfinite(density_vmax) or density_vmax <= density_vmin:
            density_vmin = global_min_density
            density_vmax = global_max_density
        if density_vmax <= density_vmin:
            density_vmax = density_vmin + 1e-6

        im_vortex = ax_vortex.imshow(
            density_plot,
            origin="lower",
            cmap="cividis",
            vmin=density_vmin,
            vmax=density_vmax,
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_vortex.fill(profile_x, profile_y,
                       color="cornflowerblue", alpha=0.35, zorder=6)
        ax_vortex.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
        ax_vortex.set(aspect=1, title="Density Field")
        ax_vortex.set_xlabel("X", fontsize=9)
        ax_vortex.set_ylabel("Y", fontsize=9)
        ax_vortex.set_xlim(0, cols - 1)
        ax_vortex.set_ylim(0, rows - 1)
        if "vortex" not in colorbars:
            colorbars["vortex"] = plt.colorbar(
                im_vortex, ax=ax_vortex, fraction=0.046, pad=0.04)
            colorbars["vortex"].set_label("kg/m³", fontsize=8)
        else:
            colorbars["vortex"].update_normal(im_vortex)
    else:
        streamwise_plot = frame["streamwise_plot"]
        vorticity_plot = frame["vorticity_plot"]

        # Top-right: Streamwise velocity
        ax_streamwise.set_facecolor("white")
        # Use symmetric colormap centered on zero
        streamwise_min = np.nanpercentile(streamwise_plot, 1)
        streamwise_max = np.nanpercentile(streamwise_plot, 99)
        streamwise_lim = max(abs(streamwise_min), abs(streamwise_max))
        im_streamwise = ax_streamwise.imshow(
            streamwise_plot,
            origin="lower",
            cmap="RdBu_r",
            vmin=-streamwise_lim,
            vmax=streamwise_lim,
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_streamwise.fill(profile_x, profile_y,
                           color="cornflowerblue", alpha=0.35, zorder=6)
        ax_streamwise.plot(profile_x, profile_y, color="blue",
                           linewidth=1.5, zorder=7)
        ax_streamwise.set(aspect=1, title="Streamwise Velocity")
        ax_streamwise.set_xlabel("X", fontsize=9)
        ax_streamwise.set_ylabel("Y", fontsize=9)
        ax_streamwise.set_xlim(0, cols - 1)
        ax_streamwise.set_ylim(0, rows - 1)
        if "streamwise" not in colorbars:
            colorbars["streamwise"] = plt.colorbar(
                im_streamwise, ax=ax_streamwise, fraction=0.046, pad=0.04)
            colorbars["streamwise"].set_label("m/s", fontsize=8)
        else:
            colorbars["streamwise"].update_normal(im_streamwise)

        # Bottom-left: Vorticity
        ax_vortex.set_facecolor("white")
        im_vortex = ax_vortex.imshow(
            vorticity_plot,
            origin="lower",
            cmap="Spectral",
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_vortex.fill(profile_x, profile_y,
                       color="cornflowerblue", alpha=0.35, zorder=6)
        ax_vortex.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
        ax_vortex.set(aspect=1, title="Vorticity Magnitude")
        ax_vortex.set_xlabel("X", fontsize=9)
        ax_vortex.set_ylabel("Y", fontsize=9)
        ax_vortex.set_xlim(0, cols - 1)
        ax_vortex.set_ylim(0, rows - 1)
        if "vortex" not in colorbars:
            colorbars["vortex"] = plt.colorbar(
                im_vortex, ax=ax_vortex, fraction=0.046, pad=0.04)
            colorbars["vortex"].set_label("1/s", fontsize=8)
        else:
            colorbars["vortex"].update_normal(im_vortex)

    # Bottom-middle: Drag/thrust/net force vs time, with altitude displacement.
    history_time = np.array(
        [item[0] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_total = np.array(
        [item[1] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_pressure = np.array(
        [item[2] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_shear = np.array(
        [item[3] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_total_signed = np.array(
        [item[4] for item in drag_history[: frame_index + 1]], dtype=float)
    history_thrust = np.array(
        [item[7] for item in drag_history[: frame_index + 1]], dtype=float)
    history_net = np.array(
        [item[8] for item in drag_history[: frame_index + 1]], dtype=float)
    history_altitude = np.array(
        [item[9] for item in drag_history[: frame_index + 1]], dtype=float)

    ax_drag.plot(history_time, history_drag_total,
                 color="#1f77b4", linewidth=2.0, label="Total")
    ax_drag.plot(history_time, history_drag_total_signed,
                 color="#17becf", linewidth=1.1, linestyle="-.", label="Signed Total")
    ax_drag.plot(history_time, history_drag_pressure,
                 color="#d62728", linewidth=1.3, linestyle="--", label="Pressure")
    ax_drag.plot(history_time, history_drag_shear,
                 color="#2ca02c", linewidth=1.3, linestyle="--", label="Shear")
    ax_drag.plot(history_time, history_thrust, color="#9467bd",
                 linewidth=1.5, linestyle=":", label="Thrust")
    ax_drag.plot(history_time, history_net, color="#ff7f0e",
                 linewidth=1.5, linestyle="-.", label="Net (T-D-W)")
    ax_drag.scatter([frame_time], [drag_total_signed],
                    color="black", s=34, zorder=3)

    ax_drag_altitude.plot(history_time, history_altitude, color="#111111",
                          linewidth=1.4, linestyle="-", label="Altitude")
    ax_drag_altitude.set_ylabel("Vertical Displacement (m)", fontsize=9)

    if history_time.size > 0:
        x_min = float(np.min(history_time))
        x_max = float(np.max(history_time))
        x_pad = max(0.05 * (x_max - x_min), 1e-3)
        ax_drag.set_xlim(x_min - x_pad, x_max + x_pad)
    else:
        ax_drag.set_xlim(0.0, 1.0)

    combined_forces = np.concatenate((history_drag_total, history_drag_total_signed, history_drag_pressure, history_drag_shear,
                                     history_thrust, history_net)) if history_drag_total.size > 0 else np.array([0.0])
    y_min = float(np.min(combined_forces))
    y_max = float(np.max(combined_forces))
    y_span = max(y_max - y_min, 1e-6)
    ax_drag.set_ylim(y_min - 0.1 * y_span, y_max + 0.1 * y_span)
    ax_drag.set_ylabel("Force (N)", fontsize=9)
    ax_drag.set_xlabel("Time (s)", fontsize=9)
    ax_drag.grid(True, alpha=0.3)
    ax_drag.set_title("Forces & Vertical Displacement vs Time", fontsize=10)
    force_handles, force_labels = ax_drag.get_legend_handles_labels()
    altitude_handles, altitude_labels = ax_drag_altitude.get_legend_handles_labels()
    ax_drag.legend(force_handles + altitude_handles,
                   force_labels + altitude_labels, fontsize=8, loc="best")

    if simulation.compressible:
        temperature_plot = frame["temperature_plot"]
        temperature_vmin = np.nanpercentile(temperature_plot, 1)
        temperature_vmax = np.nanpercentile(temperature_plot, 99)
        if (
            not np.isfinite(temperature_vmin)
            or not np.isfinite(temperature_vmax)
            or temperature_vmax <= temperature_vmin
        ):
            temperature_vmin = global_min_temperature
            temperature_vmax = global_max_temperature
        if temperature_vmax <= temperature_vmin:
            temperature_vmax = temperature_vmin + 1e-6

        # Bottom-right: Temperature
        ax_shear.set_facecolor("white")
        im_shear = ax_shear.imshow(
            temperature_plot,
            origin="lower",
            cmap="inferno",
            vmin=temperature_vmin,
            vmax=temperature_vmax,
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_shear.fill(profile_x, profile_y,
                      color="cornflowerblue", alpha=0.35, zorder=6)
        ax_shear.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
        ax_shear.set(aspect=1, title="Temperature Field")
        ax_shear.set_xlabel("X", fontsize=9)
        ax_shear.set_ylabel("Y", fontsize=9)
        ax_shear.set_xlim(0, cols - 1)
        ax_shear.set_ylim(0, rows - 1)
        if "shear" not in colorbars:
            colorbars["shear"] = plt.colorbar(
                im_shear, ax=ax_shear, fraction=0.046, pad=0.04)
            colorbars["shear"].set_label("K", fontsize=8)
        else:
            colorbars["shear"].update_normal(im_shear)
    else:
        shear_plot = frame["shear_plot"]

        # Bottom-right: Wall shear stress
        ax_shear.set_facecolor("white")
        im_shear = ax_shear.imshow(
            shear_plot,
            origin="lower",
            cmap="hot",
            vmin=0.0,
            vmax=global_max_shear,
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_shear.fill(profile_x, profile_y,
                      color="cornflowerblue", alpha=0.35, zorder=6)
        ax_shear.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
        ax_shear.set(aspect=1, title="Wall Shear Stress Magnitude")
        ax_shear.set_xlabel("X", fontsize=9)
        ax_shear.set_ylabel("Y", fontsize=9)
        ax_shear.set_xlim(0, cols - 1)
        ax_shear.set_ylim(0, rows - 1)
        if "shear" not in colorbars:
            colorbars["shear"] = plt.colorbar(
                im_shear, ax=ax_shear, fraction=0.046, pad=0.04)
            colorbars["shear"].set_label("Pa", fontsize=8)
        else:
            colorbars["shear"].update_normal(im_shear)

    freestream_mps = frame_speed / SIM_SPEED_SCALE

    # Main title with time info
    if simulation.compressible:
        fig.suptitle(
            f"Rocket-Frame Compressible CFD | t={frame_time:.2f}s | U∞={frame_speed:.1f} m/s | M∞≈{frame_mach:.2f} | Vrocket={frame_rocket_speed_mps:.1f} m/s",
            fontsize=12,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            f"Rocket-Frame CFD Analysis | t={frame_time:.2f}s | U∞={frame_speed:.2f} (solver) ≈ {freestream_mps:.1f} m/s | Vrocket={frame_rocket_speed_mps:.1f} m/s",
            fontsize=12,
            fontweight="bold",
        )


draw_frame(0)

slider_ax = fig.add_axes((0.20, 0.03, 0.65, 0.035))
time_slider = Slider(
    ax=slider_ax,
    label=f"Time Index (0 - {num_frames - 1})",
    valmin=0,
    valmax=num_frames - 1,
    valinit=0,
    valstep=1,
)


def on_slider_change(value):
    frame_index = int(value)
    draw_frame(frame_index)
    fig.canvas.draw_idle()


time_slider.on_changed(on_slider_change)

def _safe_tag(tag: str) -> str:
    """Sanitize string for use in filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)

def _build_video_basename() -> str:
    mach_tag = f"m{INLET_MACH:.2f}".replace(".", "p")
    return f"{VIDEO_BASENAME}_{COMPRESSIBLE_FLUX_SCHEME}_{mach_tag}_{_safe_tag(VIDEO_CASE_TAG)}"

def _resolve_ffmpeg_executable() -> str | None:
    env_path = os.environ.get("FFMPEG_PATH")
    candidates = [
        env_path,
        shutil.which("ffmpeg"),
        shutil.which("ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None

def save_simulation_video():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    animation = FuncAnimation(
        fig,
        lambda i: draw_frame(i) or [],
        frames=num_frames,
        interval=1000.0 / float(max(VIDEO_FPS, 1)),
        blit=False,
        repeat=False,
    )

    basename = _build_video_basename()
    mp4_path = output_dir / f"{basename}.mp4"
    gif_path = output_dir / f"{basename}.gif"

    ffmpeg_path = _resolve_ffmpeg_executable()
    saved_mp4 = False

    if ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        try:
            writer = FFMpegWriter(
                fps=VIDEO_FPS,
                codec="libx264",
                bitrate=5000,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            animation.save(str(mp4_path), writer=writer, dpi=VIDEO_DPI)
            print(f"Saved video: {mp4_path}")
            saved_mp4 = True
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"FFmpeg export failed ({exc}).")
    else:
        print("FFmpeg executable not found (set FFMPEG_PATH or install ffmpeg).")

    if EXPORT_GIF_COPY or not saved_mp4:
        gif_writer = PillowWriter(fps=VIDEO_FPS)
        animation.save(str(gif_path), writer=gif_writer, dpi=VIDEO_DPI)
        print(f"Saved GIF: {gif_path}")


if EXPORT_VIDEO:
    save_simulation_video()

if SHOW_PLOT:
    plt.show()

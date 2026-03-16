from Mechanisms.Classes import RocketProfile
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider
from Mechanisms.FluidSimulation import FluidSimulation
from Mechanisms.RocketDynamics import RocketDynamics
from Mechanisms.Graphing import format_duration, update_progress_bar

# Physics and display cadence are decoupled:
# - small physics dt keeps advection/projection stable
# - larger frame interval keeps animation length manageable
time_step = 0.1
sim_time = 250.0
frame_interval = 1.0
steps_per_frame = max(int(round(frame_interval / time_step)), 1)
num_frames = int(sim_time / frame_interval)

SIM_SPEED_SCALE = 1.0 / 80.0

# Artemis-II-inspired slender profile proportions for the 2D silhouette.
artemis_diameter_m = 8.4
artemis_length_m = 98.0
rocket_height_cells = 300  # Example value, adjust as needed
rocket_width_cells = round(rocket_height_cells * (artemis_diameter_m / artemis_length_m), 1)
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
    time_knots_s = np.array([0.0, 8.0, 40.0, 80.0, 120.0, 126.0, 320.0, 500.0, 510.0, 700.0], dtype=float)
    thrust_knots_n = np.array([39.0e6, 38.9e6, 38.5e6, 38.2e6, 37.8e6, 9.0e6, 8.9e6, 8.6e6, 0.0, 0.0], dtype=float)
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

def ambient_velocity_profile(time_s: float) -> np.ndarray:
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

    # Streamwise velocity component projected onto current freestream direction.
    streamwise_velocity = (
        sim.u * float(sim.freestream_direction[0])
        + sim.v * float(sim.freestream_direction[1])
    )

    total_drag_n, pressure_drag_n, shear_drag_n, shear_stress = dynamics.compute_drag_components_n(sim)

    return speed, vorticity, total_drag_n, pressure_drag_n, shear_drag_n, pressure, streamwise_velocity, shear_stress

# Initialise the fluid simulation
grid_size = (200, 650)  # Height x Width
viscosity = 0.0000018  # Lower diffusion to preserve wake structures
density = 1.225  # Air density
initial_relative_velocity = ambient_velocity_profile(0.0) - rocket_velocity_profile(0.0)
initial_speed = float(np.linalg.norm(initial_relative_velocity))
freestream_speed = max(initial_speed, 1e-6)
freestream_direction = tuple((initial_relative_velocity / freestream_speed).tolist())
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

quiver_stride = 16  # stride=16 on 400×400 keeps quiver density readable (~25×25 arrows)
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
global_max_shear = -np.inf

precompute_start_time = time.perf_counter()
update_progress_bar(0, num_frames, precompute_start_time)

for frame_index in range(num_frames):
    for _ in range(steps_per_frame):
        simulation.step_coupled(dynamics, dt=time_step)

    dx = simulation.u.copy()
    dy = simulation.v.copy()
    speed, vorticity, drag_total, drag_pressure, drag_shear, pressure, streamwise_velocity, shear_stress = compute_diagnostics(simulation)
    frame_time = simulation.simulation_time
    frame_speed = simulation.edge_speed
    frame_thrust_n = float(dynamics.state.thrust_n)
    frame_net_force_n = float(dynamics.state.net_force_n)
    frame_rocket_speed_mps = float(dynamics.state.velocity_mps)

    dx[obstacle] = 0.0
    dy[obstacle] = 0.0
    speed_plot = speed.copy()
    speed_plot[obstacle] = np.nan
    vorticity_plot = np.abs(vorticity)
    vorticity_plot[obstacle] = np.nan
    pressure_plot = pressure.copy()
    pressure_plot[obstacle] = np.nan
    streamwise_plot = streamwise_velocity.copy()
    streamwise_plot[obstacle] = np.nan
    shear_plot = shear_stress.copy()
    shear_plot[obstacle] = np.nan

    finite_speed = speed_plot[np.isfinite(speed_plot)]
    if finite_speed.size > 0:
        global_min_speed = min(global_min_speed, float(np.min(finite_speed)))
        global_max_speed = max(global_max_speed, float(np.max(finite_speed)))

    finite_pressure = pressure_plot[np.isfinite(pressure_plot)]
    if finite_pressure.size > 0:
        global_min_pressure = min(global_min_pressure, float(np.min(finite_pressure)))
        global_max_pressure = max(global_max_pressure, float(np.max(finite_pressure)))

    finite_shear = shear_plot[np.isfinite(shear_plot)]
    if finite_shear.size > 0:
        global_max_shear = max(global_max_shear, float(np.max(finite_shear)))

    frames.append((
        dx,
        dy,
        speed_plot,
        vorticity_plot,
        pressure_plot,
        streamwise_plot,
        shear_plot,
        frame_time,
        frame_speed,
        drag_total,
        drag_pressure,
        drag_shear,
        frame_thrust_n,
        frame_net_force_n,
        frame_rocket_speed_mps,
    ))
    drag_history.append((frame_rocket_speed_mps, drag_total, drag_pressure, drag_shear, frame_thrust_n, frame_net_force_n))
    update_progress_bar(frame_index + 1, num_frames, precompute_start_time)

if not np.isfinite(global_min_speed) or not np.isfinite(global_max_speed):
    global_min_speed, global_max_speed = 0.0, 1.0
if not np.isfinite(global_min_pressure) or not np.isfinite(global_max_pressure):
    global_min_pressure, global_max_pressure = -1.0, 1.0
if not np.isfinite(global_max_shear):
    global_max_shear = 1.0

max_dim = max(rows, cols)
fig_width = 16.0 * cols / max_dim
fig_height = 10.8 * rows / max_dim
fig = plt.figure(figsize=(fig_width, fig_height))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30, bottom=0.12, top=0.95, left=0.08, right=0.96)

ax_speed = fig.add_subplot(gs[0, 0])
ax_pressure = fig.add_subplot(gs[0, 1])
ax_streamwise = fig.add_subplot(gs[0, 2])
ax_vortex = fig.add_subplot(gs[1, 0])
ax_drag = fig.add_subplot(gs[1, 1])
ax_shear = fig.add_subplot(gs[1, 2])

colorbars = {}
field_images = {}


def draw_frame(frame_index):
    global colorbars, field_images
    for ax in [ax_speed, ax_pressure, ax_streamwise, ax_vortex, ax_shear]:
        ax.clear()
    
    ax_drag.clear()
    
    (
        dx,
        dy,
        speed_plot,
        vorticity_plot,
        pressure_plot,
        streamwise_plot,
        shear_plot,
        frame_time,
        frame_speed,
        drag_total,
        drag_pressure,
        drag_shear,
        frame_thrust_n,
        frame_net_force_n,
        frame_rocket_speed_mps,
    ) = frames[frame_index]

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
    ax_speed.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
    ax_speed.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
    ax_speed.set(aspect=1, title="Velocity Magnitude")
    ax_speed.set_xlabel("X", fontsize=9)
    ax_speed.set_ylabel("Y", fontsize=9)
    ax_speed.set_xlim(0, cols - 1)
    ax_speed.set_ylim(0, rows - 1)
    if "speed" not in colorbars:
        colorbars["speed"] = plt.colorbar(im_speed, ax=ax_speed, fraction=0.046, pad=0.04)
        colorbars["speed"].set_label("m/s", fontsize=8)
    else:
        colorbars["speed"].update_normal(im_speed)

    # Top-middle: Pressure
    ax_pressure.set_facecolor("white")
    im_pressure = ax_pressure.imshow(
        pressure_plot,
        origin="lower",
        cmap="RdBu_r",
        vmin=global_min_pressure,
        vmax=global_max_pressure,
        extent=(0, cols - 1, 0, rows - 1),
        alpha=0.75,
        interpolation="bilinear",
    )
    ax_pressure.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
    ax_pressure.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
    ax_pressure.set(aspect=1, title="Pressure Field")
    ax_pressure.set_xlabel("X", fontsize=9)
    ax_pressure.set_ylabel("Y", fontsize=9)
    ax_pressure.set_xlim(0, cols - 1)
    ax_pressure.set_ylim(0, rows - 1)
    if "pressure" not in colorbars:
        colorbars["pressure"] = plt.colorbar(im_pressure, ax=ax_pressure, fraction=0.046, pad=0.04)
        colorbars["pressure"].set_label("Pa", fontsize=8)
    else:
        colorbars["pressure"].update_normal(im_pressure)

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
    ax_streamwise.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
    ax_streamwise.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
    ax_streamwise.set(aspect=1, title="Streamwise Velocity")
    ax_streamwise.set_xlabel("X", fontsize=9)
    ax_streamwise.set_ylabel("Y", fontsize=9)
    ax_streamwise.set_xlim(0, cols - 1)
    ax_streamwise.set_ylim(0, rows - 1)
    if "streamwise" not in colorbars:
        colorbars["streamwise"] = plt.colorbar(im_streamwise, ax=ax_streamwise, fraction=0.046, pad=0.04)
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
    ax_vortex.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
    ax_vortex.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
    ax_vortex.set(aspect=1, title="Vorticity Magnitude")
    ax_vortex.set_xlabel("X", fontsize=9)
    ax_vortex.set_ylabel("Y", fontsize=9)
    ax_vortex.set_xlim(0, cols - 1)
    ax_vortex.set_ylim(0, rows - 1)
    if "vortex" not in colorbars:
        colorbars["vortex"] = plt.colorbar(im_vortex, ax=ax_vortex, fraction=0.046, pad=0.04)
        colorbars["vortex"].set_label("1/s", fontsize=8)
    else:
        colorbars["vortex"].update_normal(im_vortex)

    # Bottom-middle: Drag/thrust/net force vs rocket velocity.
    history_velocity = np.array([item[0] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_total = np.array([item[1] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_pressure = np.array([item[2] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_shear = np.array([item[3] for item in drag_history[: frame_index + 1]], dtype=float)
    history_thrust = np.array([item[4] for item in drag_history[: frame_index + 1]], dtype=float)
    history_net = np.array([item[5] for item in drag_history[: frame_index + 1]], dtype=float)

    ax_drag.plot(history_velocity, history_drag_total, color="#1f77b4", linewidth=2.0, label="Total")
    ax_drag.plot(history_velocity, history_drag_pressure, color="#d62728", linewidth=1.3, linestyle="--", label="Pressure")
    ax_drag.plot(history_velocity, history_drag_shear, color="#2ca02c", linewidth=1.3, linestyle="--", label="Shear")
    ax_drag.plot(history_velocity, history_thrust, color="#9467bd", linewidth=1.5, linestyle=":", label="Thrust")
    ax_drag.plot(history_velocity, history_net, color="#ff7f0e", linewidth=1.5, linestyle="-.", label="Net (T-D-W)")
    ax_drag.scatter([frame_rocket_speed_mps], [drag_total], color="black", s=34, zorder=3)

    if history_velocity.size > 0:
        x_min = float(np.min(history_velocity))
        x_max = float(np.max(history_velocity))
        x_pad = max(0.05 * (x_max - x_min), 1e-3)
        ax_drag.set_xlim(x_min - x_pad, x_max + x_pad)
    else:
        ax_drag.set_xlim(0.0, 1.0)

    combined_forces = np.concatenate((history_drag_total, history_drag_pressure, history_drag_shear, history_thrust, history_net)) if history_drag_total.size > 0 else np.array([0.0])
    y_min = float(np.min(combined_forces))
    y_max = float(np.max(combined_forces))
    y_span = max(y_max - y_min, 1e-6)
    ax_drag.set_ylim(y_min - 0.1 * y_span, y_max + 0.1 * y_span)
    ax_drag.set_ylabel("Force (N)", fontsize=9)
    ax_drag.set_xlabel("Rocket Speed (m/s)", fontsize=9)
    ax_drag.grid(True, alpha=0.3)
    ax_drag.set_title("Forces vs Velocity", fontsize=10)
    ax_drag.legend(fontsize=8, loc="best")

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
    ax_shear.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
    ax_shear.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
    ax_shear.set(aspect=1, title="Wall Shear Stress Magnitude")
    ax_shear.set_xlabel("X", fontsize=9)
    ax_shear.set_ylabel("Y", fontsize=9)
    ax_shear.set_xlim(0, cols - 1)
    ax_shear.set_ylim(0, rows - 1)
    if "shear" not in colorbars:
        colorbars["shear"] = plt.colorbar(im_shear, ax=ax_shear, fraction=0.046, pad=0.04)
        colorbars["shear"].set_label("Pa", fontsize=8)
    else:
        colorbars["shear"].update_normal(im_shear)
        
    combined_speed = frame_speed + frame_rocket_speed_mps

    # Main title with time info
    fig.suptitle(
        f"Rocket-Frame CFD Analysis | t={frame_time:.2f}s | U∞={frame_speed:.2f} (solver) | Vrocket={frame_rocket_speed_mps:.1f} m/s | Combined Speed={combined_speed:.2f} m/s",
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
plt.show()
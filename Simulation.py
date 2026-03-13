from Classes import RocketProfile, NoseconeModel
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider
from FluidSimulation import FluidSimulation
from Graphing import format_duration, update_progress_bar

# Physics and display cadence are decoupled:
# - small physics dt keeps advection/projection stable
# - larger frame interval keeps animation length manageable
time_step = 0.1
sim_time = 64.0
frame_interval = 1.0
steps_per_frame = max(int(round(frame_interval / time_step)), 1)
num_frames = int(sim_time / frame_interval)


# Initialise the rocket profile
# width = diameter (y), height = axial length (x) — both in grid cells.
# At 400×400, width=40/height=112 preserves the same proportions as before
# (~28% domain length, 10% blockage) with much sharper boundary resolution.
rocket = RocketProfile(name="Test Rocket", mass=1000, thrust=5000, burn_time=120, width=40.0, height=112.0)

def ambient_velocity_profile(time_s: float) -> np.ndarray:
    # Ambient wind in rocket frame should be small at liftoff; keep this mild so
    # initial relative speed does not start artificially high.
    base_wind = 0.8
    gust = 0.25 * np.sin(0.28 * time_s) + 0.15 * np.sin(0.92 * time_s + 0.35)
    return np.array([-(base_wind + gust), 0.0], dtype=float)


def rocket_velocity_profile(time_s: float) -> np.ndarray:
    # Artemis-II-like ascent speed envelope (approximate m/s knots), compressed
    # in demo-time so acceleration is visible within a short 12 s run.
    time_knots_s = np.array([0.0, 10.0, 30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 480.0], dtype=float)
    speed_knots_mps = np.array([0.0, 110.0, 420.0, 950.0, 1500.0, 2100.0, 3500.0, 5200.0, 7800.0], dtype=float)

    # Compress physical ascent timeline into demo runtime.
    demo_time_scale = 3.0
    profile_time = min(float(time_s) * demo_time_scale, float(time_knots_s[-1]))
    physical_speed_mps = float(np.interp(profile_time, time_knots_s, speed_knots_mps))

    # Convert to solver units. Larger than before so acceleration is obvious.
    sim_speed_scale = 1.0 / 120.0
    speed = physical_speed_mps * sim_speed_scale

    pitch = 0.07 * np.sin(0.09 * time_s)
    return np.array([speed * np.cos(pitch), speed * np.sin(pitch)], dtype=float)


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

    # Streamwise velocity component (u in rocket frame)
    streamwise_velocity = sim.u.copy()

    # Wall shear stress magnitude: compute velocity gradients and shear
    # τ_xy = μ * (du/dy + dv/dx), τ = sqrt(τ_xx^2 + τ_xy^2 + τ_yy^2)
    # For visualization, use simplified magnitude: μ * speed_gradient
    dudx = np.zeros_like(sim.u)
    dvdy = np.zeros_like(sim.v)
    dudy_shear = np.zeros_like(sim.u)
    dvdx_shear = np.zeros_like(sim.v)
    
    dudx[1:-1, 1:-1] = (sim.u[1:-1, 2:] - sim.u[1:-1, :-2]) * 0.5
    dvdy[1:-1, 1:-1] = (sim.v[2:, 1:-1] - sim.v[:-2, 1:-1]) * 0.5
    dudy_shear[1:-1, 1:-1] = (sim.u[2:, 1:-1] - sim.u[:-2, 1:-1]) * 0.5
    dvdx_shear[1:-1, 1:-1] = (sim.v[1:-1, 2:] - sim.v[1:-1, :-2]) * 0.5
    
    dynamic_viscosity = max(sim.viscosity * sim.density, 1e-8)
    # Shear stress magnitude: sqrt(τ_xy^2 + τ_yy^2) near boundary
    shear_stress = dynamic_viscosity * np.sqrt((dudy_shear + dvdx_shear)**2 + dvdy**2)

    if getattr(sim, "wall_distance", None) is None:
        sim._update_wall_geometry()

    obstacle = sim.obstacle_mask
    wall_distance = np.array(sim.wall_distance, dtype=float) if sim.wall_distance is not None else np.full_like(obstacle, np.nan, dtype=float)
    surface_band = (~obstacle) & np.isfinite(wall_distance) & (wall_distance > 0.0) & (wall_distance <= 1.5)

    total_drag = 0.0
    pressure_drag = 0.0
    shear_drag = 0.0

    if np.any(surface_band):
        nx = -sim.wall_normal_x[surface_band]
        ny = -sim.wall_normal_y[surface_band]
        normal_mag = np.sqrt(nx**2 + ny**2)
        valid = normal_mag > 1e-8
        if np.any(valid):
            nx = nx[valid] / normal_mag[valid]
            ny = ny[valid] / normal_mag[valid]

            p_local = pressure[surface_band][valid]

            # Pressure traction on body: -p * n_out
            pressure_force_x = -p_local * nx
            pressure_force_y = -p_local * ny

            tx = -ny
            ty = nx
            u_local = sim.u[surface_band][valid]
            v_local = sim.v[surface_band][valid]
            tangential_velocity = u_local * tx + v_local * ty

            normal_distance = np.maximum(wall_distance[surface_band][valid], 1.0)
            tangential_shear = dynamic_viscosity * tangential_velocity / normal_distance

            # Shear force on body opposes local tangential fluid motion.
            shear_force_x = -tangential_shear * tx
            shear_force_y = -tangential_shear * ty

            force_x = np.sum(pressure_force_x + shear_force_x)
            force_y = np.sum(pressure_force_y + shear_force_y)

            pressure_force_x_total = np.sum(pressure_force_x)
            pressure_force_y_total = np.sum(pressure_force_y)
            shear_force_x_total = np.sum(shear_force_x)
            shear_force_y_total = np.sum(shear_force_y)

            drag_direction = sim.freestream_direction
            total_drag = float(force_x * drag_direction[0] + force_y * drag_direction[1])
            pressure_drag = float(pressure_force_x_total * drag_direction[0] + pressure_force_y_total * drag_direction[1])
            shear_drag = float(shear_force_x_total * drag_direction[0] + shear_force_y_total * drag_direction[1])

    return speed, vorticity, total_drag, pressure_drag, shear_drag, pressure, streamwise_velocity, shear_stress

# Initialise the fluid simulation
grid_size = (300, 600)  # Height x Width
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

# Small symmetry-breaking perturbation so vortex shedding can develop.
_rng = np.random.default_rng(seed=42)
_amp = 0.025 * freestream_speed
simulation.u += _amp * _rng.standard_normal(grid_size)
simulation.v += _amp * _rng.standard_normal(grid_size)
simulation.u[simulation.obstacle_mask] = 0.0
simulation.v[simulation.obstacle_mask] = 0.0

rows, cols = grid_size

rocket_center_x = grid_size[1] // 2
rocket_center_y = grid_size[0] // 2
profile_x, profile_y = rocket.get_2d_profile_polygon(
    center_x=rocket_center_x,
    center_y=rocket_center_y,
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
    simulation.simulate_particles(steps=steps_per_frame)
    dx = simulation.u.copy()
    dy = simulation.v.copy()
    speed, vorticity, drag_total, drag_pressure, drag_shear, pressure, streamwise_velocity, shear_stress = compute_diagnostics(simulation)
    frame_time = simulation.simulation_time
    frame_speed = simulation.edge_speed

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

    frames.append((dx, dy, speed_plot, vorticity_plot, pressure_plot, streamwise_plot, shear_plot, frame_time, frame_speed, drag_total, drag_pressure, drag_shear))
    drag_history.append((frame_speed, drag_total, drag_pressure, drag_shear))
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
    
    dx, dy, speed_plot, vorticity_plot, pressure_plot, streamwise_plot, shear_plot, frame_time, frame_speed, drag_total, drag_pressure, drag_shear = frames[frame_index]

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
    ax_streamwise.set(aspect=1, title="Streamwise Velocity (u)")
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

    # Bottom-middle: Actual drag vs freestream velocity.
    history_velocity = np.array([item[0] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_total = np.array([item[1] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_pressure = np.array([item[2] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag_shear = np.array([item[3] for item in drag_history[: frame_index + 1]], dtype=float)

    ax_drag.plot(history_velocity, history_drag_total, color="#1f77b4", linewidth=2.0, label="Total")
    ax_drag.plot(history_velocity, history_drag_pressure, color="#d62728", linewidth=1.3, linestyle="--", label="Pressure")
    ax_drag.plot(history_velocity, history_drag_shear, color="#2ca02c", linewidth=1.3, linestyle="--", label="Shear")
    ax_drag.scatter([frame_speed], [drag_total], color="black", s=34, zorder=3)

    if history_velocity.size > 0:
        x_min = float(np.min(history_velocity))
        x_max = float(np.max(history_velocity))
        x_pad = max(0.05 * (x_max - x_min), 1e-3)
        ax_drag.set_xlim(x_min - x_pad, x_max + x_pad)
    else:
        ax_drag.set_xlim(0.0, 1.0)

    combined_drag = np.concatenate((history_drag_total, history_drag_pressure, history_drag_shear)) if history_drag_total.size > 0 else np.array([0.0])
    y_min = float(np.min(combined_drag))
    y_max = float(np.max(combined_drag))
    y_span = max(y_max - y_min, 1e-6)
    ax_drag.set_ylim(y_min - 0.1 * y_span, y_max + 0.1 * y_span)
    ax_drag.set_ylabel("Drag (solver units)", fontsize=9)
    ax_drag.set_xlabel("Freestream Velocity |U∞|", fontsize=9)
    ax_drag.grid(True, alpha=0.3)
    ax_drag.set_title("Actual Drag vs Velocity", fontsize=10)
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

    # Main title with time info
    fig.suptitle(f"Rocket-Frame CFD Analysis | t={frame_time:.2f}s | U∞={frame_speed:.2f} m/s", fontsize=12, fontweight="bold")

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
from Classes import RocketProfile, NoseconeModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from FluidSimulation import FluidSimulation
from Functions import calculate_reynolds_number, calculate_drag_coefficient


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
    demo_time_scale = 8.0
    profile_time = min(float(time_s) * demo_time_scale, float(time_knots_s[-1]))
    physical_speed_mps = float(np.interp(profile_time, time_knots_s, speed_knots_mps))

    # Convert to solver units. Larger than before so acceleration is obvious.
    sim_speed_scale = 1.0 / 70.0
    speed = physical_speed_mps * sim_speed_scale

    pitch = 0.07 * np.sin(0.09 * time_s)
    return np.array([speed * np.cos(pitch), speed * np.sin(pitch)], dtype=float)


def compute_diagnostics(sim: FluidSimulation):
    speed = np.hypot(sim.u, sim.v)

    dvdx = np.zeros_like(sim.v)
    dudy = np.zeros_like(sim.u)
    dvdx[1:-1, 1:-1] = (sim.v[1:-1, 2:] - sim.v[1:-1, :-2]) * 0.5
    dudy[1:-1, 1:-1] = (sim.u[2:, 1:-1] - sim.u[:-2, 1:-1]) * 0.5
    vorticity = dvdx - dudy

    # Drag proxy from freestream dynamic pressure + Reynolds-based Cd trend.
    # This tracks acceleration much better than near-wall speed (which can be
    # strongly damped by no-slip enforcement and appear to "decay to zero").
    characteristic_length = max(float(getattr(sim.rocket_profile, "height", 1.0)), 1e-6)
    dynamic_viscosity = max(sim.viscosity * sim.density, 1e-8)
    reynolds_number = calculate_reynolds_number(
        density=sim.density,
        velocity=max(sim.edge_speed, 1e-6),
        characteristic_length=characteristic_length,
        viscosity=dynamic_viscosity,
    )
    drag_coefficient = calculate_drag_coefficient(reynolds_number)
    projected_area = max(float(getattr(sim.rocket_profile, "width", 1.0)), 1.0)
    drag_proxy = float(0.5 * sim.density * (sim.edge_speed ** 2) * drag_coefficient * projected_area)

    return speed, vorticity, drag_proxy

# Initialise the fluid simulation
grid_size = (400, 400)  # 4× per axis from 100×100 for much finer boundary/wake detail
viscosity = 0.006  # Higher-Re flow so wake rolls up instead of diffusing out
time_step = 0.1  # Time step
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
    turbulence_strength=0.04,  # lower SGS damping so wake structures survive
    vorticity_confinement=0.08,
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
num_frames = 520
total_time = num_frames * time_step
obstacle = simulation.obstacle_mask
frames = []
drag_history = []

global_min = np.inf
global_max = -np.inf

for _ in range(num_frames):
    simulation.simulate_particles()
    dx = simulation.u.copy()
    dy = simulation.v.copy()
    speed, vorticity, drag_proxy = compute_diagnostics(simulation)
    frame_time = simulation.simulation_time
    frame_speed = simulation.edge_speed

    dx[obstacle] = 0.0
    dy[obstacle] = 0.0
    speed_plot = speed.copy()
    speed_plot[obstacle] = np.nan
    vorticity_plot = np.abs(vorticity)
    vorticity_plot[obstacle] = np.nan

    finite_values = speed_plot[np.isfinite(speed_plot)]
    if finite_values.size > 0:
        global_min = min(global_min, float(np.min(finite_values)))
        global_max = max(global_max, float(np.max(finite_values)))

    frames.append((dx, dy, speed_plot, vorticity_plot, frame_time, frame_speed, drag_proxy))
    drag_history.append((frame_time, drag_proxy))

if not np.isfinite(global_min) or not np.isfinite(global_max):
    global_min, global_max = 0.0, 1.0

max_dim = max(rows, cols)
fig_width = 8.0 * cols / max_dim
fig_height = 9.6 * rows / max_dim
fig, (ax, drag_ax) = plt.subplots(
    2,
    1,
    figsize=(fig_width, fig_height),
    gridspec_kw={"height_ratios": [4.5, 1.6]},
)
fig.patch.set_facecolor("white")
plt.subplots_adjust(bottom=0.16, right=0.86)
cbar_ax = fig.add_axes((0.88, 0.18, 0.025, 0.70))
vector_colorbar = None


def draw_frame(frame_index):
    global vector_colorbar
    ax.clear()
    drag_ax.clear()
    ax.set_facecolor("white")
    dx, dy, speed_plot, vorticity_plot, frame_time, frame_speed, drag_proxy = frames[frame_index]

    field = ax.imshow(
        speed_plot,
        origin="lower",
        cmap="viridis",
        vmin=global_min,
        vmax=global_max,
        extent=(0, cols - 1, 0, rows - 1),
        alpha=0.65,
        interpolation="bilinear",
    )

    speed = np.hypot(dx, dy)
    flow_mask = speed > max(0.08 * frame_speed, 1e-4)
    display_u = np.where(flow_mask, dx, np.nan)
    display_v = np.where(flow_mask, dy, np.nan)
    display_u_q = display_u[::quiver_stride, ::quiver_stride]
    display_v_q = display_v[::quiver_stride, ::quiver_stride]

    quiver = ax.quiver(
        Xq,
        Yq,
        display_u_q,
        display_v_q,
        speed_plot[::quiver_stride, ::quiver_stride],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=28,
        width=0.003,
        pivot="mid",
        minlength=0.0,
    )

    quiver.set_clim(global_min, global_max)
    if vector_colorbar is None:
        vector_colorbar = fig.colorbar(field, cax=cbar_ax)
        vector_colorbar.set_label("Velocity Magnitude")
    else:
        vector_colorbar.update_normal(field)

    ax.contour(
        X,
        Y,
        vorticity_plot,
        levels=6,
        colors="white",
        linewidths=0.45,
        alpha=0.28,
    )

    ax.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
    ax.plot(profile_x, profile_y, color="blue", linewidth=2, zorder=7)
    ax.scatter([rocket_center_x], [rocket_center_y], color="red", s=45, zorder=8)

    ax.set(aspect=1, title=f"Unsteady Rocket-Frame Flow | t={frame_time:.2f}s | U∞={frame_speed:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, cols - 1)
    ax.set_ylim(0, rows - 1)

    history_times = np.array([item[0] for item in drag_history[: frame_index + 1]], dtype=float)
    history_drag = np.array([item[1] for item in drag_history[: frame_index + 1]], dtype=float)
    drag_ax.plot(history_times, history_drag, color="#1f77b4", linewidth=2.0)
    drag_ax.scatter([frame_time], [drag_proxy], color="red", s=26, zorder=3)
    drag_ax.set_xlim(0.0, max(total_time, 1e-6))
    if history_drag.size > 0:
        y_max = max(float(np.max(history_drag)) * 1.15, 1e-6)
    else:
        y_max = 1.0
    drag_ax.set_ylim(0.0, y_max)
    drag_ax.set_ylabel("Drag Proxy")
    drag_ax.set_xlabel("Time (s)")
    drag_ax.grid(True, alpha=0.3)

draw_frame(0)

slider_ax = fig.add_axes((0.16, 0.05, 0.62, 0.03))
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
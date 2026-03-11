from Classes import RocketProfile
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from FluidSimulation import FluidSimulation


def build_adaptive_axis(size, center, refine_radius=8.0, refine_factor=4):
    base = np.arange(size, dtype=float)
    left = max(0, int(np.floor(center - refine_radius)))
    right = min(size - 1, int(np.ceil(center + refine_radius)))

    if right <= left:
        return base

    fine_count = max((right - left) * refine_factor + 1, 2)
    fine = np.linspace(left, right, fine_count)
    axis = np.unique(np.concatenate([base[:left], fine, base[right + 1:]]))
    return axis


def sample_field_bilinear(field, x_coords, y_coords):
    rows, cols = field.shape
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    x0 = np.floor(x_grid).astype(int)
    y0 = np.floor(y_grid).astype(int)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)
    x0 = np.clip(x0, 0, cols - 1)
    y0 = np.clip(y0, 0, rows - 1)

    dx = x_grid - x0
    dy = y_grid - y0

    top_left = field[y0, x0]
    top_right = field[y0, x1]
    bottom_left = field[y1, x0]
    bottom_right = field[y1, x1]

    return (
        top_left * (1 - dx) * (1 - dy)
        + top_right * dx * (1 - dy)
        + bottom_left * (1 - dx) * dy
        + bottom_right * dx * dy
    )


def build_normalized_quiver_trace(u_field, v_field, x_coords, y_coords, stride=4, scale=0.8):
    x_quiver = x_coords[::stride]
    y_quiver = y_coords[::stride]

    u_sample = sample_field_bilinear(u_field, x_quiver, y_quiver)
    v_sample = sample_field_bilinear(v_field, x_quiver, y_quiver)
    speed = np.sqrt(u_sample**2 + v_sample**2)

    u_unit = np.divide(u_sample, speed, out=np.zeros_like(u_sample), where=speed > 1e-8)
    v_unit = np.divide(v_sample, speed, out=np.zeros_like(v_sample), where=speed > 1e-8)

    x_grid, y_grid = np.meshgrid(x_quiver, y_quiver)
    quiver_fig = ff.create_quiver(
        x_grid.ravel(),
        y_grid.ravel(),
        u_unit.ravel(),
        v_unit.ravel(),
        scale=scale,
        arrow_scale=0.35,
        angle=np.pi / 9,
        line_width=1,
        name="Direction Field"
    )
    quiver_trace = quiver_fig.data[0]
    quiver_trace.update(line=dict(color="white", width=1), showlegend=False)
    return quiver_trace

# Initialize the rocket profile
rocket = RocketProfile(name="Test Rocket", mass=1000, thrust=5000, burn_time=120, width=2.0, height=5.0)

# Initialize the fluid simulation
grid_size = (50, 50)  # 50x50 grid
viscosity = 0.1  # Kinematic viscosity
time_step = 0.01  # Time step
density = 1.225  # Air density
acceleration = 2 * 9.8  # Magnitude in m/s^2
acceleration_direction = (1.0, 0.0)  # (x, y) direction; examples: (0, 1), (-1, 0), (1, 1)
simulation = FluidSimulation(
    grid_size,
    viscosity,
    time_step,
    density,
    rocket_profile=rocket,
    acceleration=acceleration,
    acceleration_direction=acceleration_direction,
)

rocket_center_x = grid_size[1] // 2
rocket_center_y = grid_size[0] // 2
profile_x, profile_y = rocket.get_2d_profile_polygon(
    center_x=rocket_center_x,
    center_y=rocket_center_y,
)

adaptive_x = build_adaptive_axis(grid_size[1], rocket_center_x, refine_radius=10.0, refine_factor=5)
adaptive_y = build_adaptive_axis(grid_size[0], rocket_center_y, refine_radius=10.0, refine_factor=5)

# Simulate particles accelerating at 2g
simulation.simulate_particles()

# Simulate and plot with time slider
frames = []
time_steps = np.linspace(0, 10, 100)  # Simulate for 10 seconds with 100 steps

for t in time_steps:
    simulation.simulate_particles()
    speed = np.sqrt(simulation.u**2 + simulation.v**2)
    speed_adaptive = sample_field_bilinear(speed, adaptive_x, adaptive_y)
    quiver_trace = build_normalized_quiver_trace(simulation.u, simulation.v, adaptive_x, adaptive_y)
    frame = go.Frame(
        data=[
            go.Contour(
                z=speed_adaptive,
                x=adaptive_x,
                y=adaptive_y,
                colorscale="Viridis",
                colorbar=dict(title="Velocity Magnitude")
            ),
            quiver_trace,
            go.Scatter(
                x=profile_x,
                y=profile_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(0, 102, 255, 0.35)",
                line=dict(color="blue", width=2),
                name="Rocket Profile"
            ),
            go.Scatter(
                x=[rocket_center_x],
                y=[rocket_center_y],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Rocket Center"
            )
        ],
        name=f"Time {t:.2f}s"
    )
    frames.append(frame)

fig = go.Figure(
    data=[
        go.Contour(
            z=sample_field_bilinear(np.sqrt(simulation.u**2 + simulation.v**2), adaptive_x, adaptive_y),
            x=adaptive_x,
            y=adaptive_y,
            colorscale="Viridis",
            colorbar=dict(title="Velocity Magnitude")
        ),
        build_normalized_quiver_trace(simulation.u, simulation.v, adaptive_x, adaptive_y),
        go.Scatter(
            x=profile_x,
            y=profile_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(0, 102, 255, 0.35)",
            line=dict(color="blue", width=2),
            name="Rocket Profile"
        ),
        go.Scatter(
            x=[rocket_center_x],
            y=[rocket_center_y],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Rocket Center"
        ),
    ],
    layout=go.Layout(
        title="Velocity Field with Rocket Profile Over Time",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        sliders=[{
            "steps": [
                {
                    "args": [[frame.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                    "label": frame.name,
                    "method": "animate"
                } for frame in frames
            ]
        }]
    ),
    frames=frames
)

fig.show()
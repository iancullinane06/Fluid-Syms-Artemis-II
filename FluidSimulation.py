import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.ndimage import map_coordinates

from Functions import calculate_reynolds_number, calculate_drag_coefficient

class FluidSimulation:
    """
    A 2D fluid simulation using a simplified Navier-Stokes solver.
    """
    def __init__(
        self,
        grid_size,
        viscosity,
        time_step,
        density,
        rocket_profile=None,
        acceleration=0.0,
        acceleration_direction=(1.0, 0.0),
        assume_laminar_edges=True,
        edge_relaxation=0.85,
        edge_speed=None,
        wall_penalty=24.0,
        wall_shear_layers=4,
        turbulence_strength=0.18,
        vorticity_confinement=0.12,
        ambient_velocity_profile: Callable[[float], np.ndarray] | None = None,
        rocket_velocity_profile: Callable[[float], np.ndarray] | None = None,
        inflow_blend=0.03,
    ):
        """
        Initialize the fluid simulation.

        Parameters:
            grid_size (tuple): Grid size (nx, ny).
            viscosity (float): Kinematic viscosity (m^2/s).
            time_step (float): Time step (s).
            density (float): Fluid density (kg/m^3).
            rocket_profile (RocketProfile, optional): Rocket profile object.
        """
        self.grid_size = grid_size  # (nx, ny)
        self.viscosity = viscosity  # Kinematic viscosity (m^2/s)
        self.time_step = time_step  # Time step (s)
        self.density = density  # Fluid density (kg/m^3)
        self.obstacle_mask = np.zeros(grid_size, dtype=bool)
        self.acceleration = acceleration  # Acceleration of particles (m/s^2)
        self.rocket_profile = rocket_profile
        self.acceleration_direction = self._normalize_direction(acceleration_direction)
        self.freestream_direction = self.acceleration_direction.copy()
        self.assume_laminar_edges = bool(assume_laminar_edges)
        self.edge_relaxation = float(np.clip(edge_relaxation, 0.0, 1.0))
        self.edge_speed = 0.0 if edge_speed is None else float(max(edge_speed, 0.0))
        self.wall_penalty = float(max(wall_penalty, 0.0))
        self.wall_shear_layers = max(int(wall_shear_layers), 1)
        self.turbulence_strength = float(max(turbulence_strength, 0.0))
        self.vorticity_confinement = float(max(vorticity_confinement, 0.0))
        self.simulation_time = 0.0
        self.ambient_velocity_profile = ambient_velocity_profile
        self.rocket_velocity_profile = rocket_velocity_profile
        self.inflow_blend = float(np.clip(inflow_blend, 0.0, 1.0))
        self.base_ambient_velocity = self.edge_speed * self.freestream_direction
        self.relative_velocity_target = self.base_ambient_velocity.copy()
        self.reference_acceleration_vector = np.zeros(2, dtype=float)
        self.wall_distance = None
        self.wall_normal_x = np.zeros(grid_size, dtype=float)
        self.wall_normal_y = np.zeros(grid_size, dtype=float)

        # Initialize velocity and pressure fields
        self.u = np.zeros(grid_size)  # x-velocity
        self.v = np.zeros(grid_size)  # y-velocity
        self.p = np.zeros(grid_size)  # Pressure field

        if self.rocket_profile is not None:
            self.add_rocket_profile(self.rocket_profile)

    def _normalize_direction(self, direction):
        """Normalize a 2D direction vector."""
        direction_array = np.asarray(direction, dtype=float)
        if direction_array.shape != (2,):
            raise ValueError("acceleration_direction must be a 2-element vector (x, y)")

        magnitude = np.linalg.norm(direction_array)
        if magnitude < 1e-12:
            raise ValueError("acceleration_direction cannot be a zero vector")

        return direction_array / magnitude

    def set_acceleration_direction(self, direction):
        """Update acceleration direction using a 2D vector (x, y)."""
        self.acceleration_direction = self._normalize_direction(direction)

    def set_uniform_flow(self, speed, direction=None):
        """Initialize/overwrite the fluid with a uniform freestream velocity field."""
        if direction is None:
            direction_vector = self.freestream_direction
        else:
            direction_vector = self._normalize_direction(direction)

        flow_speed = float(max(speed, 0.0))
        self.freestream_direction = direction_vector
        self.edge_speed = flow_speed
        self.base_ambient_velocity = flow_speed * direction_vector
        self.relative_velocity_target = self.base_ambient_velocity.copy()
        self.reference_acceleration_vector.fill(0.0)
        self.u[:, :] = flow_speed * float(direction_vector[0])
        self.v[:, :] = flow_speed * float(direction_vector[1])
        self._enforce_obstacle_boundary()
        self._enforce_domain_boundary()

    def _resolve_profile_velocity(self, profile, time_s, default_velocity):
        """Evaluate a velocity profile callback if present, otherwise return default velocity."""
        if profile is None:
            return np.asarray(default_velocity, dtype=float)

        candidate = np.asarray(profile(float(time_s)), dtype=float)
        if candidate.shape != (2,):
            raise ValueError("Velocity profile must return a 2-element vector (vx, vy)")
        return candidate

    def _update_reference_frame(self):
        """Update freestream from ambient flow and rocket motion in a rocket-fixed frame."""
        previous_relative_velocity = self.relative_velocity_target.copy()
        ambient_velocity = self._resolve_profile_velocity(
            self.ambient_velocity_profile,
            self.simulation_time,
            self.base_ambient_velocity,
        )
        rocket_velocity = self._resolve_profile_velocity(
            self.rocket_velocity_profile,
            self.simulation_time,
            np.zeros(2, dtype=float),
        )
        relative_velocity = ambient_velocity - rocket_velocity
        speed = float(np.linalg.norm(relative_velocity))
        self.relative_velocity_target = relative_velocity.copy()

        dt = max(float(self.time_step), 1e-8)
        self.reference_acceleration_vector = (self.relative_velocity_target - previous_relative_velocity) / dt

        if speed > 1e-12:
            self.freestream_direction = relative_velocity / speed
            self.edge_speed = speed
            # Do NOT blend interior cells — only update boundary conditions (done via
            # _enforce_domain_boundary which reads self.edge_speed/freestream_direction).
            # Blending every cell each step injects uniform divergence that looks like
            # source/sink nodes all over the domain.

    def add_source(self, x, y, strength):
        """
        Add a velocity source at a specific location.
        """
        self.u[y, x] += strength[0]
        self.v[y, x] += strength[1]

    def add_rocket_profile(self, rocket_profile=None, center=None, body_fraction=0.65, nose_points=128, mask=None):
        """
        Registers a solid obstacle mask from RocketProfile or direct mask.
        """
        if rocket_profile is not None:
            self.rocket_profile = rocket_profile

        if mask is not None:
            if mask.shape != self.grid_size:
                raise ValueError(f"mask shape {mask.shape} != grid_size {self.grid_size}")
            self.obstacle_mask = mask.astype(bool)
            self._update_wall_geometry()
            return self.obstacle_mask

        if self.rocket_profile is None:
            raise ValueError("No rocket_profile provided.")

        if center is None:
            center = (self.grid_size[1] // 2, self.grid_size[0] // 2)  # (x, y)

        cx, cy = center
        self.obstacle_mask = self.rocket_profile.get_2d_profile_mask(
            grid_size=self.grid_size,
            center_x=cx,
            center_y=cy,
            body_fraction=body_fraction,
            nose_points=nose_points
        )
        self._update_wall_geometry()
        return self.obstacle_mask

    def _update_wall_geometry(self):
        """Precompute wall-distance bands and obstacle normals for immersed-boundary damping."""
        obstacle = getattr(self, "obstacle_mask", None)
        if obstacle is None or not np.any(obstacle):
            self.wall_distance = np.full(self.grid_size, np.inf, dtype=float)
            self.wall_normal_x.fill(0.0)
            self.wall_normal_y.fill(0.0)
            return

        gradient_y, gradient_x = np.gradient(obstacle.astype(float))
        normal_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        valid = normal_magnitude > 1e-12

        self.wall_normal_x.fill(0.0)
        self.wall_normal_y.fill(0.0)
        self.wall_normal_x[valid] = gradient_x[valid] / normal_magnitude[valid]
        self.wall_normal_y[valid] = gradient_y[valid] / normal_magnitude[valid]

        wall_distance = np.full(self.grid_size, np.inf, dtype=float)
        wall_distance[obstacle] = 0.0
        shell = obstacle.copy()
        for layer in range(1, self.wall_shear_layers + 1):
            expanded = self._dilate_mask(shell)
            ring = expanded & ~shell
            wall_distance[ring & np.isinf(wall_distance)] = float(layer)
            shell = expanded

        self.wall_distance = wall_distance

    def _laplacian(self, field):
        """Discrete Laplacian with unit cell spacing."""
        laplacian = np.zeros_like(field)
        laplacian[1:-1, 1:-1] = (
            field[:-2, 1:-1]
            + field[2:, 1:-1]
            + field[1:-1, :-2]
            + field[1:-1, 2:]
            - 4.0 * field[1:-1, 1:-1]
        )
        return laplacian

    def _compute_vorticity(self):
        """Compute scalar vorticity for the 2D velocity field."""
        dvdx = np.zeros_like(self.v)
        dudy = np.zeros_like(self.u)
        dvdx[1:-1, 1:-1] = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) * 0.5
        dudy[1:-1, 1:-1] = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) * 0.5
        return dvdx - dudy

    def _apply_wall_shear(self):
        """Apply immersed-boundary wall damping that drives both normal and tangential velocity toward no-slip."""
        if getattr(self, "obstacle_mask", None) is None:
            return
        if not np.any(self.obstacle_mask):
            return

        obstacle = self.obstacle_mask
        penalty = np.clip(self.wall_penalty * self.time_step, 0.0, 1.0)
        self.u[obstacle] *= (1.0 - penalty)
        self.v[obstacle] *= (1.0 - penalty)

        if self.wall_distance is None:
            self._update_wall_geometry()
        wall_distance = self.wall_distance
        if wall_distance is None:
            return

        boundary_layer = np.isfinite(wall_distance) & (wall_distance > 0.0)
        if not np.any(boundary_layer):
            return

        local_speed = np.mean(np.hypot(self.u[boundary_layer], self.v[boundary_layer]))
        dynamic_viscosity = max(self.viscosity * self.density, 1e-8)
        characteristic_length = 1.0
        if self.rocket_profile is not None:
            characteristic_length = max(float(self.rocket_profile.width), 1e-6)

        reynolds_number = calculate_reynolds_number(
            density=self.density,
            velocity=max(local_speed, 1e-6),
            characteristic_length=characteristic_length,
            viscosity=dynamic_viscosity,
        )
        drag_coefficient = calculate_drag_coefficient(reynolds_number)

        distance = wall_distance[boundary_layer]
        wall_factor = 1.0 - np.clip((distance - 1.0) / max(self.wall_shear_layers, 1), 0.0, 1.0)
        damping = np.clip((0.18 + 0.12 * drag_coefficient) * self.time_step * wall_factor, 0.0, 0.85)

        nx = self.wall_normal_x[boundary_layer].copy()
        ny = self.wall_normal_y[boundary_layer].copy()
        normal_mag = np.sqrt(nx**2 + ny**2)
        fallback = normal_mag < 1e-8
        if np.any(fallback):
            nx[fallback] = -self.freestream_direction[0]
            ny[fallback] = -self.freestream_direction[1]
            normal_mag = np.sqrt(nx**2 + ny**2)

        nx = nx / np.maximum(normal_mag, 1e-8)
        ny = ny / np.maximum(normal_mag, 1e-8)
        tx = -ny
        ty = nx

        u_local = self.u[boundary_layer]
        v_local = self.v[boundary_layer]
        normal_velocity = u_local * nx + v_local * ny
        tangential_velocity = u_local * tx + v_local * ty

        normal_velocity *= (1.0 - np.clip(1.35 * damping, 0.0, 0.92))
        tangential_velocity *= (1.0 - np.clip(1.75 * damping, 0.0, 0.96))

        self.u[boundary_layer] = normal_velocity * nx + tangential_velocity * tx
        self.v[boundary_layer] = normal_velocity * ny + tangential_velocity * ty

    def _apply_turbulence_model(self):
        """Simple LES-style eddy-viscosity closure plus vorticity confinement for wake roll-up."""
        fluid = ~self.obstacle_mask
        if not np.any(fluid):
            return

        dudx = np.zeros_like(self.u)
        dudy = np.zeros_like(self.u)
        dvdx = np.zeros_like(self.v)
        dvdy = np.zeros_like(self.v)

        dudx[1:-1, 1:-1] = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) * 0.5
        dudy[1:-1, 1:-1] = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) * 0.5
        dvdx[1:-1, 1:-1] = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) * 0.5
        dvdy[1:-1, 1:-1] = (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) * 0.5

        strain = np.sqrt(2.0 * dudx**2 + 2.0 * dvdy**2 + (dudy + dvdx) ** 2)
        eddy_viscosity = (self.turbulence_strength ** 2) * strain
        wall_distance = self.wall_distance
        if wall_distance is not None:
            near_wall = np.isfinite(wall_distance)
            wall_damping = np.ones_like(eddy_viscosity)
            wall_damping[near_wall] = np.clip(wall_distance[near_wall] / max(self.wall_shear_layers, 1), 0.15, 1.0)
            eddy_viscosity *= wall_damping

        effective_viscosity = self.viscosity + eddy_viscosity
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        self.u[fluid] += self.time_step * effective_viscosity[fluid] * lap_u[fluid]
        self.v[fluid] += self.time_step * effective_viscosity[fluid] * lap_v[fluid]

        if self.vorticity_confinement <= 0.0:
            return

        vorticity = self._compute_vorticity()
        vorticity_magnitude = np.abs(vorticity)
        grad_y, grad_x = np.gradient(vorticity_magnitude)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag

        confine_u = ny * vorticity
        confine_v = -nx * vorticity
        self.u[fluid] += self.vorticity_confinement * self.time_step * confine_u[fluid]
        self.v[fluid] += self.vorticity_confinement * self.time_step * confine_v[fluid]

    def _enforce_obstacle_boundary(self):
        """
        Immersed-boundary style no-slip treatment with near-wall shear damping.
        """
        self._apply_wall_shear()

    def _dilate_mask(self, mask):
        """Single-cell dilation without periodic wraparound."""
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up_left = padded[:-2, :-2]
        up_right = padded[:-2, 2:]
        down_left = padded[2:, :-2]
        down_right = padded[2:, 2:]

        dilated = center | up | down | left | right | up_left | up_right | down_left | down_right
        return dilated

    def _apply_body_force(self):
        """Apply uniform body-force acceleration to fluid cells.

        Two sources are combined:
        1) User-configured constant acceleration (magnitude + direction)
        2) Reference-frame acceleration from d(u_ambient - u_rocket)/dt
        """
        fluid = ~self.obstacle_mask
        if not np.any(fluid):
            return

        acceleration_vector = self.reference_acceleration_vector.copy()
        if self.acceleration != 0.0:
            acceleration_vector += self.acceleration * self.acceleration_direction

        accel_step = acceleration_vector * self.time_step
        self.u[fluid] += accel_step[0]
        self.v[fluid] += accel_step[1]

    def _enforce_domain_boundary(self):
        """Outer-domain boundary model (laminar freestream or fallback no-slip)."""
        if not self.assume_laminar_edges:
            self.u[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.u[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
            self.v[:, 0] = 0.0
            self.v[:, -1] = 0.0
            return

        freestream_speed = self.edge_speed
        target_u = freestream_speed * self.freestream_direction[0]
        target_v = freestream_speed * self.freestream_direction[1]
        beta = self.edge_relaxation

        # Use a directional inlet/outlet plus slip boundaries on the cross-flow edges.
        #
        # Inlet  = the wall that faces UPSTREAM (fluid enters here) → prescribe freestream.
        # Outlet = the wall that faces DOWNSTREAM (fluid exits here) → zero-gradient
        #          (convective outflow) so nothing reflects back into the domain.
        #
        # For horizontal freestream:
        #   direction[0] < 0  → flow goes LEFT  → fluid enters RIGHT wall, exits LEFT wall.
        #   direction[0] > 0  → flow goes RIGHT → fluid enters LEFT wall, exits RIGHT wall.
        if abs(self.freestream_direction[0]) >= abs(self.freestream_direction[1]):
            # Dominant horizontal freestream.
            if self.freestream_direction[0] <= 0.0:
                # Flow is LEFTWARD → inlet on right, outlet (free-escape) on left.
                self.u[:, -1] = beta * target_u + (1.0 - beta) * self.u[:, -2]
                self.v[:, -1] = beta * target_v + (1.0 - beta) * self.v[:, -2]
                # Left wall: zero-gradient outflow — fluid escapes freely.
                self.u[:, 0] = self.u[:, 1]
                self.v[:, 0] = self.v[:, 1]
            else:
                # Flow is RIGHTWARD → inlet on left, outlet (free-escape) on right.
                self.u[:, 0] = beta * target_u + (1.0 - beta) * self.u[:, 1]
                self.v[:, 0] = beta * target_v + (1.0 - beta) * self.v[:, 1]
                # Right wall: zero-gradient outflow — fluid escapes freely.
                self.u[:, -1] = self.u[:, -2]
                self.v[:, -1] = self.v[:, -2]

            # Slip-like top/bottom: zero normal flow, zero-gradient tangential flow.
            self.u[0, :] = self.u[1, :]
            self.u[-1, :] = self.u[-2, :]
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
        else:
            # Dominant vertical freestream.
            if self.freestream_direction[1] >= 0.0:
                # Flow is UPWARD → inlet on bottom, outlet (free-escape) on top.
                self.u[0, :] = beta * target_u + (1.0 - beta) * self.u[1, :]
                self.v[0, :] = beta * target_v + (1.0 - beta) * self.v[1, :]
                self.u[-1, :] = self.u[-2, :]
                self.v[-1, :] = self.v[-2, :]
            else:
                # Flow is DOWNWARD → inlet on top, outlet (free-escape) on bottom.
                self.u[-1, :] = beta * target_u + (1.0 - beta) * self.u[-2, :]
                self.v[-1, :] = beta * target_v + (1.0 - beta) * self.v[-2, :]
                self.u[0, :] = self.u[1, :]
                self.v[0, :] = self.v[1, :]

            # Slip-like left/right: zero normal flow, zero-gradient tangential flow.
            self.u[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[:, 0] = self.v[:, 1]
            self.v[:, -1] = self.v[:, -2]

    def step(self):
        """
        Perform a single time step of the simulation.
        """
        self._update_reference_frame()
        self._apply_body_force()
        self._enforce_domain_boundary()   # seed inlet before advect
        self._enforce_obstacle_boundary()
        self._advect()
        self._enforce_domain_boundary()   # restore inlet after advect wipes boundaries to zero
        self._enforce_obstacle_boundary()
        self._diffuse()
        self._enforce_domain_boundary()   # re-support inlet before pressure solve
        self._enforce_obstacle_boundary()
        self._apply_turbulence_model()
        self._enforce_obstacle_boundary()
        self._project()
        self._enforce_obstacle_boundary()
        self._enforce_domain_boundary()   # final clean-up
        self.simulation_time += self.time_step

    def _advect(self):
        """
        Advect the velocity field using SciPy-backed semi-Lagrangian advection.

        Uses ndimage.map_coordinates (bilinear interpolation) over the full interior
        grid. This is both faster and less fragile at higher resolutions.
        Boundary cells are preserved from the current field; _enforce_domain_boundary
        is called after this method to impose inlet/outlet values.
        """
        rows, cols = self.grid_size

        j_idx, i_idx = np.mgrid[1:rows - 1, 1:cols - 1]
        fluid = ~self.obstacle_mask[1:-1, 1:-1]

        x_back = np.clip(i_idx - self.u[1:-1, 1:-1] * self.time_step, 0.0, cols - 1.0)
        y_back = np.clip(j_idx - self.v[1:-1, 1:-1] * self.time_step, 0.0, rows - 1.0)

        # map_coordinates expects coordinates in (row, col) order.
        sample_coords = np.vstack((y_back.ravel(), x_back.ravel()))
        u_interp = map_coordinates(self.u, sample_coords, order=1, mode="nearest").reshape(rows - 2, cols - 2)
        v_interp = map_coordinates(self.v, sample_coords, order=1, mode="nearest").reshape(rows - 2, cols - 2)

        u_new = self.u.copy()
        v_new = self.v.copy()

        u_new[1:-1, 1:-1] = np.where(fluid, u_interp, self.u[1:-1, 1:-1])
        v_new[1:-1, 1:-1] = np.where(fluid, v_interp, self.v[1:-1, 1:-1])

        self.u = u_new
        self.v = v_new

    def _diffuse(self):
        """
        Diffuse the velocity field using the diffusion equation.
        """
        alpha = self.viscosity * self.time_step
        for _ in range(20):  # Iterative solver
            self.u[1:-1, 1:-1] = (
                self.u[1:-1, 1:-1]
                + alpha
                * (
                    self.u[:-2, 1:-1]
                    + self.u[2:, 1:-1]
                    + self.u[1:-1, :-2]
                    + self.u[1:-1, 2:]
                )
            ) / (1 + 4 * alpha)

            self.v[1:-1, 1:-1] = (
                self.v[1:-1, 1:-1]
                + alpha
                * (
                    self.v[:-2, 1:-1]
                    + self.v[2:, 1:-1]
                    + self.v[1:-1, :-2]
                    + self.v[1:-1, 2:]
                )
            ) / (1 + 4 * alpha)

    def _project(self):
        """
        Enforce incompressibility by solving the pressure Poisson equation.

        Solves  ∇²p = ∇·u  using Gauss-Seidel, then subtracts ∇p to make the
        velocity field divergence-free.

        Gauss-Seidel form of  ∇²p = div  with unit cell spacing:
            p[i,j] = (neighbours − div[i,j]) / 4
        Note the MINUS sign: adding divergence would solve ∇²p = −∇·u and
        amplify rather than suppress divergence.
        """
        # Compute velocity divergence (right-hand side of Poisson equation).
        div = (
            self.u[1:-1, 2:] - self.u[1:-1, :-2]
            + self.v[2:, 1:-1] - self.v[:-2, 1:-1]
        ) / 2.0

        # Zero pressure everywhere before iterating (clean slate each step
        # avoids pressure accumulation artefacts).
        self.p.fill(0.0)

        # Gauss-Seidel iterations for  ∇²p = div.
        for _ in range(50):
            self.p[1:-1, 1:-1] = (
                self.p[:-2, 1:-1]
                + self.p[2:, 1:-1]
                + self.p[1:-1, :-2]
                + self.p[1:-1, 2:]
                - div               # ← correct sign
            ) / 4.0
            # Zero pressure on/inside the obstacle so it does not pollute the
            # gradient that is subtracted from fluid velocities below.
            self.p[self.obstacle_mask] = 0.0

        # Project velocity: u ← u − ∇p  (only fluid cells).
        fluid_inner = ~self.obstacle_mask[1:-1, 1:-1]
        dp_dx = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / 2.0
        dp_dy = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / 2.0
        self.u[1:-1, 1:-1] -= np.where(fluid_inner, dp_dx, 0.0)
        self.v[1:-1, 1:-1] -= np.where(fluid_inner, dp_dy, 0.0)

    def _bilinear_interpolation(self, field, x, y, x0, y0, x1, y1):
        """
        Perform bilinear interpolation for advection.
        """
        return (
            field[y0, x0] * (x1 - x) * (y1 - y)
            + field[y1, x0] * (x1 - x) * (y - y0)
            + field[y0, x1] * (x - x0) * (y1 - y)
            + field[y1, x1] * (x - x0) * (y - y0)
        )

    def simulate_particles(self, steps: int = 1) -> None:
        """
        Run the particle/fluid simulation for a number of steps.
        Keeps Main.py API stable.
        """
        for _ in range(steps):
            self.step()
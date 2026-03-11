import numpy as np
import matplotlib.pyplot as plt
from Classes import RocketProfile
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

    def add_source(self, x, y, strength):
        """
        Add a velocity source at a specific location.
        """
        self.u[y, x] += strength[0]
        self.v[y, x] += strength[1]

    def add_rocket_profile(self, rocket_profile=None, center=None, body_fraction=0.65, nose_points=24, mask=None):
        """
        Registers a solid obstacle mask from RocketProfile or direct mask.
        """
        if rocket_profile is not None:
            self.rocket_profile = rocket_profile

        if mask is not None:
            if mask.shape != self.grid_size:
                raise ValueError(f"mask shape {mask.shape} != grid_size {self.grid_size}")
            self.obstacle_mask = mask.astype(bool)
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
        return self.obstacle_mask

    def _enforce_obstacle_boundary(self):
        """
        No-slip on rocket body and Reynolds/drag-based damping near the wall.
        """
        if getattr(self, "obstacle_mask", None) is None:
            return
        if not np.any(self.obstacle_mask):
            return

        obstacle = self.obstacle_mask

        # Strict no-slip in the solid body.
        self.u[obstacle] = 0.0
        self.v[obstacle] = 0.0

        wall = self._dilate_mask(obstacle) & ~obstacle
        self.u[wall] = 0.0
        self.v[wall] = 0.0

        # Reynolds/drag-based damping in the first fluid ring outside the no-slip wall.
        boundary_layer = self._dilate_mask(wall) & ~wall & ~obstacle
        if np.any(boundary_layer):
            speed = np.sqrt(self.u**2 + self.v**2)
            local_speed = np.mean(speed[boundary_layer])

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

            damping = np.clip(self.time_step * drag_coefficient, 0.0, 0.95)
            self.u[boundary_layer] *= (1.0 - damping)
            self.v[boundary_layer] *= (1.0 - damping)

    def _dilate_mask(self, mask):
        """Single-cell 8-neighborhood mask dilation without external dependencies."""
        up = np.roll(mask, 1, axis=0)
        down = np.roll(mask, -1, axis=0)
        left = np.roll(mask, 1, axis=1)
        right = np.roll(mask, -1, axis=1)
        up_left = np.roll(up, 1, axis=1)
        up_right = np.roll(up, -1, axis=1)
        down_left = np.roll(down, 1, axis=1)
        down_right = np.roll(down, -1, axis=1)
        dilated = mask | up | down | left | right | up_left | up_right | down_left | down_right
        dilated[0, :] = False
        dilated[-1, :] = False
        dilated[:, 0] = False
        dilated[:, -1] = False
        return dilated

    def _apply_body_force(self):
        """Apply uniform acceleration to fluid cells along configured direction."""
        if self.acceleration == 0.0:
            return
        fluid = ~self.obstacle_mask
        accel_step = self.acceleration * self.time_step
        self.u[fluid] += accel_step * self.acceleration_direction[0]
        self.v[fluid] += accel_step * self.acceleration_direction[1]

    def _enforce_domain_boundary(self):
        """Simple no-slip at outer domain boundaries."""
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

    def step(self):
        """
        Perform a single time step of the simulation.
        """
        self._apply_body_force()
        self._enforce_obstacle_boundary()
        self._advect()
        self._enforce_obstacle_boundary()
        self._diffuse()
        self._enforce_obstacle_boundary()
        self._project()
        self._enforce_obstacle_boundary()
        self._enforce_domain_boundary()

    def _advect(self):
        """
        Advect the velocity field using semi-Lagrangian advection.
        """
        rows, cols = self.grid_size
        u_new = np.zeros_like(self.u)
        v_new = np.zeros_like(self.v)

        for j in range(1, rows - 1):
            for i in range(1, cols - 1):
                if self.obstacle_mask[j, i]:
                    continue
                x = i - self.u[j, i] * self.time_step
                y = j - self.v[j, i] * self.time_step

                x = np.clip(x, 0, cols - 1)
                y = np.clip(y, 0, rows - 1)

                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)

                u_new[j, i] = self._bilinear_interpolation(self.u, x, y, x0, y0, x1, y1)
                v_new[j, i] = self._bilinear_interpolation(self.v, x, y, x0, y0, x1, y1)

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
        """
        div = (
            self.u[1:-1, 2:] - self.u[1:-1, :-2]
            + self.v[2:, 1:-1] - self.v[:-2, 1:-1]
        ) / 2

        for _ in range(20):  # Iterative solver
            self.p[1:-1, 1:-1] = (
                div
                + self.p[:-2, 1:-1]
                + self.p[2:, 1:-1]
                + self.p[1:-1, :-2]
                + self.p[1:-1, 2:]
            ) / 4

        self.u[1:-1, 1:-1] -= (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / 2
        self.v[1:-1, 1:-1] -= (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / 2

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

    def plot(self):
        """
        Plot the velocity field.
        """
        plt.quiver(self.u, self.v)
        plt.title("Velocity Field")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def simulate_particles(self, steps: int = 1) -> None:
        """
        Run the particle/fluid simulation for a number of steps.
        Keeps Main.py API stable.
        """
        for _ in range(steps):
            self.step()
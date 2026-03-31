import numpy as np
from typing import Callable
from scipy.ndimage import map_coordinates

from Mechanisms.Functions import calculate_reynolds_number, calculate_drag_coefficient


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
        compressible=False,
        gamma=1.4,
        gas_constant=287.05287,
        reference_temperature=288.15,
        cfl_number=0.45,
        density_floor=1e-4,
        pressure_floor=1.0,
        compressible_velocity_diffusion=0.0,
        compressible_flux_scheme="hllc",
        outflow_sponge_strength=0.06,
        outflow_sponge_start=0.88,
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
        self.acceleration_direction = self._normalize_direction(
            acceleration_direction)
        self.freestream_direction = self.acceleration_direction.copy()
        self.assume_laminar_edges = bool(assume_laminar_edges)
        self.edge_relaxation = float(np.clip(edge_relaxation, 0.0, 1.0))
        self.edge_speed = 0.0 if edge_speed is None else float(
            max(edge_speed, 0.0))
        self.wall_penalty = float(max(wall_penalty, 0.0))
        self.wall_shear_layers = max(int(wall_shear_layers), 1)
        self.turbulence_strength = float(max(turbulence_strength, 0.0))
        self.vorticity_confinement = float(max(vorticity_confinement, 0.0))
        self.simulation_time = 0.0
        self.ambient_velocity_profile = ambient_velocity_profile
        self.rocket_velocity_profile = rocket_velocity_profile
        self.inflow_blend = float(np.clip(inflow_blend, 0.0, 1.0))
        self.compressible = bool(compressible)
        self.gamma = float(max(gamma, 1.0001))
        self.gas_constant = float(max(gas_constant, 1e-8))
        self.reference_temperature = float(max(reference_temperature, 1.0))
        self.cfl_number = float(np.clip(cfl_number, 0.05, 0.95))
        self.density_floor = float(max(density_floor, 1e-8))
        self.pressure_floor = float(max(pressure_floor, 1e-8))
        self.compressible_velocity_diffusion = float(
            max(compressible_velocity_diffusion, 0.0))
        self.outflow_sponge_strength = float(max(outflow_sponge_strength, 0.0))
        self.outflow_sponge_start = float(np.clip(outflow_sponge_start, 0.5, 0.98))
        flux_name = str(compressible_flux_scheme).strip().lower()
        if flux_name not in {"hllc", "rusanov"}:
            raise ValueError(
                "compressible_flux_scheme must be 'hllc' or 'rusanov'")
        self.compressible_flux_scheme = flux_name
        self.nozzle_pressure_bc_enabled = False
        self.nozzle_inlet_total_pressure = None
        self.nozzle_inlet_total_temperature = None
        self.nozzle_outlet_static_pressure = None
        self.base_ambient_velocity = self.edge_speed * self.freestream_direction
        self.relative_velocity_target = self.base_ambient_velocity.copy()
        self.reference_acceleration_vector = np.zeros(2, dtype=float)
        self.wall_distance = None
        self.wall_normal_x = np.zeros(grid_size, dtype=float)
        self.wall_normal_y = np.zeros(grid_size, dtype=float)
        self.last_step_dt = self.time_step
        self.freestream_density = float(max(self.density, self.density_floor))
        self.freestream_temperature = self.reference_temperature
        self.freestream_pressure = float(max(
            self.freestream_density * self.gas_constant * self.freestream_temperature,
            self.pressure_floor,
        ))

        # Initialize velocity and pressure fields
        self.u = np.zeros(grid_size)  # x-velocity
        self.v = np.zeros(grid_size)  # y-velocity
        self.p = np.zeros(grid_size)  # Pressure field
        self.rho = np.full(grid_size, self.freestream_density, dtype=float)
        self.rho_u = np.zeros(grid_size, dtype=float)
        self.rho_v = np.zeros(grid_size, dtype=float)
        self.energy = np.full(
            grid_size,
            self.freestream_pressure / (self.gamma - 1.0),
            dtype=float,
        )
        self.temperature = np.full(
            grid_size, self.freestream_temperature, dtype=float)
        self.sound_speed = np.full(
            grid_size,
            np.sqrt(self.gamma * self.freestream_pressure / self.freestream_density),
            dtype=float,
        )
        self.mach = np.zeros(grid_size, dtype=float)

        if self.rocket_profile is not None:
            self.add_rocket_profile(self.rocket_profile)

        if self.compressible:
            self._sync_conservative_from_primitive()

    def _normalize_direction(self, direction):
        """Normalize a 2D direction vector."""
        direction_array = np.asarray(direction, dtype=float)
        if direction_array.shape != (2,):
            raise ValueError(
                "acceleration_direction must be a 2-element vector (x, y)")

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
        if self.compressible:
            self.rho[:, :] = self.freestream_density
            self.p[:, :] = self.freestream_pressure
            self.temperature[:, :] = self.freestream_temperature
            self._sync_conservative_from_primitive()
        self._enforce_obstacle_boundary()
        self._enforce_domain_boundary()

    def set_freestream_thermodynamics(self, density=None, temperature_k=None, pressure_pa=None):
        """Update the reference thermodynamic state used by compressible boundaries."""
        rho = self.freestream_density if density is None else float(
            max(density, self.density_floor))
        temperature = self.freestream_temperature if temperature_k is None else float(
            max(temperature_k, 1.0))
        pressure = self.freestream_pressure if pressure_pa is None else float(
            max(pressure_pa, self.pressure_floor))

        if density is not None and temperature_k is not None and pressure_pa is None:
            pressure = max(rho * self.gas_constant * temperature, self.pressure_floor)
        elif density is not None and pressure_pa is not None and temperature_k is None:
            temperature = max(pressure / (rho * self.gas_constant), 1.0)
        elif temperature_k is not None and pressure_pa is not None and density is None:
            rho = max(pressure / (self.gas_constant * temperature), self.density_floor)

        self.freestream_density = rho
        self.freestream_temperature = temperature
        self.freestream_pressure = pressure

    def set_freestream_mach(self, mach_number, direction=None, temperature_k=None, pressure_pa=None, density=None):
        """Convenience helper for compressible setups based on Mach number."""
        self.set_freestream_thermodynamics(
            density=density,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
        )
        sound_speed = np.sqrt(
            self.gamma * self.freestream_pressure / self.freestream_density)
        self.set_uniform_flow(float(max(mach_number, 0.0)) * sound_speed, direction)

    def set_nozzle_pressure_boundary(
        self,
        enabled=False,
        inlet_total_pressure=None,
        inlet_total_temperature=None,
        outlet_static_pressure=None,
    ):
        """Enable pressure-driven x-direction nozzle boundary conditions for compressible runs."""
        self.nozzle_pressure_bc_enabled = bool(enabled)
        self.nozzle_inlet_total_pressure = (
            None
            if inlet_total_pressure is None
            else float(max(inlet_total_pressure, self.pressure_floor))
        )
        self.nozzle_inlet_total_temperature = (
            None
            if inlet_total_temperature is None
            else float(max(inlet_total_temperature, 1.0))
        )
        self.nozzle_outlet_static_pressure = (
            None
            if outlet_static_pressure is None
            else float(max(outlet_static_pressure, self.pressure_floor))
        )

    def _enforce_nozzle_pressure_boundary(self):
        """Pressure-driven left-inlet / right-outlet boundary condition for nozzle-style domains."""
        rows, cols = self.grid_size
        if cols < 3:
            return

        beta = self.edge_relaxation
        gamma = self.gamma
        gas_constant = self.gas_constant
        inlet_total_pressure = (
            self.freestream_pressure
            if self.nozzle_inlet_total_pressure is None
            else self.nozzle_inlet_total_pressure
        )
        inlet_total_temperature = (
            self.freestream_temperature
            if self.nozzle_inlet_total_temperature is None
            else self.nozzle_inlet_total_temperature
        )
        outlet_static_pressure = (
            self.freestream_pressure
            if self.nozzle_outlet_static_pressure is None
            else self.nozzle_outlet_static_pressure
        )

        inlet_u = np.maximum(self.u[:, 1], 1e-8)
        inlet_v = self.v[:, 1]
        inlet_speed = np.hypot(inlet_u, inlet_v)
        inlet_sound = np.sqrt(np.maximum(gamma * self.p[:, 1] / np.maximum(self.rho[:, 1], self.density_floor), 1e-12))
        inlet_mach = np.clip(inlet_speed / np.maximum(inlet_sound, 1e-8), 0.0, 2.0)

        temperature_static = inlet_total_temperature / np.maximum(
            1.0 + 0.5 * (gamma - 1.0) * inlet_mach**2,
            1e-8,
        )
        pressure_static = inlet_total_pressure / np.maximum(
            (1.0 + 0.5 * (gamma - 1.0) * inlet_mach**2) ** (gamma / (gamma - 1.0)),
            1e-8,
        )
        density_static = np.maximum(
            pressure_static / np.maximum(gas_constant * temperature_static, 1e-8),
            self.density_floor,
        )

        self.rho[:, 0] = beta * density_static + (1.0 - beta) * self.rho[:, 1]
        self.p[:, 0] = beta * pressure_static + (1.0 - beta) * self.p[:, 1]
        self.u[:, 0] = np.maximum(beta * inlet_u + (1.0 - beta) * self.u[:, 1], 0.0)
        self.v[:, 0] = beta * inlet_v + (1.0 - beta) * self.v[:, 1]

        self.p[:, -1] = beta * outlet_static_pressure + (1.0 - beta) * self.p[:, -2]
        self.rho[:, -1] = self.rho[:, -2]
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]

        self.rho[0, :] = self.rho[1, :]
        self.rho[-1, :] = self.rho[-2, :]
        self.p[0, :] = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]
        self.u[0, :] = self.u[1, :]
        self.u[-1, :] = self.u[-2, :]
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

        self.temperature = np.maximum(
            self.p / (self.rho * self.gas_constant),
            1.0,
        )
        self._sync_conservative_from_primitive()

    def get_conservative_fields(self):
        """Return copies of the conservative variables for inspection or debugging."""
        return {
            "rho": self.rho.copy(),
            "rho_u": self.rho_u.copy(),
            "rho_v": self.rho_v.copy(),
            "energy": self.energy.copy(),
        }

    def _resolve_profile_velocity(self, profile, time_s, default_velocity):
        """Evaluate a velocity profile callback if present, otherwise return default velocity."""
        if profile is None:
            return np.asarray(default_velocity, dtype=float)

        candidate = np.asarray(profile(float(time_s)), dtype=float)
        if candidate.shape != (2,):
            raise ValueError(
                "Velocity profile must return a 2-element vector (vx, vy)")
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

        dt = max(float(self.last_step_dt), 1e-8)
        self.reference_acceleration_vector = (
            self.relative_velocity_target - previous_relative_velocity) / dt

        if speed > 1e-12:
            self.freestream_direction = relative_velocity / speed
            self.edge_speed = speed
            # Do NOT blend interior cells — only update boundary conditions (done via
            # _enforce_domain_boundary which reads self.edge_speed/freestream_direction).
            # Blending every cell each step injects uniform divergence that looks like
            # source/sink nodes all over the domain.

    def _update_scalar_density(self):
        """Keep legacy scalar density in sync with the mean fluid density field."""
        fluid = ~self.obstacle_mask
        if np.any(fluid):
            self.density = float(max(np.mean(self.rho[fluid]), self.density_floor))
        else:
            self.density = float(max(np.mean(self.rho), self.density_floor))

    def _safe_denom(self, value, eps=1e-8):
        """Keep denominator away from zero while preserving sign."""
        sign = np.where(value >= 0.0, 1.0, -1.0)
        return np.where(np.abs(value) < eps, sign * eps, value)

    def _sync_conservative_from_primitive(self):
        """Rebuild conservative variables from primitive fields."""
        self.rho = np.where(np.isfinite(self.rho), self.rho,
                            self.freestream_density)
        self.rho = np.maximum(self.rho, self.density_floor)

        invalid_pressure = (~np.isfinite(self.p)) | (self.p <= self.pressure_floor)
        if np.any(invalid_pressure):
            self.p[invalid_pressure] = np.maximum(
                self.rho[invalid_pressure] * self.gas_constant * self.temperature[invalid_pressure],
                self.pressure_floor,
            )

        invalid_temperature = (~np.isfinite(self.temperature)) | (self.temperature <= 1.0)
        if np.any(invalid_temperature):
            self.temperature[invalid_temperature] = np.maximum(
                self.p[invalid_temperature]
                / (self.rho[invalid_temperature] * self.gas_constant),
                1.0,
            )

        self.rho_u = self.rho * self.u
        self.rho_v = self.rho * self.v
        kinetic_energy = 0.5 * self.rho * (self.u**2 + self.v**2)
        internal_energy = np.maximum(
            self.p / (self.gamma - 1.0),
            self.pressure_floor / (self.gamma - 1.0),
        )
        self.energy = internal_energy + kinetic_energy
        self.sound_speed = np.sqrt(self.gamma * self.p / self.rho)
        self.mach = np.hypot(self.u, self.v) / np.maximum(self.sound_speed, 1e-8)
        self._update_scalar_density()

    def _sync_primitive_from_conservative(self):
        """Recover primitive variables from conservative fields."""
        self.rho = np.where(np.isfinite(self.rho), self.rho,
                            self.freestream_density)
        self.rho = np.maximum(self.rho, self.density_floor)

        self.u = np.divide(
            self.rho_u,
            self.rho,
            out=np.zeros_like(self.rho_u),
            where=self.rho > self.density_floor,
        )
        self.v = np.divide(
            self.rho_v,
            self.rho,
            out=np.zeros_like(self.rho_v),
            where=self.rho > self.density_floor,
        )

        kinetic_energy = 0.5 * self.rho * (self.u**2 + self.v**2)
        internal_energy = np.maximum(
            self.energy - kinetic_energy,
            self.pressure_floor / (self.gamma - 1.0),
        )
        self.p = np.maximum((self.gamma - 1.0) * internal_energy,
                            self.pressure_floor)
        self.temperature = np.maximum(
            self.p / (self.rho * self.gas_constant),
            1.0,
        )
        self.sound_speed = np.sqrt(self.gamma * self.p / self.rho)
        self.mach = np.hypot(self.u, self.v) / np.maximum(self.sound_speed, 1e-8)
        self._update_scalar_density()

    def _conservative_state(self):
        """Pack conservative fields into a single array for flux updates."""
        return np.stack((self.rho, self.rho_u, self.rho_v, self.energy), axis=0)

    def _pressure_from_state(self, state):
        """Equation of state for conservative variables."""
        rho = np.maximum(state[0], self.density_floor)
        momentum_sq = state[1] ** 2 + state[2] ** 2
        kinetic_energy = 0.5 * momentum_sq / rho
        internal_energy = np.maximum(
            state[3] - kinetic_energy,
            self.pressure_floor / (self.gamma - 1.0),
        )
        return np.maximum((self.gamma - 1.0) * internal_energy,
                          self.pressure_floor)

    def _sound_speed_from_state(self, state):
        """Local sound speed for a conservative state array."""
        rho = np.maximum(state[0], self.density_floor)
        pressure = self._pressure_from_state(state)
        return np.sqrt(self.gamma * pressure / rho)

    def _compute_stable_time_step(self):
        """Compute a conservative CFL-limited time step for compressible updates."""
        if not self.compressible:
            return self.time_step

        fluid = ~self.obstacle_mask
        if not np.any(fluid):
            return self.time_step

        signal_speed_x = np.abs(self.u) + self.sound_speed
        signal_speed_y = np.abs(self.v) + self.sound_speed
        max_signal = float(max(
            np.max(signal_speed_x[fluid]),
            np.max(signal_speed_y[fluid]),
            1e-8,
        ))
        return float(max(min(self.time_step, self.cfl_number / max_signal), 1e-6))

    def _euler_flux(self, state, axis):
        """Compute inviscid Euler fluxes in the selected axis."""
        rho = np.maximum(state[0], self.density_floor)
        mx = state[1]
        my = state[2]
        energy = state[3]
        pressure = self._pressure_from_state(state)
        u = mx / rho
        v = my / rho

        if axis == 0:
            return np.stack((
                mx,
                mx * u + pressure,
                my * u,
                (energy + pressure) * u,
            ), axis=0)

        return np.stack((
            my,
            mx * v,
            my * v + pressure,
            (energy + pressure) * v,
        ), axis=0)

    def _rusanov_flux(self, left_state, right_state, axis):
        """Local Lax-Friedrichs/Rusanov interface flux."""
        left_flux = self._euler_flux(left_state, axis)
        right_flux = self._euler_flux(right_state, axis)

        rho_left = np.maximum(left_state[0], self.density_floor)
        rho_right = np.maximum(right_state[0], self.density_floor)
        sound_left = self._sound_speed_from_state(left_state)
        sound_right = self._sound_speed_from_state(right_state)

        if axis == 0:
            velocity_left = np.abs(left_state[1] / rho_left)
            velocity_right = np.abs(right_state[1] / rho_right)
        else:
            velocity_left = np.abs(left_state[2] / rho_left)
            velocity_right = np.abs(right_state[2] / rho_right)

        s_max = np.maximum(velocity_left + sound_left,
                           velocity_right + sound_right)
        return 0.5 * (left_flux + right_flux) - 0.5 * s_max[None, ...] * (right_state - left_state)

    def _hllc_flux(self, left_state, right_state, axis):
        """HLLC approximate Riemann flux for 2D Euler states across one axis-aligned face."""
        flux_left = self._euler_flux(left_state, axis)
        flux_right = self._euler_flux(right_state, axis)

        rho_left = np.maximum(left_state[0], self.density_floor)
        rho_right = np.maximum(right_state[0], self.density_floor)
        pressure_left = self._pressure_from_state(left_state)
        pressure_right = self._pressure_from_state(right_state)
        sound_left = self._sound_speed_from_state(left_state)
        sound_right = self._sound_speed_from_state(right_state)

        if axis == 0:
            normal_momentum_left = left_state[1]
            normal_momentum_right = right_state[1]
            tangential_momentum_left = left_state[2]
            tangential_momentum_right = right_state[2]
        else:
            normal_momentum_left = left_state[2]
            normal_momentum_right = right_state[2]
            tangential_momentum_left = left_state[1]
            tangential_momentum_right = right_state[1]

        normal_velocity_left = normal_momentum_left / rho_left
        normal_velocity_right = normal_momentum_right / rho_right
        tangential_velocity_left = tangential_momentum_left / rho_left
        tangential_velocity_right = tangential_momentum_right / rho_right

        wave_speed_left = np.minimum(
            normal_velocity_left - sound_left,
            normal_velocity_right - sound_right,
        )
        wave_speed_right = np.maximum(
            normal_velocity_left + sound_left,
            normal_velocity_right + sound_right,
        )

        denominator = (
            rho_left * (wave_speed_left - normal_velocity_left)
            - rho_right * (wave_speed_right - normal_velocity_right)
        )
        denominator = self._safe_denom(denominator)
        contact_speed = (
            pressure_right
            - pressure_left
            + rho_left * normal_velocity_left * (wave_speed_left - normal_velocity_left)
            - rho_right * normal_velocity_right * (wave_speed_right - normal_velocity_right)
        ) / denominator

        pressure_star_left = pressure_left + rho_left * \
            (wave_speed_left - normal_velocity_left) * \
            (contact_speed - normal_velocity_left)
        pressure_star_right = pressure_right + rho_right * \
            (wave_speed_right - normal_velocity_right) * \
            (contact_speed - normal_velocity_right)
        pressure_star = np.maximum(
            0.5 * (pressure_star_left + pressure_star_right),
            self.pressure_floor,
        )

        denom_left_star = wave_speed_left - contact_speed
        denom_right_star = wave_speed_right - contact_speed
        denom_left_star = self._safe_denom(denom_left_star)
        denom_right_star = self._safe_denom(denom_right_star)

        rho_star_left = rho_left * \
            (wave_speed_left - normal_velocity_left) / denom_left_star
        rho_star_right = rho_right * \
            (wave_speed_right - normal_velocity_right) / denom_right_star
        rho_star_left = np.maximum(rho_star_left, self.density_floor)
        rho_star_right = np.maximum(rho_star_right, self.density_floor)

        energy_left = left_state[3]
        energy_right = right_state[3]
        energy_star_left = (
            (wave_speed_left - normal_velocity_left) * energy_left
            - pressure_left * normal_velocity_left
            + pressure_star * contact_speed
        ) / denom_left_star
        energy_star_right = (
            (wave_speed_right - normal_velocity_right) * energy_right
            - pressure_right * normal_velocity_right
            + pressure_star * contact_speed
        ) / denom_right_star

        normal_momentum_star_left = rho_star_left * contact_speed
        normal_momentum_star_right = rho_star_right * contact_speed
        tangential_momentum_star_left = rho_star_left * tangential_velocity_left
        tangential_momentum_star_right = rho_star_right * tangential_velocity_right

        star_left = np.zeros_like(left_state)
        star_right = np.zeros_like(right_state)
        star_left[0] = rho_star_left
        star_right[0] = rho_star_right

        if axis == 0:
            star_left[1] = normal_momentum_star_left
            star_left[2] = tangential_momentum_star_left
            star_right[1] = normal_momentum_star_right
            star_right[2] = tangential_momentum_star_right
        else:
            star_left[1] = tangential_momentum_star_left
            star_left[2] = normal_momentum_star_left
            star_right[1] = tangential_momentum_star_right
            star_right[2] = normal_momentum_star_right

        star_left[3] = energy_star_left
        star_right[3] = energy_star_right

        flux = flux_right.copy()
        mask_left = wave_speed_left >= 0.0
        mask_left_star = (wave_speed_left < 0.0) & (contact_speed >= 0.0)
        mask_right_star = (contact_speed < 0.0) & (wave_speed_right >= 0.0)

        if np.any(mask_left):
            flux[:, mask_left] = flux_left[:, mask_left]
        if np.any(mask_left_star):
            flux[:, mask_left_star] = (
                flux_left[:, mask_left_star]
                + wave_speed_left[mask_left_star][None, :]
                * (star_left[:, mask_left_star] - left_state[:, mask_left_star])
            )
        if np.any(mask_right_star):
            flux[:, mask_right_star] = (
                flux_right[:, mask_right_star]
                + wave_speed_right[mask_right_star][None, :]
                * (star_right[:, mask_right_star] - right_state[:, mask_right_star])
            )

        return flux

    def _interface_flux(self, left_state, right_state, axis):
        """Dispatch interface flux computation for compressible Euler updates."""
        if self.compressible_flux_scheme == "hllc":
            return self._hllc_flux(left_state, right_state, axis)
        return self._rusanov_flux(left_state, right_state, axis)

    def _advance_compressible_euler(self, dt):
        """Advance one first-order finite-volume Euler step using dimensional splitting."""
        state = self._conservative_state()
        updated = state.copy()

        if self.grid_size[1] > 2:
            flux_x = self._interface_flux(state[:, :, :-1], state[:, :, 1:], axis=0)
            updated[:, :, 1:-1] -= dt * (flux_x[:, :, 1:] - flux_x[:, :, :-1])

        state_y = updated.copy()
        if self.grid_size[0] > 2:
            flux_y = self._interface_flux(state_y[:, :-1, :], state_y[:, 1:, :], axis=1)
            state_y[:, 1:-1, :] -= dt * (flux_y[:, 1:, :] - flux_y[:, :-1, :])

        if np.any(self.obstacle_mask):
            obstacle = self.obstacle_mask
            state_y[0, obstacle] = np.maximum(state[0, obstacle], self.density_floor)
            state_y[1, obstacle] = 0.0
            state_y[2, obstacle] = 0.0
            state_y[3, obstacle] = np.maximum(
                state[3, obstacle],
                self.freestream_pressure / (self.gamma - 1.0),
            )

        self.rho = state_y[0]
        self.rho_u = state_y[1]
        self.rho_v = state_y[2]
        self.energy = state_y[3]

        self.rho = np.where(np.isfinite(self.rho), self.rho, self.freestream_density)
        self.rho = np.maximum(self.rho, self.density_floor)
        self.rho_u = np.where(np.isfinite(self.rho_u), self.rho_u, 0.0)
        self.rho_v = np.where(np.isfinite(self.rho_v), self.rho_v, 0.0)
        self.energy = np.where(
            np.isfinite(self.energy),
            self.energy,
            self.freestream_pressure / (self.gamma - 1.0),
        )
        kinetic = 0.5 * (self.rho_u**2 + self.rho_v**2) / np.maximum(self.rho, self.density_floor)
        self.energy = np.maximum(self.energy, kinetic + self.pressure_floor / (self.gamma - 1.0))

        self._sync_primitive_from_conservative()

    def _enforce_compressible_domain_boundary(self):
        """Directional inflow/outflow boundary conditions for the compressible branch."""
        if not self.assume_laminar_edges:
            self.u[0, :] = self.u[1, :]
            self.u[-1, :] = self.u[-2, :]
            self.v[:, 0] = self.v[:, 1]
            self.v[:, -1] = self.v[:, -2]
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
            self.u[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.rho[0, :] = self.rho[1, :]
            self.rho[-1, :] = self.rho[-2, :]
            self.rho[:, 0] = self.rho[:, 1]
            self.rho[:, -1] = self.rho[:, -2]
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]
            self.temperature = np.maximum(
                self.p / (self.rho * self.gas_constant),
                1.0,
            )
            self._sync_conservative_from_primitive()
            return

        if (
            self.nozzle_pressure_bc_enabled
            and abs(self.freestream_direction[0]) >= abs(self.freestream_direction[1])
            and self.freestream_direction[0] >= 0.0
        ):
            self._enforce_nozzle_pressure_boundary()
            return

        target_u = self.edge_speed * self.freestream_direction[0]
        target_v = self.edge_speed * self.freestream_direction[1]
        target_rho = self.freestream_density
        target_p = self.freestream_pressure
        beta = self.edge_relaxation

        if abs(self.freestream_direction[0]) >= abs(self.freestream_direction[1]):
            if self.freestream_direction[0] <= 0.0:
                self.rho[:, -1] = beta * target_rho + (1.0 - beta) * self.rho[:, -2]
                self.u[:, -1] = beta * target_u + (1.0 - beta) * self.u[:, -2]
                self.v[:, -1] = beta * target_v + (1.0 - beta) * self.v[:, -2]
                self.p[:, -1] = beta * target_p + (1.0 - beta) * self.p[:, -2]

                self.rho[:, 0] = self.rho[:, 1]
                self.u[:, 0] = self.u[:, 1]
                self.v[:, 0] = self.v[:, 1]
                self.p[:, 0] = self.p[:, 1]
            else:
                self.rho[:, 0] = beta * target_rho + (1.0 - beta) * self.rho[:, 1]
                self.u[:, 0] = beta * target_u + (1.0 - beta) * self.u[:, 1]
                self.v[:, 0] = beta * target_v + (1.0 - beta) * self.v[:, 1]
                self.p[:, 0] = beta * target_p + (1.0 - beta) * self.p[:, 1]

                self.rho[:, -1] = self.rho[:, -2]
                self.u[:, -1] = self.u[:, -2]
                self.v[:, -1] = self.v[:, -2]
                self.p[:, -1] = self.p[:, -2]

            self.rho[0, :] = self.rho[1, :]
            self.rho[-1, :] = self.rho[-2, :]
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
            self.u[0, :] = self.u[1, :]
            self.u[-1, :] = self.u[-2, :]
            self.v[0, :] = 0.0
            self.v[-1, :] = 0.0
        else:
            if self.freestream_direction[1] >= 0.0:
                self.rho[0, :] = beta * target_rho + (1.0 - beta) * self.rho[1, :]
                self.u[0, :] = beta * target_u + (1.0 - beta) * self.u[1, :]
                self.v[0, :] = beta * target_v + (1.0 - beta) * self.v[1, :]
                self.p[0, :] = beta * target_p + (1.0 - beta) * self.p[1, :]

                self.rho[-1, :] = self.rho[-2, :]
                self.u[-1, :] = self.u[-2, :]
                self.v[-1, :] = self.v[-2, :]
                self.p[-1, :] = self.p[-2, :]
            else:
                self.rho[-1, :] = beta * target_rho + (1.0 - beta) * self.rho[-2, :]
                self.u[-1, :] = beta * target_u + (1.0 - beta) * self.u[-2, :]
                self.v[-1, :] = beta * target_v + (1.0 - beta) * self.v[-2, :]
                self.p[-1, :] = beta * target_p + (1.0 - beta) * self.p[-2, :]

                self.rho[0, :] = self.rho[1, :]
                self.u[0, :] = self.u[1, :]
                self.v[0, :] = self.v[1, :]
                self.p[0, :] = self.p[1, :]

            self.rho[:, 0] = self.rho[:, 1]
            self.rho[:, -1] = self.rho[:, -2]
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]
            self.u[:, 0] = 0.0
            self.u[:, -1] = 0.0
            self.v[:, 0] = self.v[:, 1]
            self.v[:, -1] = self.v[:, -2]

        self.temperature = np.maximum(
            self.p / (self.rho * self.gas_constant),
            1.0,
        )
        self._sync_conservative_from_primitive()

    def _apply_compressible_obstacle_state(self):
        """Populate obstacle cells with quiescent thermodynamic states for the compressible branch."""
        obstacle = getattr(self, "obstacle_mask", None)
        if obstacle is None or not np.any(obstacle):
            return

        fluid = ~obstacle
        ghost_cells = obstacle & self._dilate_mask(fluid)
        rows, cols = self.grid_size

        if np.any(ghost_cells):
            for gy, gx in np.argwhere(ghost_cells):
                y0 = max(gy - 1, 0)
                y1 = min(gy + 2, rows)
                x0 = max(gx - 1, 0)
                x1 = min(gx + 2, cols)
                local_fluid = fluid[y0:y1, x0:x1]
                if np.any(local_fluid):
                    self.rho[gy, gx] = max(
                        float(np.mean(self.rho[y0:y1, x0:x1][local_fluid])),
                        self.density_floor,
                    )
                    self.p[gy, gx] = max(
                        float(np.mean(self.p[y0:y1, x0:x1][local_fluid])),
                        self.pressure_floor,
                    )
                else:
                    self.rho[gy, gx] = self.freestream_density
                    self.p[gy, gx] = self.freestream_pressure

        deep_interior = obstacle & ~ghost_cells
        if np.any(deep_interior):
            self.rho[deep_interior] = self.freestream_density
            self.p[deep_interior] = self.freestream_pressure

        self.u[obstacle] = 0.0
        self.v[obstacle] = 0.0
        self.temperature[obstacle] = np.maximum(
            self.p[obstacle] / (self.rho[obstacle] * self.gas_constant),
            1.0,
        )
        self.rho_u[obstacle] = 0.0
        self.rho_v[obstacle] = 0.0
        self.energy[obstacle] = self.p[obstacle] / (self.gamma - 1.0)

    def _apply_outflow_sponge(self, dt):
        """Relax downstream band toward freestream to suppress outlet reflections."""
        if self.outflow_sponge_strength <= 0.0:
            return

        rows, cols = self.grid_size
        fluid = ~self.obstacle_mask
        if not np.any(fluid):
            return

        beta_max = np.clip(self.outflow_sponge_strength * dt, 0.0, 0.25)
        if beta_max <= 0.0:
            return

        target_u = self.edge_speed * self.freestream_direction[0]
        target_v = self.edge_speed * self.freestream_direction[1]

        if abs(self.freestream_direction[0]) >= abs(self.freestream_direction[1]):
            n0 = int(self.outflow_sponge_start * cols)
            n0 = int(np.clip(n0, 1, cols - 2))
            ramp = np.linspace(0.0, 1.0, cols - n0)[None, :]
            beta = beta_max * (ramp**2)
            if self.freestream_direction[0] >= 0.0:
                sl = (slice(None), slice(n0, cols))
            else:
                sl = (slice(None), slice(0, cols - n0))
                beta = beta[:, ::-1]
        else:
            n0 = int(self.outflow_sponge_start * rows)
            n0 = int(np.clip(n0, 1, rows - 2))
            ramp = np.linspace(0.0, 1.0, rows - n0)[:, None]
            beta = beta_max * (ramp**2)
            if self.freestream_direction[1] >= 0.0:
                sl = (slice(n0, rows), slice(None))
            else:
                sl = (slice(0, rows - n0), slice(None))
                beta = beta[::-1, :]

        local_fluid = fluid[sl]
        if not np.any(local_fluid):
            return

        self.rho[sl] = np.where(local_fluid, (1.0 - beta) * self.rho[sl] + beta * self.freestream_density, self.rho[sl])
        self.p[sl] = np.where(local_fluid, (1.0 - beta) * self.p[sl] + beta * self.freestream_pressure, self.p[sl])
        self.u[sl] = np.where(local_fluid, (1.0 - beta) * self.u[sl] + beta * target_u, self.u[sl])
        self.v[sl] = np.where(local_fluid, (1.0 - beta) * self.v[sl] + beta * target_v, self.v[sl])
        self.temperature = np.maximum(self.p / (self.rho * self.gas_constant), 1.0)
        self._sync_conservative_from_primitive()

    def _step_compressible(self):
        """Advance the compressible branch using a simple first-order finite-volume update."""
        local_dt = self._compute_stable_time_step()
        self._apply_body_force(dt=local_dt)
        self._enforce_domain_boundary()
        self._enforce_obstacle_boundary()
        self._advance_compressible_euler(local_dt)
        self._apply_outflow_sponge(local_dt)

        # Optional extra smoothing for difficult setups.
        # Default is OFF because first-order Rusanov already introduces strong
        # numerical diffusion; adding implicit velocity diffusion here can wipe
        # out wake structures and smear shock features.
        if self.compressible_velocity_diffusion > 0.0:
            original_viscosity = self.viscosity
            self.viscosity = self.compressible_velocity_diffusion
            self._diffuse(dt=local_dt)
            self.viscosity = original_viscosity
            self._sync_conservative_from_primitive()

        self._enforce_obstacle_boundary()
        self._enforce_domain_boundary()
        self.last_step_dt = local_dt
        self.simulation_time += local_dt

    def add_source(self, x, y, strength):
        """
        Add a velocity source at a specific location.
        """
        self.u[y, x] += strength[0]
        self.v[y, x] += strength[1]
        if self.compressible:
            self._sync_conservative_from_primitive()

    def add_rocket_profile(
        self,
        rocket_profile=None,
        center=None,
        body_fraction=0.65,
        nose_points=128,
        fore_spike_fraction=0.0,
        fore_spike_half_width_fraction=0.18,
        mask=None,
    ):
        """
        Registers a solid obstacle mask from RocketProfile or direct mask.
        """
        if rocket_profile is not None:
            self.rocket_profile = rocket_profile

        if mask is not None:
            if mask.shape != self.grid_size:
                raise ValueError(
                    f"mask shape {mask.shape} != grid_size {self.grid_size}")
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
            nose_points=nose_points,
            fore_spike_fraction=fore_spike_fraction,
            fore_spike_half_width_fraction=fore_spike_half_width_fraction,
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

        local_speed = np.mean(
            np.hypot(self.u[boundary_layer], self.v[boundary_layer]))
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
        wall_factor = 1.0 - \
            np.clip((distance - 1.0) / max(self.wall_shear_layers, 1), 0.0, 1.0)
        damping = np.clip((0.18 + 0.12 * drag_coefficient) *
                          self.time_step * wall_factor, 0.0, 0.85)

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
        tangential_velocity *= (1.0 - np.clip(0.85 * damping, 0.0, 0.85))

        self.u[boundary_layer] = normal_velocity * \
            nx + tangential_velocity * tx
        self.v[boundary_layer] = normal_velocity * \
            ny + tangential_velocity * ty

    def _apply_ghost_cell_reconstruction(self):
        """Reconstruct obstacle ghost cells from adjacent fluid cells (no-slip at wall midpoint)."""
        if getattr(self, "obstacle_mask", None) is None:
            return
        obstacle = self.obstacle_mask
        if not np.any(obstacle):
            return

        fluid = ~obstacle
        if not np.any(fluid):
            return

        # Ghost cells are obstacle cells adjacent to at least one fluid cell.
        ghost_cells = obstacle & self._dilate_mask(fluid)
        if not np.any(ghost_cells):
            self.u[obstacle] = 0.0
            self.v[obstacle] = 0.0
            return

        u_ghost = np.zeros_like(self.u)
        v_ghost = np.zeros_like(self.v)

        rows, cols = self.grid_size
        ghost_indices = np.argwhere(ghost_cells)

        for gy, gx in ghost_indices:
            y0 = max(gy - 1, 0)
            y1 = min(gy + 2, rows)
            x0 = max(gx - 1, 0)
            x1 = min(gx + 2, cols)

            local_fluid = fluid[y0:y1, x0:x1]
            if not np.any(local_fluid):
                continue

            local_u = self.u[y0:y1, x0:x1][local_fluid]
            local_v = self.v[y0:y1, x0:x1][local_fluid]

            # Mirror at wall midpoint with u_wall=0 => u_ghost = -u_fluid.
            u_ghost[gy, gx] = -float(np.mean(local_u))
            v_ghost[gy, gx] = -float(np.mean(local_v))

        self.u[ghost_cells] = u_ghost[ghost_cells]
        self.v[ghost_cells] = v_ghost[ghost_cells]

        # Deep interior does not affect fluid stencils directly; keep it quiescent.
        deep_interior = obstacle & ~ghost_cells
        if np.any(deep_interior):
            self.u[deep_interior] = 0.0
            self.v[deep_interior] = 0.0

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
            wall_damping[near_wall] = np.clip(
                wall_distance[near_wall] / max(self.wall_shear_layers, 1), 0.15, 1.0)
            eddy_viscosity *= wall_damping

        effective_viscosity = self.viscosity + eddy_viscosity
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        self.u[fluid] += self.time_step * \
            effective_viscosity[fluid] * lap_u[fluid]
        self.v[fluid] += self.time_step * \
            effective_viscosity[fluid] * lap_v[fluid]

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
        self.u[fluid] += self.vorticity_confinement * \
            self.time_step * confine_u[fluid]
        self.v[fluid] += self.vorticity_confinement * \
            self.time_step * confine_v[fluid]

    def _enforce_obstacle_boundary(self):
        """
        Immersed-boundary no-slip treatment using ghost-cell reconstruction.
        """
        if self.compressible:
            self._apply_compressible_obstacle_state()
            return
        self._apply_ghost_cell_reconstruction()

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

    def _apply_body_force(self, dt=None):
        """Apply uniform body-force acceleration to fluid cells.

        Two sources are combined:
        1) User-configured constant acceleration (magnitude + direction)
        2) Reference-frame acceleration from d(u_ambient - u_rocket)/dt
        """
        fluid = ~self.obstacle_mask
        if not np.any(fluid):
            return

        step_dt = self.time_step if dt is None else float(dt)

        acceleration_vector = self.reference_acceleration_vector.copy()
        if self.acceleration != 0.0:
            acceleration_vector += self.acceleration * self.acceleration_direction

        if self.compressible:
            mom_x = self.rho_u.copy()
            mom_y = self.rho_v.copy()
            self.rho_u[fluid] += self.rho[fluid] * acceleration_vector[0] * step_dt
            self.rho_v[fluid] += self.rho[fluid] * acceleration_vector[1] * step_dt
            self.energy[fluid] += step_dt * (
                mom_x[fluid] * acceleration_vector[0]
                + mom_y[fluid] * acceleration_vector[1]
            )
            self._sync_primitive_from_conservative()
            return

        accel_step = acceleration_vector * step_dt
        self.u[fluid] += accel_step[0]
        self.v[fluid] += accel_step[1]

    def _enforce_domain_boundary(self):
        """Outer-domain boundary model (laminar freestream or fallback no-slip)."""
        if self.compressible:
            self._enforce_compressible_domain_boundary()
            return

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
        if self.compressible:
            self._step_compressible()
            return

        self._apply_body_force()
        self._enforce_domain_boundary()   # seed inlet before advect
        self._enforce_obstacle_boundary()
        self._advect()
        # restore inlet after advect wipes boundaries to zero
        self._enforce_domain_boundary()
        self._enforce_obstacle_boundary()
        self._diffuse()
        self._enforce_domain_boundary()   # re-support inlet before pressure solve
        self._enforce_obstacle_boundary()
        self._apply_turbulence_model()
        self._enforce_obstacle_boundary()
        self._project()
        self._enforce_obstacle_boundary()
        self._enforce_domain_boundary()   # final clean-up
        self.last_step_dt = self.time_step
        self.simulation_time += self.time_step

    def step_coupled(self, dynamics, dt=None):
        """Advance fluid one step, then update rocket dynamics from the resulting flow field.

        This keeps the two-way coupling sequence in one place:
        1) advance CFD with current rocket-frame velocity,
        2) compute drag from updated flow,
        3) integrate rocket state for the next step.
        """
        self.step()
        coupling_dt = self.last_step_dt if dt is None else float(dt)
        return dynamics.integrate_step(self, coupling_dt, self.simulation_time)

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

        x_back = np.clip(i_idx - self.u[1:-1, 1:-1]
                         * self.time_step, 0.0, cols - 1.0)
        y_back = np.clip(j_idx - self.v[1:-1, 1:-1]
                         * self.time_step, 0.0, rows - 1.0)

        # map_coordinates expects coordinates in (row, col) order.
        sample_coords = np.vstack((y_back.ravel(), x_back.ravel()))
        u_interp = map_coordinates(
            self.u, sample_coords, order=1, mode="nearest").reshape(rows - 2, cols - 2)
        v_interp = map_coordinates(
            self.v, sample_coords, order=1, mode="nearest").reshape(rows - 2, cols - 2)

        u_new = self.u.copy()
        v_new = self.v.copy()

        u_new[1:-1, 1:-1] = np.where(fluid, u_interp, self.u[1:-1, 1:-1])
        v_new[1:-1, 1:-1] = np.where(fluid, v_interp, self.v[1:-1, 1:-1])

        self.u = u_new
        self.v = v_new

    def _diffuse(self, dt=None):
        """
        Diffuse the velocity field using the diffusion equation.
        """
        step_dt = self.time_step if dt is None else float(dt)
        alpha = self.viscosity * step_dt
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

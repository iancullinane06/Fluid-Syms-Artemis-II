from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from Mechanisms.Classes import AtmosphereState
from Mechanisms.FluidSimulation import FluidSimulation


@dataclass
class RocketState:
    mass_kg: float = 0.0
    altitude_m: float = 0.0
    velocity_mps: float = 0.0
    acceleration_mps2: float = 0.0
    thrust_n: float = 0.0
    drag_n: float = 0.0
    net_force_n: float = 0.0


class RocketDynamics:
    EARTH_RADIUS_M = 6_371_000.0
    R_AIR = 287.05287
    GAMMA_AIR = 1.4
    SUTHERLAND_REFERENCE_T = 273.15
    SUTHERLAND_REFERENCE_MU = 1.716e-5
    SUTHERLAND_S = 110.4

    def __init__(
        self,
        mass_kg: float,
        thrust_profile: Callable[[float], float],
        gravity_mps2: float = 9.80665,
        sim_speed_scale: float = 1.0 / 120.0,
        drag_force_scale_n: float = 18000.0,
        flight_direction: tuple[float, float] = (1.0, 0.0),
        dry_mass_kg: float | None = None,
        specific_impulse_s: float | None = None,
    ) -> None:
        self.mass_kg = float(max(mass_kg, 1e-6))
        self.thrust_profile = thrust_profile
        self.gravity_mps2 = float(max(gravity_mps2, 0.0))
        self.sim_speed_scale = float(max(sim_speed_scale, 1e-9))
        self.drag_force_scale_n = float(max(drag_force_scale_n, 1e-9))
        self.dry_mass_kg = float(self.mass_kg if dry_mass_kg is None else np.clip(
            dry_mass_kg, 1e-6, self.mass_kg))
        self.specific_impulse_s = None if specific_impulse_s is None else float(
            max(specific_impulse_s, 1e-6))
        self.current_mass_kg = self.mass_kg
        self.reference_density_kg_m3 = 1.225

        direction = np.asarray(flight_direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-12:
            raise ValueError("flight_direction cannot be zero")
        self.flight_direction = direction / direction_norm

        self.state = RocketState(
            mass_kg=self.current_mass_kg, thrust_n=float(self.thrust_profile(0.0)))

    def _dynamic_viscosity_sutherland(self, temperature_k: float) -> float:
        temperature = max(float(temperature_k), 1.0)
        numerator = self.SUTHERLAND_REFERENCE_MU * \
            (temperature / self.SUTHERLAND_REFERENCE_T) ** 1.5
        return numerator * (self.SUTHERLAND_REFERENCE_T + self.SUTHERLAND_S) / (temperature + self.SUTHERLAND_S)

    def atmosphere_at_altitude(self, altitude_m: float) -> AtmosphereState:
        hb = np.array([0.0, 11_000.0, 20_000.0, 32_000.0,
                      47_000.0, 51_000.0, 71_000.0, 84_852.0], dtype=float)
        lapse = np.array([-0.0065, 0.0, 0.0010, 0.0028,
                         0.0, -0.0028, -0.0020], dtype=float)

        temperature_bases = [288.15]
        pressure_bases = [101_325.0]

        for index, lapse_rate in enumerate(lapse):
            delta_h = hb[index + 1] - hb[index]
            base_temperature = temperature_bases[index]
            base_pressure = pressure_bases[index]

            if abs(lapse_rate) < 1e-12:
                next_temperature = base_temperature
                exponent = -self.gravity_mps2 * delta_h / \
                    (self.R_AIR * base_temperature)
                next_pressure = base_pressure * math.exp(exponent)
            else:
                next_temperature = base_temperature + lapse_rate * delta_h
                exponent = -self.gravity_mps2 / (lapse_rate * self.R_AIR)
                next_pressure = base_pressure * \
                    (next_temperature / base_temperature) ** exponent

            temperature_bases.append(next_temperature)
            pressure_bases.append(next_pressure)

        altitude = float(np.clip(altitude_m, hb[0], hb[-1]))
        layer = int(np.searchsorted(hb, altitude, side="right") - 1)
        layer = min(layer, len(lapse) - 1)

        base_altitude = hb[layer]
        lapse_rate = lapse[layer]
        base_temperature = temperature_bases[layer]
        base_pressure = pressure_bases[layer]
        delta_h = altitude - base_altitude

        if abs(lapse_rate) < 1e-12:
            temperature = base_temperature
            pressure = base_pressure * \
                math.exp(-self.gravity_mps2 * delta_h /
                         (self.R_AIR * temperature))
        else:
            temperature = base_temperature + lapse_rate * delta_h
            exponent = -self.gravity_mps2 / (lapse_rate * self.R_AIR)
            pressure = base_pressure * (temperature / base_temperature) ** exponent

        density = pressure / (self.R_AIR * temperature)
        viscosity = self._dynamic_viscosity_sutherland(temperature)
        speed_of_sound = math.sqrt(self.GAMMA_AIR * self.R_AIR * temperature)
        gravity = self.gravity_mps2 * \
            (self.EARTH_RADIUS_M / (self.EARTH_RADIUS_M + altitude)) ** 2

        return AtmosphereState(
            altitude_m=altitude,
            temperature_k=temperature,
            pressure_pa=pressure,
            density_kg_m3=density,
            dynamic_viscosity_pa_s=viscosity,
            speed_of_sound_m_s=speed_of_sound,
            gravity_m_s2=gravity,
        )

    def rocket_velocity_profile(self, _: float) -> np.ndarray:
        speed_solver = self.state.velocity_mps * self.sim_speed_scale
        return speed_solver * self.flight_direction

    @staticmethod
    def compute_surface_force_components(sim: FluidSimulation) -> tuple[float, float, float, np.ndarray]:
        pressure = sim.p
        pressure_reference = 0.0
        if getattr(sim, "compressible", False):
            pressure_reference = float(
                max(getattr(sim, "freestream_pressure", 0.0), 0.0))
        dynamic_viscosity = max(sim.viscosity * sim.density, 1e-8)

        dudx = np.zeros_like(sim.u)
        dudy = np.zeros_like(sim.u)
        dvdx = np.zeros_like(sim.v)
        dvdy = np.zeros_like(sim.v)
        dudx[1:-1, 1:-1] = (sim.u[1:-1, 2:] - sim.u[1:-1, :-2]) * 0.5
        dudy[1:-1, 1:-1] = (sim.u[2:, 1:-1] - sim.u[:-2, 1:-1]) * 0.5
        dvdx[1:-1, 1:-1] = (sim.v[1:-1, 2:] - sim.v[1:-1, :-2]) * 0.5
        dvdy[1:-1, 1:-1] = (sim.v[2:, 1:-1] - sim.v[:-2, 1:-1]) * 0.5

        tau_xx = 2.0 * dynamic_viscosity * dudx
        tau_yy = 2.0 * dynamic_viscosity * dvdy
        tau_xy = dynamic_viscosity * (dudy + dvdx)
        shear_stress = np.sqrt(tau_xx**2 + 2.0 * tau_xy**2 + tau_yy**2)

        if getattr(sim, "wall_distance", None) is None:
            sim._update_wall_geometry()

        obstacle = sim.obstacle_mask
        wall_distance = (
            np.array(sim.wall_distance, dtype=float)
            if sim.wall_distance is not None
            else np.full_like(obstacle, np.nan, dtype=float)
        )
        surface_band = (
            (~obstacle)
            & np.isfinite(wall_distance)
            & (wall_distance > 0.0)
            & (wall_distance <= 1.5)
        )

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

                p_local = pressure[surface_band][valid] - pressure_reference
                pressure_force_x = -p_local * nx
                pressure_force_y = -p_local * ny

                tau_xx_local = tau_xx[surface_band][valid]
                tau_xy_local = tau_xy[surface_band][valid]
                tau_yy_local = tau_yy[surface_band][valid]

                # Viscous traction vector: t_visc = τ · n
                shear_force_x = tau_xx_local * nx + tau_xy_local * ny
                shear_force_y = tau_xy_local * nx + tau_yy_local * ny

                force_x = np.sum(pressure_force_x + shear_force_x)
                force_y = np.sum(pressure_force_y + shear_force_y)

                pressure_force_x_total = np.sum(pressure_force_x)
                pressure_force_y_total = np.sum(pressure_force_y)
                shear_force_x_total = np.sum(shear_force_x)
                shear_force_y_total = np.sum(shear_force_y)

                drag_direction = sim.freestream_direction
                total_drag = float(
                    force_x * drag_direction[0] + force_y * drag_direction[1])
                pressure_drag = float(
                    pressure_force_x_total * drag_direction[0] + pressure_force_y_total * drag_direction[1])
                shear_drag = float(
                    shear_force_x_total * drag_direction[0] + shear_force_y_total * drag_direction[1])

        return total_drag, pressure_drag, shear_drag, shear_stress

    def compute_drag_components_n(self, sim: FluidSimulation) -> tuple[float, float, float, np.ndarray]:
        total_drag_solver, pressure_drag_solver, shear_drag_solver, shear_stress = self.compute_surface_force_components(
            sim)
        density_scale = float(max(sim.density, 1e-9) /
                              max(self.reference_density_kg_m3, 1e-9))
        total_drag_n = total_drag_solver * self.drag_force_scale_n * density_scale
        pressure_drag_n = pressure_drag_solver * \
            self.drag_force_scale_n * density_scale
        shear_drag_n = shear_drag_solver * self.drag_force_scale_n * density_scale
        return total_drag_n, pressure_drag_n, shear_drag_n, shear_stress

    def compute_drag_components_n_magnitude(self, sim: FluidSimulation) -> tuple[float, float, float, np.ndarray]:
        """Return non-negative drag magnitudes for robust flight-state integration."""
        total_drag_n, pressure_drag_n, shear_drag_n, shear_stress = self.compute_drag_components_n(
            sim)
        return (
            max(float(total_drag_n), 0.0),
            max(float(pressure_drag_n), 0.0),
            max(float(shear_drag_n), 0.0),
            shear_stress,
        )

    def integrate_step(self, sim: FluidSimulation, dt: float, time_s: float) -> RocketState:
        atmosphere = self.atmosphere_at_altitude(self.state.altitude_m)
        sim.density = max(float(atmosphere.density_kg_m3), 1e-8)
        sim.viscosity = max(
            float(atmosphere.dynamic_viscosity_pa_s / max(atmosphere.density_kg_m3, 1e-8)),
            1e-12,
        )
        if getattr(sim, "compressible", False):
            sim.set_freestream_thermodynamics(
                density=atmosphere.density_kg_m3,
                temperature_k=atmosphere.temperature_k,
                pressure_pa=atmosphere.pressure_pa,
            )

        drag_n, _, _, _ = self.compute_drag_components_n_magnitude(sim)
        thrust_n = float(self.thrust_profile(float(time_s)))

        dt = float(max(dt, 0.0))
        if (
            self.specific_impulse_s is not None
            and dt > 0.0
            and thrust_n > 0.0
            and self.current_mass_kg > self.dry_mass_kg
        ):
            mass_flow_kgps = thrust_n / \
                (self.specific_impulse_s * self.gravity_mps2)
            propellant_burn = min(mass_flow_kgps * dt,
                                  self.current_mass_kg - self.dry_mass_kg)
            self.current_mass_kg -= propellant_burn

        active_mass_kg = max(self.current_mass_kg, 1e-6)
        net_force_n = thrust_n - drag_n - active_mass_kg * atmosphere.gravity_m_s2
        acceleration = net_force_n / active_mass_kg

        self.state.thrust_n = thrust_n
        self.state.drag_n = drag_n
        self.state.net_force_n = net_force_n
        self.state.acceleration_mps2 = acceleration
        self.state.mass_kg = active_mass_kg
        self.state.velocity_mps = max(
            self.state.velocity_mps + acceleration * dt, 0.0)
        self.state.altitude_m = max(
            0.0, self.state.altitude_m + self.state.velocity_mps * dt)
        return self.state

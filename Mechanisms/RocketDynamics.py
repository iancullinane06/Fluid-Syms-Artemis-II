from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from Mechanisms.FluidSimulation import FluidSimulation


@dataclass
class RocketState:
    mass_kg: float = 0.0
    velocity_mps: float = 0.0
    acceleration_mps2: float = 0.0
    thrust_n: float = 0.0
    drag_n: float = 0.0
    net_force_n: float = 0.0


class RocketDynamics:
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

        direction = np.asarray(flight_direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-12:
            raise ValueError("flight_direction cannot be zero")
        self.flight_direction = direction / direction_norm

        self.state = RocketState(
            mass_kg=self.current_mass_kg, thrust_n=float(self.thrust_profile(0.0)))

    def rocket_velocity_profile(self, _: float) -> np.ndarray:
        speed_solver = self.state.velocity_mps * self.sim_speed_scale
        return speed_solver * self.flight_direction

    @staticmethod
    def compute_surface_force_components(sim: FluidSimulation) -> tuple[float, float, float, np.ndarray]:
        pressure = sim.p
        dynamic_viscosity = max(sim.viscosity * sim.density, 1e-8)

        dudy_shear = np.zeros_like(sim.u)
        dvdx_shear = np.zeros_like(sim.v)
        dvdy = np.zeros_like(sim.v)
        dudy_shear[1:-1, 1:-1] = (sim.u[2:, 1:-1] - sim.u[:-2, 1:-1]) * 0.5
        dvdx_shear[1:-1, 1:-1] = (sim.v[1:-1, 2:] - sim.v[1:-1, :-2]) * 0.5
        dvdy[1:-1, 1:-1] = (sim.v[2:, 1:-1] - sim.v[:-2, 1:-1]) * 0.5
        shear_stress = dynamic_viscosity * \
            np.sqrt((dudy_shear + dvdx_shear) ** 2 + dvdy ** 2)

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

                p_local = pressure[surface_band][valid]
                pressure_force_x = -p_local * nx
                pressure_force_y = -p_local * ny

                tx = -ny
                ty = nx
                u_local = sim.u[surface_band][valid]
                v_local = sim.v[surface_band][valid]
                tangential_velocity = u_local * tx + v_local * ty

                # Wall shear stress = μ * dV_tan/dn ≈ μ * V_tan / d  (assuming no-slip at wall)
                # The force the fluid exerts on the body is in the direction of the near-wall
                # tangential velocity (fluid drags the surface in the direction it flows).
                # No minus sign — the shear force is +τ_w * t̂, not −τ_w * t̂.
                normal_distance = np.maximum(
                    wall_distance[surface_band][valid], 1.0)
                tangential_shear = dynamic_viscosity * tangential_velocity / normal_distance
                shear_force_x = tangential_shear * tx
                shear_force_y = tangential_shear * ty

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
        total_drag_n = total_drag_solver * self.drag_force_scale_n
        pressure_drag_n = pressure_drag_solver * self.drag_force_scale_n
        shear_drag_n = shear_drag_solver * self.drag_force_scale_n
        return total_drag_n, pressure_drag_n, shear_drag_n, shear_stress

    def integrate_step(self, sim: FluidSimulation, dt: float, time_s: float) -> RocketState:
        total_drag_n, _, _, _ = self.compute_drag_components_n(sim)
        drag_n = max(float(total_drag_n), 0.0)
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
        net_force_n = thrust_n - drag_n - active_mass_kg * self.gravity_mps2
        acceleration = net_force_n / active_mass_kg

        self.state.thrust_n = thrust_n
        self.state.drag_n = drag_n
        self.state.net_force_n = net_force_n
        self.state.acceleration_mps2 = acceleration
        self.state.mass_kg = active_mass_kg
        self.state.velocity_mps = max(
            self.state.velocity_mps + acceleration * dt, 0.0)
        return self.state

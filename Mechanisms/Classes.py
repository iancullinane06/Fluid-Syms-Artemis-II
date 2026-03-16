import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class RocketProfile:
    name: str
    mass: float
    thrust: float
    burn_time: float
    width: float  # Width of the rocket profile in meters
    height: float  # Height of the rocket profile in meters

    def generate_2d_profile(self, resolution):
        """
        Generate a 2D profile of the rocket based on its dimensions.

        Parameters:
            resolution (int): Number of points per meter.

        Returns:
            tuple: Meshgrid arrays (X, Y) and the 2D profile.
        """
        x = np.linspace(-self.width / 2, self.width /
                        2, int(self.width * resolution))
        y = np.linspace(0, self.height, int(self.height * resolution))
        X, Y = np.meshgrid(x, y)
        # Simplified profile (can be refined for actual shape)
        profile = np.ones_like(X)
        return X, Y, profile

    def get_2d_profile_polygon(
        self,
        center_x: float,
        center_y: float,
        body_fraction: float = 0.65,
        nose_points: int = 24,
        fore_spike_fraction: float = 0.0,
        fore_spike_half_width_fraction: float = 0.18,
    ):
        """
        Closed 2D rocket polygon (x, y):
        cylindrical body + rounded nose cone + optional fore spike.
        Uses self.height (length) and self.width (diameter).
        """
        body_len = self.height * body_fraction
        nose_len = self.height * (1.0 - body_fraction)
        radius = self.width / 2.0
        fore_spike_len = max(self.height * max(fore_spike_fraction, 0.0), 0.0)
        fore_spike_half = radius * \
            np.clip(fore_spike_half_width_fraction, 0.05, 0.45)

        x_tail = center_x - body_len / 2.0
        x_nose_base = center_x + body_len / 2.0

        # Nose arc: lower base -> tip -> upper base
        theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, nose_points)
        nose_x = x_nose_base + nose_len * np.cos(theta)
        nose_y = center_y + radius * np.sin(theta)
        tip_idx = int(np.argmax(nose_x))
        nose_tip_x = float(nose_x[tip_idx])
        fore_base_x = nose_tip_x - 0.06 * max(nose_len, 1e-9)

        if fore_spike_len > 0.0:
            spike_tip_x = nose_tip_x + fore_spike_len
            lower_x = nose_x[:tip_idx]
            lower_y = nose_y[:tip_idx]
            upper_x = nose_x[tip_idx + 1:]
            upper_y = nose_y[tip_idx + 1:]
            poly_x = np.array(
                [
                    x_tail,
                    x_nose_base,
                    *lower_x.tolist(),
                    fore_base_x,
                    spike_tip_x,
                    fore_base_x,
                    *upper_x.tolist(),
                    x_nose_base,
                    x_tail,
                    x_tail,
                ],
                dtype=float,
            )
            poly_y = np.array(
                [
                    center_y - radius,
                    center_y - radius,
                    *lower_y.tolist(),
                    center_y - fore_spike_half,
                    center_y,
                    center_y + fore_spike_half,
                    *upper_y.tolist(),
                    center_y + radius,
                    center_y + radius,
                    center_y - radius,
                ],
                dtype=float,
            )
        else:
            poly_x = np.array(
                [x_tail, x_nose_base, *nose_x.tolist(), x_nose_base,
                 x_tail, x_tail],
                dtype=float,
            )
            poly_y = np.array(
                [center_y - radius, center_y - radius, *nose_y.tolist(),
                 center_y + radius, center_y + radius, center_y - radius],
                dtype=float,
            )
        return poly_x, poly_y

    def get_2d_profile_mask(
        self,
        grid_size: tuple[int, int],   # (rows, cols)
        center_x: float,
        center_y: float,
        body_fraction: float = 0.65,
        nose_points: int = 24,
        fore_spike_fraction: float = 0.0,
        fore_spike_half_width_fraction: float = 0.18,
    ) -> np.ndarray:
        """Rasterized solid mask for the 2D rocket profile."""
        poly_x, poly_y = self.get_2d_profile_polygon(
            center_x,
            center_y,
            body_fraction,
            nose_points,
            fore_spike_fraction,
            fore_spike_half_width_fraction,
        )

        h, w = grid_size
        xg, yg = np.meshgrid(np.arange(w), np.arange(h))
        x = xg.astype(float)
        y = yg.astype(float)

        inside = np.zeros((h, w), dtype=bool)
        j = len(poly_x) - 1
        for i in range(len(poly_x)):
            xi, yi = poly_x[i], poly_y[i]
            xj, yj = poly_x[j], poly_y[j]
            crosses = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            )
            inside ^= crosses
            j = i

        return inside

    def __str__(self):
        return (f"RocketProfile(name={self.name}, mass={self.mass}, thrust={self.thrust}, "
                f"burn_time={self.burn_time}, width={self.width}, height={self.height})")


@dataclass(frozen=True)
class AtmosphereState:
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    speed_of_sound_m_s: float
    gravity_m_s2: float


@dataclass(frozen=True)
class FlightSample:
    time_s: float
    altitude_m: float
    velocity_m_s: float
    acceleration_m_s2: float
    thrust_n: float
    drag_n: float
    reynolds_number: float
    drag_coefficient: float
    mach_number: float
    dynamic_pressure_pa: float
    density_kg_m3: float
    temperature_k: float


@dataclass(frozen=True)
class NoseconeModel:
    key: str
    label: str
    drag_factor: float
    wave_drag_peak: float

from __future__ import annotations
import argparse
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Mechanisms.Classes import RocketProfile, AtmosphereState, FlightSample, NoseconeModel
from Mechanisms.Functions import calculate_drag_coefficient, calculate_reynolds_number

G0 = 9.80665
EARTH_RADIUS_M = 6_371_000.0
R_AIR = 287.05287
GAMMA_AIR = 1.4
SUTHERLAND_REFERENCE_T = 273.15
SUTHERLAND_REFERENCE_MU = 1.716e-5
SUTHERLAND_S = 110.4


NOSECONE_MODELS: dict[str, NoseconeModel] = {
	"artemis2_rounded_conic": NoseconeModel("artemis2_rounded_conic", "Artemis II Rounded-Conic", 0.90, 0.15),
	"elliptical": NoseconeModel("elliptical", "Elliptical", 0.86, 0.13),
	"conic": NoseconeModel("conic", "Conic", 1.00, 0.22),
	"bi_conic": NoseconeModel("bi_conic", "Bi-Conic", 0.95, 0.18),
	"ogive": NoseconeModel("ogive", "Ogive", 0.84, 0.11),
	"von_karman": NoseconeModel("von_karman", "Von Kármán", 0.80, 0.09),
}


def dynamic_viscosity_sutherland(temperature_k: float) -> float:
	temperature = max(float(temperature_k), 1.0)
	numerator = SUTHERLAND_REFERENCE_MU * (temperature / SUTHERLAND_REFERENCE_T) ** 1.5
	return numerator * (SUTHERLAND_REFERENCE_T + SUTHERLAND_S) / (temperature + SUTHERLAND_S)


def standard_atmosphere(altitude_m: float) -> AtmosphereState:
	"""
	1976 Standard Atmosphere approximation up to 84.852 km.
	"""
	hb = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0], dtype=float)
	lapse = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020], dtype=float)

	temperature_bases = [288.15]
	pressure_bases = [101_325.0]

	for index, lapse_rate in enumerate(lapse):
		delta_h = hb[index + 1] - hb[index]
		base_temperature = temperature_bases[index]
		base_pressure = pressure_bases[index]

		if abs(lapse_rate) < 1e-12:
			next_temperature = base_temperature
			exponent = -G0 * delta_h / (R_AIR * base_temperature)
			next_pressure = base_pressure * math.exp(exponent)
		else:
			next_temperature = base_temperature + lapse_rate * delta_h
			exponent = -G0 / (lapse_rate * R_AIR)
			next_pressure = base_pressure * (next_temperature / base_temperature) ** exponent

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
		pressure = base_pressure * math.exp(-G0 * delta_h / (R_AIR * temperature))
	else:
		temperature = base_temperature + lapse_rate * delta_h
		exponent = -G0 / (lapse_rate * R_AIR)
		pressure = base_pressure * (temperature / base_temperature) ** exponent

	density = pressure / (R_AIR * temperature)
	viscosity = dynamic_viscosity_sutherland(temperature)
	speed_of_sound = math.sqrt(GAMMA_AIR * R_AIR * temperature)
	gravity = G0 * (EARTH_RADIUS_M / (EARTH_RADIUS_M + altitude)) ** 2

	return AtmosphereState(
		altitude_m=altitude,
		temperature_k=temperature,
		pressure_pa=pressure,
		density_kg_m3=density,
		dynamic_viscosity_pa_s=viscosity,
		speed_of_sound_m_s=speed_of_sound,
		gravity_m_s2=gravity,
	)


def reference_area(profile: RocketProfile) -> float:
	radius = max(float(profile.width), 1e-6) / 2.0
	return math.pi * radius * radius


def fineness_ratio(profile: RocketProfile) -> float:
	return max(float(profile.height) / max(float(profile.width), 1e-6), 1e-6)


def profile_drag_adjustment(profile: RocketProfile) -> float:
	"""
	Applies a mild correction so slender profiles get lower drag than blunt ones.
	"""
	ratio = fineness_ratio(profile)
	return float(np.clip(1.18 - 0.085 * (ratio - 2.0), 0.28, 1.15))


def nosecone_drag_adjustment(nosecone_style: str) -> float:
	model = NOSECONE_MODELS.get(nosecone_style)
	if model is None:
		return 1.0
	return model.drag_factor


def nosecone_wave_drag(mach_number: float, nosecone_style: str) -> float:
	model = NOSECONE_MODELS.get(nosecone_style)
	if model is None:
		model = NOSECONE_MODELS["conic"]

	mach = max(float(mach_number), 0.0)
	peak = model.wave_drag_peak

	if mach <= 0.8:
		return 0.0
	if mach <= 1.2:
		t = (mach - 0.8) / 0.4
		smoothstep = t * t * (3.0 - 2.0 * t)
		return peak * smoothstep

	return peak / (1.0 + 0.35 * (mach - 1.2))


def reynolds_and_drag(
	profile: RocketProfile,
	velocity_m_s: float,
	atmosphere: AtmosphereState,
	nosecone_style: str = "artemis2_rounded_conic",
) -> tuple[float, float]:
	characteristic_length = max(float(profile.height), float(profile.width), 1e-6)
	reynolds_number = calculate_reynolds_number(
		density=atmosphere.density_kg_m3,
		velocity=max(abs(velocity_m_s), 1e-6),
		characteristic_length=characteristic_length,
		viscosity=atmosphere.dynamic_viscosity_pa_s,
	)
	mach_number = abs(velocity_m_s) / max(atmosphere.speed_of_sound_m_s, 1e-6)
	base_cd = calculate_drag_coefficient(reynolds_number)
	cd_after_profile = base_cd * profile_drag_adjustment(profile)
	cd_after_shape = cd_after_profile * nosecone_drag_adjustment(nosecone_style)
	wave_drag_cd = nosecone_wave_drag(mach_number, nosecone_style)
	adjusted_cd = max(0.015, cd_after_shape + wave_drag_cd)
	return reynolds_number, adjusted_cd


def drag_force(
	profile: RocketProfile,
	velocity_m_s: float,
	atmosphere: AtmosphereState,
	nosecone_style: str = "artemis2_rounded_conic",
) -> tuple[float, float, float]:
	reynolds_number, drag_coefficient = reynolds_and_drag(profile, velocity_m_s, atmosphere, nosecone_style=nosecone_style)
	area = reference_area(profile)
	drag = 0.5 * atmosphere.density_kg_m3 * drag_coefficient * area * velocity_m_s * abs(velocity_m_s)
	return drag, reynolds_number, drag_coefficient


def thrust_profile(time_s: float, propulsive_force_n: float, burn_time_s: float) -> float:
	return float(propulsive_force_n if time_s <= burn_time_s else 0.0)


def acceleration_along_path(
	profile: RocketProfile,
	altitude_m: float,
	velocity_m_s: float,
	thrust_n: float,
	nosecone_style: str = "artemis2_rounded_conic",
) -> tuple[float, float, float, AtmosphereState]:
	atmosphere = standard_atmosphere(altitude_m)
	drag_n, reynolds_number, drag_coefficient = drag_force(
		profile,
		velocity_m_s,
		atmosphere,
		nosecone_style=nosecone_style,
	)
	weight_n = float(profile.mass) * atmosphere.gravity_m_s2
	acceleration = (thrust_n - weight_n - drag_n) / float(profile.mass)
	return acceleration, reynolds_number, drag_coefficient, atmosphere


def solve_terminal_velocity(
	profile: RocketProfile,
	propulsive_force_n: float,
	altitude_m: float,
	nosecone_style: str = "artemis2_rounded_conic",
	initial_guess_m_s: float = 1.0,
	tolerance: float = 1e-4,
	max_iterations: int = 80,
) -> tuple[float, float, float] | None:
	atmosphere = standard_atmosphere(altitude_m)
	available_force = float(propulsive_force_n) - float(profile.mass) * atmosphere.gravity_m_s2
	if available_force <= 0.0:
		return None

	area = reference_area(profile)
	velocity = max(abs(initial_guess_m_s), 1.0)

	for _ in range(max_iterations):
		reynolds_number, drag_coefficient = reynolds_and_drag(
			profile,
			velocity,
			atmosphere,
			nosecone_style=nosecone_style,
		)
		next_velocity = math.sqrt((2.0 * available_force) / (atmosphere.density_kg_m3 * drag_coefficient * area))
		if abs(next_velocity - velocity) < tolerance:
			return next_velocity, reynolds_number, drag_coefficient
		velocity = next_velocity

	reynolds_number, drag_coefficient = reynolds_and_drag(
		profile,
		velocity,
		atmosphere,
		nosecone_style=nosecone_style,
	)
	return velocity, reynolds_number, drag_coefficient


def simulate_profile_ascent(
	profile: RocketProfile,
	propulsive_force_n: float,
	total_time_s: float,
	time_step_s: float,
	nosecone_style: str = "artemis2_rounded_conic",
	launch_altitude_m: float = 0.0,
) -> list[FlightSample]:
	samples: list[FlightSample] = []
	altitude = float(max(launch_altitude_m, 0.0))
	velocity = 0.0
	time_s = 0.0
	steps = max(int(math.ceil(total_time_s / time_step_s)), 1)

	for _ in range(steps + 1):
		thrust_n = thrust_profile(time_s, propulsive_force_n, float(profile.burn_time))
		acceleration, reynolds_number, drag_coefficient, atmosphere = acceleration_along_path(
			profile=profile,
			altitude_m=altitude,
			velocity_m_s=velocity,
			thrust_n=thrust_n,
			nosecone_style=nosecone_style,
		)
		drag_n, _, _ = drag_force(profile, velocity, atmosphere, nosecone_style=nosecone_style)
		mach_number = abs(velocity) / max(atmosphere.speed_of_sound_m_s, 1e-6)
		dynamic_pressure = 0.5 * atmosphere.density_kg_m3 * velocity * velocity

		samples.append(
			FlightSample(
				time_s=time_s,
				altitude_m=altitude,
				velocity_m_s=velocity,
				acceleration_m_s2=acceleration,
				thrust_n=thrust_n,
				drag_n=abs(drag_n),
				reynolds_number=reynolds_number,
				drag_coefficient=drag_coefficient,
				mach_number=mach_number,
				dynamic_pressure_pa=dynamic_pressure,
				density_kg_m3=atmosphere.density_kg_m3,
				temperature_k=atmosphere.temperature_k,
			)
		)

		velocity = velocity + acceleration * time_step_s
		altitude = max(0.0, altitude + velocity * time_step_s)
		time_s += time_step_s

	return samples


def summarise_drag_flight(
	profile: RocketProfile,
	propulsive_force_n: float,
	samples: list[FlightSample],
	nosecone_style: str = "artemis2_rounded_conic",
) -> dict[str, float | str | None]:
	launch_terminal = solve_terminal_velocity(
		profile,
		propulsive_force_n,
		altitude_m=0.0,
		nosecone_style=nosecone_style,
	)
	final_sample = samples[-1]
	powered_samples = [sample for sample in samples if sample.thrust_n > 0.0]
	powered_terminal = solve_terminal_velocity(
		profile,
		propulsive_force_n,
		altitude_m=final_sample.altitude_m,
		nosecone_style=nosecone_style,
		initial_guess_m_s=max(abs(final_sample.velocity_m_s), 1.0),
	)

	peak_drag_sample = max(samples, key=lambda sample: sample.drag_n)
	mean_drag = float(np.mean([sample.drag_n for sample in samples]))
	impulse_like_drag = float(np.trapezoid([sample.drag_n for sample in samples], [sample.time_s for sample in samples]))
	ballistic_coefficient = float(profile.mass) / max(
		final_sample.drag_coefficient * reference_area(profile),
		1e-9,
	)

	terminal_reached = False
	if powered_samples and launch_terminal is not None:
		target = launch_terminal[0]
		terminal_reached = any(
			abs(sample.velocity_m_s - target) <= max(0.03 * target, 1.0)
			and abs(sample.acceleration_m_s2) < 0.5
			for sample in powered_samples
		)

	return {
		"final_altitude_m": final_sample.altitude_m,
		"final_drag_n": final_sample.drag_n,
		"peak_drag_n": peak_drag_sample.drag_n,
		"peak_drag_time_s": peak_drag_sample.time_s,
		"mean_drag_n": mean_drag,
		"drag_time_integral_n_s": impulse_like_drag,
		"ballistic_coefficient": ballistic_coefficient,
		"launch_terminal_velocity_m_s": None if launch_terminal is None else launch_terminal[0],
		"launch_terminal_re": None if launch_terminal is None else launch_terminal[1],
		"launch_terminal_cd": None if launch_terminal is None else launch_terminal[2],
		"final_altitude_terminal_velocity_m_s": None if powered_terminal is None else powered_terminal[0],
		"final_altitude_terminal_re": None if powered_terminal is None else powered_terminal[1],
		"final_altitude_terminal_cd": None if powered_terminal is None else powered_terminal[2],
		"terminal_reached_during_burn": "yes" if terminal_reached else "no",
	}


def print_drag_summary(
	profile: RocketProfile,
	propulsive_force_n: float,
	samples: list[FlightSample],
	nosecone_style: str = "artemis2_rounded_conic",
) -> None:
	summary = summarise_drag_flight(profile, propulsive_force_n, samples, nosecone_style=nosecone_style)
	sea_level = standard_atmosphere(0.0)
	final_altitude_m = float(samples[-1].altitude_m)
	final_atmosphere = standard_atmosphere(final_altitude_m)

	model = NOSECONE_MODELS.get(nosecone_style, NOSECONE_MODELS["conic"])
	print(f"\nProfile: {profile.name} | Nosecone: {model.label}")
	print(f"Mass: {profile.mass:.2f} kg | Diameter: {profile.width:.3f} m | Length: {profile.height:.3f} m")
	print(f"F_prop: {propulsive_force_n:.2f} N | Burn time: {profile.burn_time:.2f} s")
	print(f"Reference area: {reference_area(profile):.4f} m^2 | Fineness ratio: {fineness_ratio(profile):.2f}")
	print(f"Sea-level density: {sea_level.density_kg_m3:.4f} kg/m^3 | Sea-level viscosity: {sea_level.dynamic_viscosity_pa_s:.3e} Pa·s")
	print(f"Final altitude density: {final_atmosphere.density_kg_m3:.4f} kg/m^3 | Gravity there: {final_atmosphere.gravity_m_s2:.4f} m/s^2")
	print("\nDrag-focused flight summary")
	print(f"- Final altitude: {summary['final_altitude_m']:.2f} m")
	print(f"- Final drag force: {summary['final_drag_n']:.2f} N")
	print(f"- Peak drag force: {summary['peak_drag_n']:.2f} N at t={summary['peak_drag_time_s']:.2f} s")
	print(f"- Mean drag force: {summary['mean_drag_n']:.2f} N")
	print(f"- Drag time-integral: {summary['drag_time_integral_n_s']:.2f} N·s")
	print(f"- Ballistic coefficient: {summary['ballistic_coefficient']:.2f} kg/m^2")

	if summary["launch_terminal_velocity_m_s"] is None:
		print("- Sea-level powered terminal velocity: unavailable (F_prop does not exceed weight)")
	else:
		print(
			"- Sea-level powered terminal velocity: "
			f"{summary['launch_terminal_velocity_m_s']:.2f} m/s "
			f"(Re={summary['launch_terminal_re']:.3e}, Cd={summary['launch_terminal_cd']:.3f})"
		)

	if summary["final_altitude_terminal_velocity_m_s"] is None:
		print("- Final-altitude powered terminal velocity: unavailable (F_prop does not exceed weight)")
	else:
		print(
			"- Final-altitude powered terminal velocity: "
			f"{summary['final_altitude_terminal_velocity_m_s']:.2f} m/s "
			f"(Re={summary['final_altitude_terminal_re']:.3e}, Cd={summary['final_altitude_terminal_cd']:.3f})"
		)

	print(f"- Terminal velocity reached during burn: {summary['terminal_reached_during_burn']}")
	print("\nSampled drag history (roughly 12 points)")

	stride = max(len(samples) // 12, 1)
	for sample in samples[::stride]:
		print(
			f"t={sample.time_s:6.2f} s | alt={sample.altitude_m:9.2f} m | "
			f"drag={sample.drag_n:11.2f} N | a={sample.acceleration_m_s2:7.3f} m/s^2 | "
			f"Re={sample.reynolds_number:9.3e} | Cd={sample.drag_coefficient:6.3f} | "
			f"rho={sample.density_kg_m3:6.3f} kg/m^3"
		)


def plot_drag_history(samples: list[FlightSample], title: str) -> None:
	sns.set_theme(style="darkgrid", palette="Blues")
	times = np.array([sample.time_s for sample in samples], dtype=float)
	drag = np.array([sample.drag_n for sample in samples], dtype=float)

	fig, ax = plt.subplots(figsize=(10, 5.5))
	ax.plot(times, drag, linewidth=2.4, marker="o", markersize=3, alpha=0.8, color="#1f77b4")
	ax.set_xlabel("Time (s)", fontsize=11)
	ax.set_ylabel("Drag Force (N)", fontsize=11)
	ax.set_title(title, fontsize=13, fontweight="bold")
	fig.tight_layout()
	plt.show()


def plot_drag_comparison(results: dict[str, list[FlightSample]]) -> None:
	sns.set_theme(style="darkgrid", palette="Blues")
	styles = list(results.keys())
	palette = sns.color_palette("Blues", len(styles))

	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle("Rocket Flight Analysis: 6 Nosecone Designs", fontsize=16, fontweight="bold", y=0.995)

	# Subplot 1: Drag Force vs Time
	ax1 = axes[0, 0]
	for color, style in zip(palette, styles):
		samples = results[style]
		times = np.array([sample.time_s for sample in samples], dtype=float)
		drag = np.array([sample.drag_n for sample in samples], dtype=float)
		label = NOSECONE_MODELS.get(style, NoseconeModel(style, style, 1.0, 0.1)).label
		ax1.plot(times, drag, linewidth=2.3, color=color, label=label, marker="o", markersize=2, alpha=0.8)
	ax1.set_xlabel("Time (s)", fontsize=10)
	ax1.set_ylabel("Drag Force (N)", fontsize=10)
	ax1.set_title("Drag Force vs Time", fontsize=11, fontweight="bold")
	ax1.legend(loc="upper left", fontsize=8)

	# Subplot 2: Reynolds Number vs Time
	ax2 = axes[0, 1]
	for color, style in zip(palette, styles):
		samples = results[style]
		times = np.array([sample.time_s for sample in samples], dtype=float)
		reynolds = np.array([sample.reynolds_number for sample in samples], dtype=float)
		label = NOSECONE_MODELS.get(style, NoseconeModel(style, style, 1.0, 0.1)).label
		ax2.plot(times, reynolds, linewidth=2.3, color=color, label=label, marker="s", markersize=2, alpha=0.8)
	ax2.set_xlabel("Time (s)", fontsize=10)
	ax2.set_ylabel("Reynolds Number", fontsize=10)
	ax2.set_title("Reynolds Number vs Time", fontsize=11, fontweight="bold")
	ax2.set_yscale("log")

	# Subplot 3: Drag Coefficient vs Velocity
	ax3 = axes[1, 0]
	for color, style in zip(palette, styles):
		samples = results[style]
		velocities = np.array([sample.velocity_m_s for sample in samples], dtype=float)
		drag_coeff = np.array([sample.drag_coefficient for sample in samples], dtype=float)
		label = NOSECONE_MODELS.get(style, NoseconeModel(style, style, 1.0, 0.1)).label
		ax3.plot(velocities, drag_coeff, linewidth=2.3, color=color, label=label, marker="^", markersize=2, alpha=0.8)
	ax3.set_xlabel("Velocity (m/s)", fontsize=10)
	ax3.set_ylabel("Drag Coefficient (Cd)", fontsize=10)
	ax3.set_title("Drag Coefficient vs Velocity", fontsize=11, fontweight="bold")

	# Subplot 4: Mach Number vs Time
	ax4 = axes[1, 1]
	for color, style in zip(palette, styles):
		samples = results[style]
		times = np.array([sample.time_s for sample in samples], dtype=float)
		mach = np.array([sample.mach_number for sample in samples], dtype=float)
		label = NOSECONE_MODELS.get(style, NoseconeModel(style, style, 1.0, 0.1)).label
		ax4.plot(times, mach, linewidth=2.3, color=color, label=label, marker="D", markersize=2, alpha=0.8)
	ax4.set_xlabel("Time (s)", fontsize=10)
	ax4.set_ylabel("Mach Number", fontsize=10)
	ax4.set_title("Mach Number vs Time", fontsize=11, fontweight="bold")
	ax4.legend(loc="upper left", fontsize=8)
	ax4.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="Mach 1 (Sonic)")

	fig.tight_layout()
	plt.show()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Simulate rocket drag and ascent")
	parser.add_argument("--name", type=str, default="Artemis II")
	parser.add_argument("--mass", type=float, default=2_600_000.0, help="Launch mass in kg")
	parser.add_argument("--width", type=float, default=8.4, help="Vehicle diameter in m")
	parser.add_argument("--height", type=float, default=98.0, help="Vehicle height in m")
	parser.add_argument("--thrust", type=float, default=39_000_000.0, help="Liftoff thrust in N")
	parser.add_argument("--burn-time", type=float, default=480.0, help="Burn time in s")
	parser.add_argument("--duration", type=float, default=300.0, help="Total simulation duration in s")
	parser.add_argument("--launch-altitude", type=float, default=0.0, help="Initial launch altitude in m")
	parser.add_argument("--dt", type=float, default=0.5, help="Time step for simulation in s")
	parser.add_argument(
		"--mode",
		type=str,
		choices=["single", "compare"],
		default="compare",
		help="Run a single nosecone simulation or compare all built-in nosecone designs",
	)
	parser.add_argument(
		"--nosecone",
		type=str,
		choices=list(NOSECONE_MODELS.keys()),
		default="artemis2_rounded_conic",
		help="Nosecone model used in single mode",
	)
	plot_group = parser.add_mutually_exclusive_group()
	plot_group.add_argument("--plot", dest="plot", action="store_true", help="Plot the simulation results")
	plot_group.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plotting")
	parser.set_defaults(plot=True)

	return parser


def main() -> None:
	args = build_parser().parse_args()
	profile = RocketProfile(
		name=args.name,
		mass=args.mass,
		thrust=args.thrust,
		burn_time=args.burn_time,
		width=args.width,
		height=args.height,
	)

	if args.mode == "single":
		samples = simulate_profile_ascent(
			profile=profile,
			propulsive_force_n=args.thrust,
			total_time_s=args.duration,
			time_step_s=args.dt,
			nosecone_style=args.nosecone,
			launch_altitude_m=args.launch_altitude,
		)
		print_drag_summary(profile, args.thrust, samples, nosecone_style=args.nosecone)
		if args.plot:
			model_label = NOSECONE_MODELS.get(args.nosecone, NoseconeModel(args.nosecone, args.nosecone, 1.0, 0.1)).label
			plot_drag_history(samples, title=f"Drag Force vs Time | {model_label}")
		return

	comparison_styles = [
		"artemis2_rounded_conic",
		"elliptical",
		"conic",
		"bi_conic",
		"ogive",
		"von_karman",
	]
	all_results: dict[str, list[FlightSample]] = {}

	for style in comparison_styles:
		samples = simulate_profile_ascent(
			profile=profile,
			propulsive_force_n=args.thrust,
			total_time_s=args.duration,
			time_step_s=args.dt,
			nosecone_style=style,
			launch_altitude_m=args.launch_altitude,
		)
		all_results[style] = samples
		print_drag_summary(profile, args.thrust, samples, nosecone_style=style)

	if args.plot:
		plot_drag_comparison(all_results)


if __name__ == "__main__":
	main()
 
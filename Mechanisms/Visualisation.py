import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.widgets import Slider


def _safe_tag(tag: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in tag)


def _build_video_basename(
    video_basename: str,
    compressible_flux_scheme: str,
    inlet_mach: float,
    video_case_tag: str,
) -> str:
    mach_tag = f"m{inlet_mach:.2f}".replace(".", "p")
    return f"{video_basename}_{compressible_flux_scheme}_{mach_tag}_{_safe_tag(video_case_tag)}"


def _resolve_ffmpeg_executable() -> str | None:
    env_path = os.environ.get("FFMPEG_PATH")
    candidates = [
        env_path,
        shutil.which("ffmpeg"),
        shutil.which("ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _save_simulation_video(
    fig: Any,
    draw_frame,
    num_frames: int,
    video_fps: int,
    video_dpi: int,
    video_basename: str,
    compressible_flux_scheme: str,
    inlet_mach: float,
    video_case_tag: str,
    export_gif_copy: bool,
) -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    animation = FuncAnimation(
        fig,
        lambda index: draw_frame(index) or [],
        frames=num_frames,
        interval=1000.0 / float(max(video_fps, 1)),
        blit=False,
        repeat=False,
    )

    basename = _build_video_basename(
        video_basename=video_basename,
        compressible_flux_scheme=compressible_flux_scheme,
        inlet_mach=inlet_mach,
        video_case_tag=video_case_tag,
    )
    mp4_path = output_dir / f"{basename}.mp4"
    gif_path = output_dir / f"{basename}.gif"

    ffmpeg_path = _resolve_ffmpeg_executable()
    saved_mp4 = False

    if ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        try:
            writer = FFMpegWriter(
                fps=video_fps,
                codec="libx264",
                bitrate=5000,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            animation.save(str(mp4_path), writer=writer, dpi=video_dpi)
            print(f"Saved video: {mp4_path}")
            saved_mp4 = True
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"FFmpeg export failed ({exc}).")
    else:
        print("FFmpeg executable not found (set FFMPEG_PATH or install ffmpeg).")

    if export_gif_copy or not saved_mp4:
        gif_writer = PillowWriter(fps=video_fps)
        animation.save(str(gif_path), writer=gif_writer, dpi=video_dpi)
        print(f"Saved GIF: {gif_path}")


def build_and_render_visualisations(
    simulation: Any,
    frames: list[dict[str, Any]],
    drag_history: list[tuple[Any, ...]],
    profile_x: np.ndarray,
    profile_y: np.ndarray,
    rows: int,
    cols: int,
    num_frames: int,
    sim_speed_scale: float,
    figure_height_scale: float,
    limits: dict[str, float],
    export_video: bool,
    show_plot: bool,
    video_fps: int,
    video_dpi: int,
    video_basename: str,
    export_gif_copy: bool,
    video_case_tag: str,
    compressible_flux_scheme: str,
    inlet_mach: float,
) -> None:
    max_dim = max(rows, cols)
    fig_width = 16.0 * cols / max_dim
    fig_height = 10.8 * rows / max_dim * figure_height_scale
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.35,
        wspace=0.30,
        bottom=0.12,
        top=0.95,
        left=0.08,
        right=0.96,
    )

    ax_speed = fig.add_subplot(gs[0, 0])
    ax_pressure = fig.add_subplot(gs[0, 1])
    ax_streamwise = fig.add_subplot(gs[0, 2])
    ax_vortex = fig.add_subplot(gs[1, 0])
    ax_drag = fig.add_subplot(gs[1, 1])
    ax_drag_altitude = ax_drag.twinx()
    ax_shear = fig.add_subplot(gs[1, 2])

    colorbars: dict[str, Any] = {}

    def draw_frame(frame_index: int):
        for axis in [ax_speed, ax_pressure, ax_streamwise, ax_vortex, ax_shear]:
            axis.clear()

        ax_drag.clear()
        ax_drag_altitude.clear()

        frame = frames[frame_index]
        speed_plot = frame["speed_plot"]
        pressure_plot = frame["pressure_plot"]
        frame_time = frame["frame_time"]
        frame_speed = frame["frame_speed"]
        frame_mach = frame["frame_mach"]
        drag_total = frame["drag_total"]
        drag_pressure = frame["drag_pressure"]
        drag_shear = frame["drag_shear"]
        drag_total_signed = frame["drag_total_signed"]
        frame_thrust_n = frame["frame_thrust_n"]
        frame_net_force_n = frame["frame_net_force_n"]
        frame_rocket_speed_mps = frame["frame_rocket_speed_mps"]

        ax_speed.set_facecolor("white")
        im_speed = ax_speed.imshow(
            speed_plot,
            origin="lower",
            cmap="viridis",
            vmin=limits["global_min_speed"],
            vmax=limits["global_max_speed"],
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

        ax_pressure.set_facecolor("white")
        pressure_vmin = limits["global_min_pressure"]
        pressure_vmax = limits["global_max_pressure"]
        pressure_title = "Pressure Field"
        if simulation.compressible:
            pressure_min = np.nanpercentile(pressure_plot, 1)
            pressure_max = np.nanpercentile(pressure_plot, 99)
            pressure_lim = max(abs(pressure_min), abs(pressure_max), 1.0)
            pressure_vmin = -pressure_lim
            pressure_vmax = pressure_lim
            pressure_title = "Gauge Pressure Field"

        im_pressure = ax_pressure.imshow(
            pressure_plot,
            origin="lower",
            cmap="RdBu_r",
            vmin=pressure_vmin,
            vmax=pressure_vmax,
            extent=(0, cols - 1, 0, rows - 1),
            alpha=0.75,
            interpolation="bilinear",
        )
        ax_pressure.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
        ax_pressure.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
        ax_pressure.set(aspect=1, title=pressure_title)
        ax_pressure.set_xlabel("X", fontsize=9)
        ax_pressure.set_ylabel("Y", fontsize=9)
        ax_pressure.set_xlim(0, cols - 1)
        ax_pressure.set_ylim(0, rows - 1)
        if "pressure" not in colorbars:
            colorbars["pressure"] = plt.colorbar(im_pressure, ax=ax_pressure, fraction=0.046, pad=0.04)
            colorbars["pressure"].set_label("Pa", fontsize=8)
        else:
            colorbars["pressure"].update_normal(im_pressure)

        if simulation.compressible:
            mach_plot = frame["mach_plot"]
            density_plot = frame["density_plot"]
            temperature_plot = frame["temperature_plot"]

            mach_vmin = np.nanpercentile(mach_plot, 1)
            mach_vmax = np.nanpercentile(mach_plot, 99)
            if not np.isfinite(mach_vmin) or not np.isfinite(mach_vmax) or mach_vmax <= mach_vmin:
                mach_vmin = limits["global_min_mach"]
                mach_vmax = limits["global_max_mach"]
            if mach_vmax <= mach_vmin:
                mach_vmax = mach_vmin + 1e-6

            ax_streamwise.set_facecolor("white")
            im_streamwise = ax_streamwise.imshow(
                mach_plot,
                origin="lower",
                cmap="plasma",
                vmin=mach_vmin,
                vmax=mach_vmax,
                extent=(0, cols - 1, 0, rows - 1),
                alpha=0.75,
                interpolation="bilinear",
            )
            ax_streamwise.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
            ax_streamwise.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
            ax_streamwise.set(aspect=1, title="Local Mach Number")
            ax_streamwise.set_xlabel("X", fontsize=9)
            ax_streamwise.set_ylabel("Y", fontsize=9)
            ax_streamwise.set_xlim(0, cols - 1)
            ax_streamwise.set_ylim(0, rows - 1)
            if "streamwise" not in colorbars:
                colorbars["streamwise"] = plt.colorbar(
                    im_streamwise,
                    ax=ax_streamwise,
                    fraction=0.046,
                    pad=0.04,
                )
                colorbars["streamwise"].set_label("M", fontsize=8)
            else:
                colorbars["streamwise"].update_normal(im_streamwise)

            ax_vortex.set_facecolor("white")
            density_vmin = np.nanpercentile(density_plot, 1)
            density_vmax = np.nanpercentile(density_plot, 99)
            if not np.isfinite(density_vmin) or not np.isfinite(density_vmax) or density_vmax <= density_vmin:
                density_vmin = limits["global_min_density"]
                density_vmax = limits["global_max_density"]
            if density_vmax <= density_vmin:
                density_vmax = density_vmin + 1e-6

            im_vortex = ax_vortex.imshow(
                density_plot,
                origin="lower",
                cmap="cividis",
                vmin=density_vmin,
                vmax=density_vmax,
                extent=(0, cols - 1, 0, rows - 1),
                alpha=0.75,
                interpolation="bilinear",
            )
            ax_vortex.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
            ax_vortex.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
            ax_vortex.set(aspect=1, title="Density Field")
            ax_vortex.set_xlabel("X", fontsize=9)
            ax_vortex.set_ylabel("Y", fontsize=9)
            ax_vortex.set_xlim(0, cols - 1)
            ax_vortex.set_ylim(0, rows - 1)
            if "vortex" not in colorbars:
                colorbars["vortex"] = plt.colorbar(im_vortex, ax=ax_vortex, fraction=0.046, pad=0.04)
                colorbars["vortex"].set_label("kg/m³", fontsize=8)
            else:
                colorbars["vortex"].update_normal(im_vortex)
        else:
            streamwise_plot = frame["streamwise_plot"]
            vorticity_plot = frame["vorticity_plot"]

            ax_streamwise.set_facecolor("white")
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
            ax_streamwise.set(aspect=1, title="Streamwise Velocity")
            ax_streamwise.set_xlabel("X", fontsize=9)
            ax_streamwise.set_ylabel("Y", fontsize=9)
            ax_streamwise.set_xlim(0, cols - 1)
            ax_streamwise.set_ylim(0, rows - 1)
            if "streamwise" not in colorbars:
                colorbars["streamwise"] = plt.colorbar(
                    im_streamwise,
                    ax=ax_streamwise,
                    fraction=0.046,
                    pad=0.04,
                )
                colorbars["streamwise"].set_label("m/s", fontsize=8)
            else:
                colorbars["streamwise"].update_normal(im_streamwise)

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

        history_time = np.array([item[0] for item in drag_history[: frame_index + 1]], dtype=float)
        history_drag_total = np.array([item[1] for item in drag_history[: frame_index + 1]], dtype=float)
        history_drag_pressure = np.array([item[2] for item in drag_history[: frame_index + 1]], dtype=float)
        history_drag_shear = np.array([item[3] for item in drag_history[: frame_index + 1]], dtype=float)
        history_drag_total_signed = np.array([item[4] for item in drag_history[: frame_index + 1]], dtype=float)
        history_thrust = np.array([item[7] for item in drag_history[: frame_index + 1]], dtype=float)
        history_net = np.array([item[8] for item in drag_history[: frame_index + 1]], dtype=float)
        history_altitude = np.array([item[9] for item in drag_history[: frame_index + 1]], dtype=float)

        ax_drag.plot(history_time, history_drag_total, color="#1f77b4", linewidth=2.0, label="Total")
        ax_drag.plot(
            history_time,
            history_drag_total_signed,
            color="#17becf",
            linewidth=1.1,
            linestyle="-.",
            label="Signed Total",
        )
        ax_drag.plot(
            history_time,
            history_drag_pressure,
            color="#d62728",
            linewidth=1.3,
            linestyle="--",
            label="Pressure",
        )
        ax_drag.plot(
            history_time,
            history_drag_shear,
            color="#2ca02c",
            linewidth=1.3,
            linestyle="--",
            label="Shear",
        )
        ax_drag.plot(history_time, history_thrust, color="#9467bd", linewidth=1.5, linestyle=":", label="Thrust")
        ax_drag.plot(
            history_time,
            history_net,
            color="#ff7f0e",
            linewidth=1.5,
            linestyle="-.",
            label="Net (T-D-W)",
        )
        ax_drag.scatter([frame_time], [drag_total_signed], color="black", s=34, zorder=3)

        ax_drag_altitude.plot(history_time, history_altitude, color="#111111", linewidth=1.4, linestyle="-", label="Altitude")
        ax_drag_altitude.set_ylabel("Altitude (m)", fontsize=9)

        if history_time.size > 0:
            x_min = float(np.min(history_time))
            x_max = float(np.max(history_time))
            x_pad = max(0.05 * (x_max - x_min), 1e-3)
            ax_drag.set_xlim(x_min - x_pad, x_max + x_pad)
        else:
            ax_drag.set_xlim(0.0, 1.0)

        combined_forces = (
            np.concatenate((
                history_drag_total,
                history_drag_total_signed,
                history_drag_pressure,
                history_drag_shear,
                history_thrust,
                history_net,
            ))
            if history_drag_total.size > 0
            else np.array([0.0])
        )
        y_min = float(np.min(combined_forces))
        y_max = float(np.max(combined_forces))
        y_span = max(y_max - y_min, 1e-6)
        ax_drag.set_ylim(y_min - 0.1 * y_span, y_max + 0.1 * y_span)
        ax_drag.set_ylabel("Force (N)", fontsize=9)
        ax_drag.set_xlabel("Time (s)", fontsize=9)
        ax_drag.grid(True, alpha=0.3)
        ax_drag.set_title("Forces & Altitude vs Time", fontsize=10)
        force_handles, force_labels = ax_drag.get_legend_handles_labels()
        altitude_handles, altitude_labels = ax_drag_altitude.get_legend_handles_labels()
        ax_drag.legend(force_handles + altitude_handles, force_labels + altitude_labels, fontsize=8, loc="best")

        if simulation.compressible:
            temperature_plot = frame["temperature_plot"]
            temperature_vmin = np.nanpercentile(temperature_plot, 1)
            temperature_vmax = np.nanpercentile(temperature_plot, 99)
            if (
                not np.isfinite(temperature_vmin)
                or not np.isfinite(temperature_vmax)
                or temperature_vmax <= temperature_vmin
            ):
                temperature_vmin = limits["global_min_temperature"]
                temperature_vmax = limits["global_max_temperature"]
            if temperature_vmax <= temperature_vmin:
                temperature_vmax = temperature_vmin + 1e-6

            ax_shear.set_facecolor("white")
            im_shear = ax_shear.imshow(
                temperature_plot,
                origin="lower",
                cmap="inferno",
                vmin=temperature_vmin,
                vmax=temperature_vmax,
                extent=(0, cols - 1, 0, rows - 1),
                alpha=0.75,
                interpolation="bilinear",
            )
            ax_shear.fill(profile_x, profile_y, color="cornflowerblue", alpha=0.35, zorder=6)
            ax_shear.plot(profile_x, profile_y, color="blue", linewidth=1.5, zorder=7)
            ax_shear.set(aspect=1, title="Temperature Field")
            ax_shear.set_xlabel("X", fontsize=9)
            ax_shear.set_ylabel("Y", fontsize=9)
            ax_shear.set_xlim(0, cols - 1)
            ax_shear.set_ylim(0, rows - 1)
            if "shear" not in colorbars:
                colorbars["shear"] = plt.colorbar(im_shear, ax=ax_shear, fraction=0.046, pad=0.04)
                colorbars["shear"].set_label("K", fontsize=8)
            else:
                colorbars["shear"].update_normal(im_shear)
        else:
            shear_plot = frame["shear_plot"]

            ax_shear.set_facecolor("white")
            im_shear = ax_shear.imshow(
                shear_plot,
                origin="lower",
                cmap="hot",
                vmin=0.0,
                vmax=limits["global_max_shear"],
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

        freestream_mps = frame_speed / sim_speed_scale
        if simulation.compressible:
            fig.suptitle(
                (
                    f"Rocket-Frame Compressible CFD | t={frame_time:.2f}s | "
                    f"U∞={frame_speed:.1f} m/s | M∞≈{frame_mach:.2f} | "
                    f"Vrocket={frame_rocket_speed_mps:.1f} m/s"
                ),
                fontsize=12,
                fontweight="bold",
            )
        else:
            fig.suptitle(
                (
                    f"Rocket-Frame CFD Analysis | t={frame_time:.2f}s | "
                    f"U∞={frame_speed:.2f} (solver) ≈ {freestream_mps:.1f} m/s | "
                    f"Vrocket={frame_rocket_speed_mps:.1f} m/s"
                ),
                fontsize=12,
                fontweight="bold",
            )

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

    def on_slider_change(value: float):
        draw_frame(int(value))
        fig.canvas.draw_idle()

    time_slider.on_changed(on_slider_change)

    if export_video:
        _save_simulation_video(
            fig=fig,
            draw_frame=draw_frame,
            num_frames=num_frames,
            video_fps=video_fps,
            video_dpi=video_dpi,
            video_basename=video_basename,
            compressible_flux_scheme=compressible_flux_scheme,
            inlet_mach=inlet_mach,
            video_case_tag=video_case_tag,
            export_gif_copy=export_gif_copy,
        )

    if show_plot:
        plt.show()

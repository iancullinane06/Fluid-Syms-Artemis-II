import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.widgets import Slider
from Mechanisms.FluidSimulation import FluidSimulation
from Mechanisms.Graphing import format_duration, update_progress_bar

# ─────────────────────────────────────────────
#  Simulation config
# ─────────────────────────────────────────────
GAMMA           = 1.4
GAS_CONSTANT    = 287.05
REF_TEMPERATURE = 288.15
REF_PRESSURE    = 101325.0
REF_DENSITY     = REF_PRESSURE / (GAS_CONSTANT * REF_TEMPERATURE)

INLET_MACH      = 0.2       # subsonic inlet — nozzle does the rest
CFL_NUMBER      = 0.45
FLUX_SCHEME     = "hllc"
INLET_TOTAL_PRESSURE = REF_PRESSURE * 1.25

# For this geometry (Ae/At ~ 3.6), a fully supersonic exit needs very low pb/p0.
BACK_PRESSURE_RATIO  = 0.035
BACK_PRESSURE        = INLET_TOTAL_PRESSURE * BACK_PRESSURE_RATIO

sim_time        = 6.0
frame_interval  = 0.5
num_frames      = int(sim_time / frame_interval)
time_step       = 0.02

EXPORT_VIDEO    = True
SHOW_PLOT       = True
VIDEO_FPS       = 20
VIDEO_DPI       = 130
VIDEO_BASENAME  = "nozzle_sim"

# ─────────────────────────────────────────────
#  Grid
# ─────────────────────────────────────────────
ROWS = 120
COLS = 400

# ─────────────────────────────────────────────
#  Nozzle geometry
# ─────────────────────────────────────────────
def make_nozzle_mask(
    rows: int,
    cols: int,
    throat_fraction: float = 0.18,
    inlet_fraction:  float = 0.42,
    exit_fraction:   float = 0.65,
) -> np.ndarray:
    """
    Boolean obstacle mask for a 2-D planar converging-diverging nozzle.
    True  = solid wall
    False = fluid
    Fractions are of the half-height of the grid.
    """
    mask     = np.zeros((rows, cols), dtype=bool)
    half_h   = rows / 2.0

    inlet_half  = inlet_fraction  * half_h
    throat_half = throat_fraction * half_h
    exit_half   = exit_fraction   * half_h

    cx = rows // 2   # centreline row index

    for col in range(cols):
        x = col / (cols - 1)   # 0 → 1

        if x < 0.4:
            # Converging section  (0 → throat at x=0.4)
            t = x / 0.4
            hw = throat_half + (inlet_half - throat_half) * 0.5 * (1.0 + np.cos(np.pi * t))
        else:
            # Diverging section  (throat at x=0.4 → exit at x=1)
            t = (x - 0.4) / 0.6
            hw = throat_half + (exit_half - throat_half) * 0.5 * (1.0 - np.cos(np.pi * t))

        hw = max(int(round(hw)), 2)
        mask[: cx - hw, col] = True   # top wall
        mask[cx + hw :, col] = True   # bottom wall

    return mask


nozzle_mask = make_nozzle_mask(ROWS, COLS)

# ─────────────────────────────────────────────
#  Fluid simulation
# ─────────────────────────────────────────────
inlet_sound_speed = np.sqrt(GAMMA * REF_PRESSURE / REF_DENSITY)
inlet_speed       = INLET_MACH * inlet_sound_speed

simulation = FluidSimulation(
    grid_size              = (ROWS, COLS),
    viscosity              = 1.48e-5,        # air kinematic viscosity
    time_step              = time_step,
    density                = REF_DENSITY,
    acceleration           = 0.0,
    acceleration_direction = (1.0, 0.0),
    edge_speed             = inlet_speed,
    assume_laminar_edges   = True,
    edge_relaxation        = 0.92,
    wall_penalty           = 24.0,
    wall_shear_layers      = 4,
    turbulence_strength    = 0.8,
    vorticity_confinement  = 0.0,
    compressible           = True,
    gamma                  = GAMMA,
    gas_constant           = GAS_CONSTANT,
    reference_temperature  = REF_TEMPERATURE,
    cfl_number             = CFL_NUMBER,
    compressible_flux_scheme = FLUX_SCHEME,
    compressible_velocity_diffusion = 0.0,
)

simulation.set_freestream_thermodynamics(
    density        = REF_DENSITY,
    temperature_k  = REF_TEMPERATURE,
    pressure_pa    = REF_PRESSURE,
)
simulation.set_nozzle_pressure_boundary(
    enabled=True,
    inlet_total_pressure=INLET_TOTAL_PRESSURE,
    inlet_total_temperature=REF_TEMPERATURE,
    outlet_static_pressure=BACK_PRESSURE,
)

# Register nozzle walls — reuses add_rocket_profile's mask path
simulation.add_rocket_profile(mask=nozzle_mask)

# Seed the domain with the inlet flow
simulation.set_uniform_flow(inlet_speed, direction=(1.0, 0.0))

# ─────────────────────────────────────────────
#  Pre-compute frames  (mirrors Simulation.py)
# ─────────────────────────────────────────────
obstacle = simulation.obstacle_mask
frames   = []

global_min_mach        =  np.inf
global_max_mach        = -np.inf
global_min_pressure    =  np.inf
global_max_pressure    = -np.inf
global_min_density     =  np.inf
global_max_density     = -np.inf
global_min_temperature =  np.inf
global_max_temperature = -np.inf

precompute_start = time.perf_counter()
update_progress_bar(0, num_frames, precompute_start)

for frame_index in range(num_frames):
    target_time = min((frame_index + 1) * frame_interval, sim_time)
    while simulation.simulation_time < target_time - 1e-12:
        simulation.step()

    # ── gather fields ──────────────────────────────
    speed      = np.hypot(simulation.u, simulation.v)
    mach_plot  = simulation.mach.copy()
    p_plot     = simulation.p.copy() - REF_PRESSURE   # gauge
    rho_plot   = simulation.rho.copy()
    temp_plot  = simulation.temperature.copy()

    # Mask walls
    mach_plot [obstacle] = np.nan
    p_plot    [obstacle] = np.nan
    rho_plot  [obstacle] = np.nan
    temp_plot [obstacle] = np.nan

    # Centreline Mach profile for the extra panel
    cx         = ROWS // 2
    cl_mach    = simulation.mach[cx, :].copy()

    # ── global range tracking ──────────────────────
    for arr, gmin_name, gmax_name in [
        (mach_plot,  "global_min_mach",        "global_max_mach"),
        (p_plot,     "global_min_pressure",     "global_max_pressure"),
        (rho_plot,   "global_min_density",      "global_max_density"),
        (temp_plot,  "global_min_temperature",  "global_max_temperature"),
    ]:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            globals()[gmin_name] = min(globals()[gmin_name], float(np.min(finite)))
            globals()[gmax_name] = max(globals()[gmax_name], float(np.max(finite)))

    frames.append({
        "frame_time"  : simulation.simulation_time,
        "mach_plot"   : mach_plot,
        "p_plot"      : p_plot,
        "rho_plot"    : rho_plot,
        "temp_plot"   : temp_plot,
        "cl_mach"     : cl_mach,
    })
    update_progress_bar(frame_index + 1, num_frames, precompute_start)

# Fallback safe ranges
def _safe_range(lo, hi, default_lo, default_hi):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return default_lo, default_hi
    return lo, hi

global_min_mach,        global_max_mach        = _safe_range(global_min_mach,        global_max_mach,        0.0,   2.5)
global_min_pressure,    global_max_pressure    = _safe_range(global_min_pressure,    global_max_pressure,   -5e4,  5e4)
global_min_density,     global_max_density     = _safe_range(global_min_density,     global_max_density,     0.5,  1.5)
global_min_temperature, global_max_temperature = _safe_range(global_min_temperature, global_max_temperature, 200.0, 350.0)

# ─────────────────────────────────────────────
#  Figure layout  (2 × 3, matches Simulation.py style)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("white")
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                       bottom=0.12, top=0.93, left=0.07, right=0.97)

ax_mach   = fig.add_subplot(gs[0, 0])   # Mach field
ax_press  = fig.add_subplot(gs[0, 1])   # Gauge pressure
ax_temp   = fig.add_subplot(gs[0, 2])   # Temperature
ax_density= fig.add_subplot(gs[1, 0])   # Density
ax_cl     = fig.add_subplot(gs[1, 1])   # Centreline Mach plot
ax_blank  = fig.add_subplot(gs[1, 2])   # reserved / info panel

colorbars = {}

EXTENT = (0, COLS - 1, 0, ROWS - 1)

def draw_frame(frame_index: int):
    frame      = frames[frame_index]
    frame_time = frame["frame_time"]
    mach_plot  = frame["mach_plot"]
    p_plot     = frame["p_plot"]
    rho_plot   = frame["rho_plot"]
    temp_plot  = frame["temp_plot"]
    cl_mach    = frame["cl_mach"]

    for ax in [ax_mach, ax_press, ax_temp, ax_density, ax_cl, ax_blank]:
        ax.clear()

    # ── helpers ─────────────────────────────────────
    def _imshow(ax, data, cmap, vmin, vmax, title, cbar_key, cbar_label):
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=EXTENT, alpha=0.82, interpolation="bilinear")
        ax.imshow(obstacle, origin="lower", cmap="gray_r",
                  vmin=0, vmax=1, extent=EXTENT, alpha=0.55)
        ax.set(title=title, aspect=1,
               xlim=(0, COLS - 1), ylim=(0, ROWS - 1))
        ax.set_xlabel("X (grid cells)", fontsize=8)
        ax.set_ylabel("Y (grid cells)", fontsize=8)
        if cbar_key not in colorbars:
            colorbars[cbar_key] = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            colorbars[cbar_key].set_label(cbar_label, fontsize=8)
        else:
            colorbars[cbar_key].update_normal(im)
        return im

    # ── Mach (plasma + M=1 sonic line) ──────────────
    im_m = _imshow(ax_mach, mach_plot, "plasma",
                   0.0, global_max_mach,
                   "Mach Number", "mach", "M")
    try:
        ax_mach.contour(mach_plot, levels=[1.0],
                        colors="white", linewidths=1.2,
                        extent=EXTENT, origin="lower")
        ax_mach.text(2, ROWS - 6, "white line = M=1 sonic line",
                     color="white", fontsize=6, va="top")
    except (ValueError, RuntimeError):
        pass

    # ── Gauge pressure ───────────────────────────────
    p_lim = max(abs(global_min_pressure), abs(global_max_pressure), 1.0)
    _imshow(ax_press, p_plot, "RdBu_r",
            -p_lim, p_lim,
            "Gauge Pressure", "press", "Pa")

    # ── Temperature ──────────────────────────────────
    _imshow(ax_temp, temp_plot, "inferno",
            global_min_temperature, global_max_temperature,
            "Temperature", "temp", "K")

    # ── Density ──────────────────────────────────────
    _imshow(ax_density, rho_plot, "cividis",
            global_min_density, global_max_density,
            "Density", "density", "kg/m³")

    # ── Centreline Mach plot ─────────────────────────
    x_axis = np.arange(COLS)
    ax_cl.plot(x_axis, cl_mach, color="#e6550d", linewidth=1.6)
    ax_cl.axhline(1.0, color="steelblue", linestyle="--",
                  linewidth=1.1, label="M = 1 (sonic)")
    ax_cl.axvline(COLS * 0.4, color="gray", linestyle=":",
                  linewidth=1.0, label="Throat")
    ax_cl.set_xlim(0, COLS - 1)
    ax_cl.set_ylim(0, max(global_max_mach * 1.05, 1.2))
    ax_cl.set_xlabel("X (grid cells)", fontsize=8)
    ax_cl.set_ylabel("Mach",           fontsize=8)
    ax_cl.set_title("Centreline Mach Profile")
    ax_cl.legend(fontsize=7)
    ax_cl.grid(True, alpha=0.3)

    # ── Info panel ───────────────────────────────────
    ax_blank.set_axis_off()
    throat_col  = int(COLS * 0.4)
    throat_mach = float(simulation.mach[ROWS // 2, throat_col])
    exit_mach   = float(np.nanmean(simulation.mach[
        ROWS // 2 - 3 : ROWS // 2 + 3, -5:]))
    info_text = (
        f"Simulation time : {frame_time:.2f} s\n"
        f"Inlet Mach      : {INLET_MACH:.2f}\n"
        f"Throat Mach     : {throat_mach:.3f}\n"
        f"Exit Mach (avg) : {exit_mach:.3f}\n"
        f"p0 / pb         : {INLET_TOTAL_PRESSURE / BACK_PRESSURE:.3f}\n"
        f"\nFlux scheme : {FLUX_SCHEME.upper()}\n"
        f"CFL         : {CFL_NUMBER}\n"
        f"Grid        : {ROWS} × {COLS}\n"
        f"γ           : {GAMMA}"
    )
    ax_blank.text(0.05, 0.95, info_text,
                  transform=ax_blank.transAxes,
                  fontsize=9, va="top", fontfamily="monospace",
                bbox={"boxstyle": "round", "facecolor": "lightyellow",
                    "edgecolor": "gray", "alpha": 0.8})
    ax_blank.set_title("Run Parameters", fontsize=10)

    fig.suptitle(
        f"De Laval Nozzle — 2D Compressible Flow  |  t = {frame_time:.2f} s",
        fontsize=13, fontweight="bold",
    )


draw_frame(0)

# ─────────────────────────────────────────────
#  Slider  (same pattern as Simulation.py)
# ─────────────────────────────────────────────
slider_ax  = fig.add_axes((0.20, 0.03, 0.65, 0.030))
time_slider = Slider(
    ax       = slider_ax,
    label    = f"Frame  (0 – {num_frames - 1})",
    valmin   = 0,
    valmax   = num_frames - 1,
    valinit  = 0,
    valstep  = 1,
)

def on_slider_change(value):
    draw_frame(int(value))
    fig.canvas.draw_idle()

time_slider.on_changed(on_slider_change)

# ─────────────────────────────────────────────
#  Video export  (same pattern as Simulation.py)
# ─────────────────────────────────────────────
def save_simulation_video():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(
        fig,
        lambda i: draw_frame(i) or [],
        frames   = num_frames,
        interval = 1000.0 / max(VIDEO_FPS, 1),
        blit     = False,
        repeat   = False,
    )
    mp4_path = output_dir / f"{VIDEO_BASENAME}.mp4"
    try:
        writer = FFMpegWriter(fps=VIDEO_FPS, codec="libx264", bitrate=4000,
                              extra_args=["-pix_fmt", "yuv420p"])
        anim.save(str(mp4_path), writer=writer, dpi=VIDEO_DPI)
        print(f"Saved: {mp4_path}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        gif_path = output_dir / f"{VIDEO_BASENAME}.gif"
        print(f"FFmpeg failed ({exc}), falling back to GIF: {gif_path}")
        anim.save(str(gif_path), writer=PillowWriter(fps=VIDEO_FPS), dpi=VIDEO_DPI)

if EXPORT_VIDEO:
    save_simulation_video()

if SHOW_PLOT:
    plt.show()
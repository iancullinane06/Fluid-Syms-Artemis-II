# Fluid-Syms-Artemis-II

2D rocket-frame CFD-style prototype for visualising flow, wake formation, drag trends, and (in compressible mode) Mach/thermodynamic structure around obstacles.

## What this project currently includes

This repository now contains three related simulations:

1. **Rocket flow demo (default: compressible)** in [Simulation.py](Simulation.py)
2. **Nozzle flow demo (compressible)** in [NozzleSim.py](NozzleSim.py)
3. **1D ascent/drag model** in [Drag.py](Drag.py)

The core solver lives in [Mechanisms/FluidSimulation.py](Mechanisms/FluidSimulation.py) and supports both:

- **Incompressible branch** (semi-Lagrangian Navier–Stokes style projection method)
- **Compressible branch** (finite-volume Euler update with Riemann fluxes: HLLC or Rusanov)

## Main files

- [Simulation.py](Simulation.py): Artemis-inspired rocket scenario, coupled to rocket dynamics and visualisation
- [NozzleSim.py](NozzleSim.py): converging-diverging nozzle demo with Mach/pressure/temperature/density panels
- [Drag.py](Drag.py): standalone ascent + drag analysis against atmosphere models
- [Mechanisms/FluidSimulation.py](Mechanisms/FluidSimulation.py): CFD core (incompressible + compressible stepping)
- [Mechanisms/RocketDynamics.py](Mechanisms/RocketDynamics.py): rocket state integration and drag decomposition
- [Mechanisms/Visualisation.py](Mechanisms/Visualisation.py): frame rendering + MP4/GIF export helpers
- [Mechanisms/Classes.py](Mechanisms/Classes.py): dataclasses and geometry helpers
- [Mechanisms/Functions.py](Mechanisms/Functions.py): Reynolds number and drag-coefficient utility functions

## Solver overview

### Rocket-frame relative freestream

Both modes update boundary freestream from ambient and rocket motion:

$$
\mathbf{u}_{\infty}(t)=\mathbf{u}_{\text{ambient}}(t)-\mathbf{u}_{\text{rocket}}(t)
$$

with

$$
U_{\infty}(t)=\left\|\mathbf{u}_{\infty}(t)\right\|,
\quad
\hat{\mathbf{d}}_{\infty}(t)=\frac{\mathbf{u}_{\infty}(t)}{\left\|\mathbf{u}_{\infty}(t)\right\|}
$$

### Incompressible branch (projection pipeline)

In incompressible mode, each step follows this sequence (see `step()`):

1. Apply body/reference-frame acceleration
2. Enforce directional inlet/outlet boundaries
3. Semi-Lagrangian advection via `scipy.ndimage.map_coordinates`
4. Iterative viscous diffusion
5. LES-style turbulence closure + optional vorticity confinement
6. Pressure Poisson solve and velocity projection to approximately divergence-free flow
7. Obstacle + domain boundary cleanup

Core relation for projection:

$$
\nabla^2 p = \nabla\cdot\mathbf{u}^{*},
\quad
\mathbf{u}^{n+1}=\mathbf{u}^{*}-\nabla p
$$

### Compressible branch (finite-volume Euler)

In compressible mode, each step uses:

1. CFL-limited local time step
2. Conservative Euler update (dimensional splitting)
3. Interface flux via HLLC (default) or Rusanov
4. Compressible obstacle state reconstruction
5. Directional inflow/outflow boundaries + optional downstream sponge
6. Primitive/conservative synchronization and thermodynamic updates

### Obstacle treatment

- Incompressible: immersed-boundary style no-slip reconstruction near the wall
- Compressible: obstacle cells are reset to quiescent thermodynamic states with wall velocity constrained to zero

## Boundary conditions

- Direction-aware inlet/outlet selection from freestream direction
- Inlet blended toward target freestream state
- Outlet uses near zero-gradient outflow behavior
- Cross-flow boundaries use slip-like handling for the incompressible branch

## Current demo defaults

### [Simulation.py](Simulation.py)

- `USE_COMPRESSIBLE = True`
- Grid: `140 x 460` (compressible) or `200 x 650` (if switched to incompressible)
- Compressible defaults: `COMPRESSIBLE_BASE_MACH = 1.45`, `COMPRESSIBLE_CFL_NUMBER = 0.70`, `COMPRESSIBLE_FLUX_SCHEME = "hllc"`
- Time setup: `time_step = 0.02`, `sim_time = 80.0`, `frame_interval = 0.5` in compressible mode

### [NozzleSim.py](NozzleSim.py)

- Compressible converging-diverging nozzle case
- Grid: `120 x 400`
- Defaults: `INLET_MACH = 0.2`, `CFL_NUMBER = 0.45`, `FLUX_SCHEME = "hllc"`

## Outputs

- Visual outputs are written to [outputs/](outputs/)
- MP4 export is attempted when FFmpeg is available
- GIF fallback/export is supported via Pillow writer

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib
- seaborn (used by [Drag.py](Drag.py))

Install dependencies:

```bash
pip install numpy scipy matplotlib seaborn
```

Optional for MP4 export:

- Install FFmpeg and ensure it is on PATH, or set `FFMPEG_PATH`

## Run

Rocket demo:

```bash
python Simulation.py
```

Nozzle demo:

```bash
python NozzleSim.py
```

1D ascent/drag comparison:

```bash
python Drag.py --mode compare --duration 300 --dt 1
```

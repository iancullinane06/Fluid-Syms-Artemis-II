# Fluid-Syms-Artemis-II

2D rocket-frame CFD-style prototype for visualising flow, wake formation, and drag trends around a rocket obstacle.

## What this sim does

The solver evolves a 2D velocity field around a rasterised rocket profile using a simplified incompressible Navier–Stokes pipeline:

1. Update rocket-frame freestream from ambient and rocket velocity profiles
2. Advect velocity (semi-Lagrangian, SciPy interpolation)
3. Diffuse velocity (iterative viscous solve)
4. Apply turbulence closure (Smagorinsky + vorticity confinement)
5. Project to incompressible flow (Poisson pressure solve)
6. Enforce obstacle no-slip + inlet/outlet boundaries

Main files:

- [FluidSimulation.py](FluidSimulation.py): solver core
- [Simulation.py](Simulation.py): demo setup, Artemis-like ascent profile, visualisation
- [Classes.py](Classes.py): rocket geometry/profile dataclasses
- [Functions.py](Functions.py): Reynolds number and drag-coefficient helpers
- [Drag.py](Drag.py): separate 1D ascent/drag model (not tightly coupled to 2D solver)

## Core math (matching the code)

### 1) Rocket-frame relative freestream

The boundary freestream is updated from:

$$
\mathbf{u}_{\infty}(t) = \mathbf{u}_{\text{ambient}}(t) - \mathbf{u}_{\text{rocket}}(t)
$$

with speed and direction:

$$
U_{\infty}(t) = \left\|\mathbf{u}_{\infty}(t)\right\|,
\quad
\hat{\mathbf{d}}_{\infty}(t) = \frac{\mathbf{u}_{\infty}(t)}{\left\|\mathbf{u}_{\infty}(t)\right\|}
$$

### 2) Semi-Lagrangian advection

For each cell center $\mathbf{x}_{ij}$, backtrace to departure point:

$$
\mathbf{x}_d = \mathbf{x}_{ij} - \mathbf{u}(\mathbf{x}_{ij}, t)\,\Delta t
$$

Then sample with bilinear interpolation (implemented with `scipy.ndimage.map_coordinates`):

$$
\mathbf{u}^{*}(\mathbf{x}_{ij}) = \mathcal{I}\left[\mathbf{u}^{n}\right](\mathbf{x}_d)
$$

### 3) Viscous diffusion

The code applies iterative smoothing equivalent to:

$$
\frac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u}
$$

in discrete form over interior cells.

### 4) Immersed-boundary no-slip wall damping

Near the obstacle, velocity is decomposed into normal and tangential components:

$$
u_n = \mathbf{u}\cdot\hat{\mathbf{n}},
\quad
u_t = \mathbf{u}\cdot\hat{\mathbf{t}}
$$

and damped toward no-slip with stronger tangential damping in the wall band:

$$
u_n \leftarrow (1-\alpha_n)u_n,
\quad
u_t \leftarrow (1-\alpha_t)u_t
$$

where $\alpha_n, \alpha_t$ depend on wall distance, timestep, and a Reynolds-based drag coefficient estimate.

### 5) LES turbulence model + vorticity confinement

Smagorinsky-style eddy viscosity:

$$
\nu_t = C_s^2\,\|S\|,
\quad
\nu_{\text{eff}} = \nu + \nu_t
$$

and optional vorticity confinement force:

$$
\mathbf{f}_{vc} = \epsilon\,(\hat{\mathbf{n}}_\omega \times \omega)
$$

(2D scalar-vorticity form in code).

### 6) Pressure projection (incompressibility)

Solve Poisson equation:

$$
\nabla^2 p = \nabla\cdot\mathbf{u}^{*}
$$

using Gauss-Seidel, then project:

$$
\mathbf{u}^{n+1} = \mathbf{u}^{*} - \nabla p
$$

This enforces approximately divergence-free flow:

$$
\nabla\cdot\mathbf{u}^{n+1} \approx 0
$$

### 7) Drag proxy used in visualisation

The plotted drag trend is a proxy combining near-wall dynamic-pressure-like and shear-like terms:

$$
D_{\text{proxy}} \sim \tfrac{1}{2}\rho\langle u^2+v^2\rangle_{\text{wall band}} + \rho\nu\langle |u_t|\rangle_{\text{wall band}}
$$

It is useful for trends, not a full force integration.

## Boundary conditions in this implementation

- Direction-aware inlet/outlet selection from freestream direction
- Outlet uses zero-gradient outflow (free escape)
- Cross-flow edges use slip-like conditions
- Obstacle interior is solid; wall layers damp to no-slip

## Current demo configuration

In [Simulation.py](Simulation.py):

- Grid: `300 x 600`
- Rocket profile scaled to keep proportions at high resolution
- Artemis-like ascent speed envelope (scaled to stable solver units)
- Warm-up steps before frame capture to reduce startup transient artifacts

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python Simulation.py
```

Optional 1D ascent/drag comparison:

```bash
python Drag.py --mode compare --duration 300 --dt 1
```

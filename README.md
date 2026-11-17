# PSE for SU2-NEMO AIR-7 boundary layers

Local stability theory (LST), linear PSE (LPSE) and nonlinear PSE (NPSE) for
hypersonic boundary layers computed with **SU2-NEMO**
(`SOLVER= NEMO_NAVIER_STOKES`, two-temperature TTv model, `AIR-7`), using
either SU2's built-in thermochemistry (`FLUID_MODEL= SU2_NONEQ`) or
**Mutation++** (`FLUID_MODEL= MUTATIONPP`, `GAS_MODEL= air_7`).

The stability machinery follows **Krypton** (Lacombe, SoftwareX
SOFTX-D-22-00069): LST start, BDF streamwise marching with the
normalisation condition, mixed `d2/dxdy` terms kept, Andersson-type
stabilisation `M -> M + s L`, and NPSE by harmonic balance with mean-flow
distortion. Krypton's perfect-gas `(u, v, w, p, T)` equations are replaced
by the full 12-variable AIR-7 TTv system.

```
pip install -r requirements.txt
python3 -m pytest tests -q                                   # ~1 min
python3 scripts/run_pse.py config/mach10_130kft.json         # similarity base flow
python3 scripts/run_pse.py config/su2_nemo_air7.json         # SU2-NEMO base flow
python3 scripts/run_pse.py config/npse_mach10.json           # NPSE
mpirun -n 384 python3 scripts/run_pse.py config/su2_nemo_air7.json
```

## State and equations

```
q = [rho_e-, rho_N2, rho_O2, rho_NO, rho_N, rho_O, rho_NO+, u, v, w, T, Tve]
```

Species order is SU2's AIR-7 order (= `GAS_COMPOSITION` order). The shipped
Mutation++ mixture `mutationpp_data/mixtures/air_7.xml` lists the same order,
so one `GAS_COMPOSITION` serves both libraries.

`R = dU/dt + d(lF)/ds + dG/dn + dH/dz + E = 0` in body-fitted orthogonal
coordinates (`s` along the wall, `n` normal, `z` spanwise or `r*phi`), with
`l = 1/(1 + kappa n)`. `E` holds the axisymmetric divergence terms, hoop
stresses, curvature terms and minus the sources. Viscous fluxes are SU2's
(`CNEMONumerics::GetViscousProjFlux`):
`J_s = -(rho D_s grad Y_s - Y_s sum rho D_k grad Y_k)`, heat flux
`k_tr grad T + k_ve grad Tve + sum J_s h_s`, ve flux
`k_ve grad Tve + sum J_s e_ve,s`. All Jacobians are exact complex-step
derivatives of these functions (`jacobians.py`).

## Thermochemistry

### `gas.library = "su2tclib"` (SU2_NONEQ, AIR-7)

Reproduced from `CSU2TCLib.cpp`; the data tables were generated from the C++
source, not retyped:

* energies, Cv, enthalpies: rigid rotor, harmonic oscillator, SU2's
  electronic level sets, Scalabrin formation enthalpies, electron energy in
  the ve mode;
* 22 reactions, Park rates, SU2's smoothed rate-controlling temperatures
  (`T_min = 800 K`, `eps = 80 K`), Park `Keq` curve fits interpolated in
  number density with SU2's exact table-row logic;
* Landau-Teller with Millikan-White (harmonic mixture rule) + Park
  `3e-21 (5e4/T)^2` cross section; non-preferential C-V coupling;
* `TRANSPORT_COEFF_MODEL`: WILKE (Blottner/Eucken/Wilke + collision-integral
  diffusion), GUPTA-YOS, SUTHERLAND.

Reproduced as SU2 has them:

* NO+ carries the NO molar mass: `N + O -> NO+ + e-` creates one electron
  mass per mole.
* **GUPTA-YOS conductivity** (not reproduced by default): SU2 writes the partner concentration as
  `rho_i/(rho M_j)`. With e- present this makes the conductivity 6-10x too
  small (Pr 4-7 instead of 0.7). The default uses the textbook rule; set
  `gas.gy_partner_bug = true` to reproduce SU2 exactly. If you ran SU2 with
  GUPTA-YOS, check your wall heat flux.

Deliberate deviations (scale with Y_NO+ or Y_e-, < 1e-10 here): NO+ gets its
trans-rotational Cv (SU2 leaves it 0), and every species diffuses (SU2's
loop skips NO+ and includes e-).

### `gas.library = "mutationpp"` (MUTATIONPP, air_7)

Mutation++ ships no `air_7`. `mutationpp_data/` provides `air_7.xml` and
`air7_SU2.xml` (SU2's AIR-7 rates as 7 Mutation++ reactions). Copy them into
your `$MPP_DATA_DIRECTORY` before running SU2 with `GAS_MODEL= air_7`.

Re-implemented from Mutation++ (`RrhoDB`, `RateManager`, `OmegaVT/CV/CElec/CE`):
RRHO thermo with 298.15 K formation enthalpies, Gibbs-energy backward rates,
Mutation++ rate temperatures (`sqrt(T Tv)` for heavy dissociation),
Millikan-White with `VT.xml` constants. Data are read from
`$MPP_DATA_DIRECTORY` when set, otherwise the embedded copy.
Transport comes from the Mutation++ Python module (local first-order
expansion per station, exact for LST/LPSE); without the module the
SU2TCLib formulas are used and a warning is printed. `OmegaET`/`OmegaI`
are omitted (need e-/heavy collision integrals; zero here).

## SU2-NEMO base flow

`baseflow.source = "su2"`:

* reads the native `.su2` mesh (wall marker) and restart/volume files
  (CSV or SU2 binary); `solution` may be a glob of ILES snapshots;
* species from `Density_k`, velocity from `Velocity_*` or `Momentum_*`,
  temperatures from `Temperature_tr/_ve` or, if absent, inverted from
  `Energy/Energy_ve` with the gas model;
* snapshot averaging: `average = "primitive"` (Reynolds) or
  `"conservative"` (then T, Tve inverted);
* 3-D data are projected to the meridional (`axisym`) or x-y plane and
  coincident points merged;
* wall curve from the marker, smoothed, normals into the fluid, curvature;
  profiles by Delaunay-linear interpolation along the normals; velocities
  rotated to (tangential, normal);
* stations are wall arclength + `s0` (or axial x with `station_axis = "x"`).

`examples/su2_nemo_air7_iles.cfg` lists the matching SU2 options.

**ILES note.** With `KIND_TURB_MODEL= NONE` SU2 solves the laminar NEMO
equations; the PSE uses the same molecular transport. The scheme's numerical
dissipation is not in the PSE. Average enough snapshots that the mean is
converged upstream of transition.

## Stability formulation

```
(L0 + L1 D + L2 D^2) q + (M0 + M1 D) dq/dx = -NLT
```

`M1 D` are Krypton's mixed-derivative terms. Non-parallel terms use central
x-differences of the Jacobians. BDF1/BDF2 on the actual station spacing.

**Boundary conditions.** 23 conditions: 11 at the wall, 12 at the far field.

* wall: no-slip, `T' = Tve' = 0` (or adiabatic), and zero perturbation
  diffusive flux (non-catalytic) or `Y'_s = 0` (super-catalytic), linearised
  from the base-flow flux itself. The most abundant species' row keeps
  mixture continuity.
* far field: asymptotic (Riccati) decay condition from a row/column-balanced
  far-field pencil; Dirichlet and Neumann (MFD mode) available.
* **trace-species locking**: species whose mass fraction never exceeds
  `physics.trace_threshold` (1e-6) get `Y'_s = 0`. Left free, trace e-/ions
  produce spurious passive-scalar and electron-pressure eigenmodes that are
  grid dependent and pass the "trapped, wall-peaked" filter.

**Mode identification.** Shift-invert Arnoldi on the alpha-quadratic pencil,
filter on far-field energy and wall pressure, pick the most amplified trapped
mode, polish by bordered Newton.

**Stabilisation.** `pse.stabilize = k` sets `s = k max(0.5/|alpha_r| - dx, 0)`
(Krypton uses k = 1). Off by default; needed when `dx < 1/|alpha_r|`.

**NPSE.** Modes `(m, n)`, `0 <= m <= n_harmonics`, `|n| <= n_spanwise`.
Nonlinear terms are evaluated in flux form on a zero-padded (t, z) grid:
`N_X = FFT[X(qbar + q') - X(qbar)] - linear part` for `X = U, lF, G, H, E`,
so the full TTv physics enters. Fundamentals keep the normalisation;
harmonics are phase locked. `(0,0)` is the mean-flow distortion (streamwise
pressure gradient dropped, Neumann far field).

## Results at the shipped condition

Mach 10, 130 kft, 7 deg cone, `Tw = 1000 K`, Re/m = 1.4e6. The trapped second
mode is amplified only in a narrow band, ~0.5 u_e/(2 delta99) (165 kHz at
x = 0.6-0.8 m), and N stays O(1) to x = 0.9 m. At this low unit Reynolds
number the strongly amplified inviscid root is the *supersonic* (radiating,
c < 1 - 1/M_e) mode, which the trapped-mode filter excludes. Use
`analysis.type = "lst"` and widen `modes.f_*` to map the band for your case.

## Parallel and GPU

* **Mode level**: dynamic MPI-3 RMA work counter, band-centre-first order.
  Compute loops poke MPI progress (`pse/progress.py`) so the counter on
  rank 0 is served while rank 0 computes; without this, ranks stalled in
  `Fetch_and_op` (busy fraction 0.42 at 4 ranks).
* **Shared base-flow Jacobians** (`parallel.share_jacobians`, default on):
  every station's Jacobians are computed once per node, split over the
  node's ranks, into one MPI-3 shared-memory block (~0.4 GB for 121 x 161),
  instead of once per mode.
* **Wall-normal level**: `parallel.group_size > 1`, exact Schur-complement
  solver (CPU).

Measured on a 4-core box, 16 modes, 13 stations, ny = 121:

| ranks | modes/s | per rank | parallel eff. |
|---|---|---|---|
| 1 | 0.281 | 0.281 | 1.00 |
| 2 | 0.563 | 0.281 | 1.00 |
| 4 | 1.070 | 0.268 | 0.95 |

Before these changes: 0.659 modes/s at 4 ranks (8 modes), 0.453 without
shared Jacobians. Keep >= 3 modes per rank to limit the scheduling tail.

**GPU (A100 SXM)**: `parallel.backend = "gpu"` (or `PSE_BACKEND=gpu`) moves
every LU (march solves, Newton, Arnoldi shift-invert) to the device with
CuPy: the sparse system goes over as CSR and is densified and factored on
the GPU (complex128 LU of n ~ 2000 in about a millisecond on FP64 tensor
cores). Rank r uses GPU `local_rank % n_gpus`; run several ranks per GPU
under MPS (`scripts/submit_a100.slurm`, 8 ranks per A100). Needs
`pip install cupy-cuda12x`. Use `group_size = 1` with the GPU back end.
The device code path is tested with NumPy/SciPy standing in for CuPy; it
has not been run on an A100 here (no GPU in this environment). Jacobian
evaluation stays on the CPU (it is now done once per node).

**Lockstep batched march** (`parallel.lockstep = true`, `parallel.batch`):
a batch of modes is marched station by station together. Each mode's
operator is assembled directly as node-offset blocks from the shared
Jacobians; grouping 4 nodes into 48 x 48 super-blocks makes the system
block tridiagonal, solved by batched banded LU with partial pivoting
(`pse/batched.py`): a CUDA kernel factors each 96 x 48 pivoting panel
(one thread block per mode) and the trailing updates are batched FP64
GEMMs (cuBLAS, A100 tensor cores). Work per solve drops from the dense
O((ny NV)^3) to O(nb S^3), about 300x less at ny = 161. The same code runs
with NumPy on the CPU, which is how it is verified: alpha agrees with the
per-mode march to 1e-11 and the N envelope is identical. The CUDA panel
kernel checks itself against the array-op version on first use and falls
back if they disagree. It has not been compiled or run on a GPU here.
On a CPU the per-mode path with SuperLU is faster; use lockstep with
`backend = "gpu"`, batch 32-128 per GPU. The LST start of each mode
(Arnoldi) and the 23 x 23 far-field eigenproblems stay on the host.

`scripts/submit_384.slurm` (CPU), `scripts/submit_a100.slurm` (GPU),
`scripts/scaling_benchmark.py --backend cpu|gpu [--no-share]`,
`mpiexec -n 4 python3 tests/mpi_check.py` (Schur solver, scheduler,
shared Jacobians).

## Outputs

* LST/LPSE: `pse_modes.h5` (or `.npz`), `n_factor_envelope.csv`
* NPSE: `npse.npz` with `alpha_m_n`, `u_max_m_n`, `T_max_m_n`, `p_wall_m_n`,
  `N_m_n` per mode
* `baseflow.npz`

## Verification (`pytest tests`)

* atmosphere, oblique shock / Taylor-Maccoll
* gas: energy inversion, Cv vs complex-step dE/dT, Prandtl numbers, Keq
  table rows, reaction mass/charge balance, VT sign, GY quirk, Mutation++
  formation enthalpies, reaction typing, forward-rate agreement with SU2
* Jacobians: characteristic speeds `u, u +- a`, complex step vs finite
  differences; uniform stream preserved by the axisymmetric and curvature
  metrics (residual zero to round-off)
* LST/PSE: 23 far-field exponents / 12 decaying, trace locking, trapped
  amplified second mode, grid convergence < 1 %, PSE first station = LST
* NPSE: FFT round trip, small amplitude = LPSE, harmonics and MFD quadratic
* SU2: mesh and CSV/binary readers, base flow reproduced from a synthetic
  SU2 case (both with temperatures and via energy inversion), snapshot
  averaging
* Mutation++ transport bridge against a stand-in module; driver end to end

## Known limitations

* Wall curvature enters through the Lame factor; the base-flow x-derivatives
  of `kappa` beyond `dl/ds` are neglected.
* NPSE harmonics are phase locked to the fundamental; no receptivity model.
* NPSE nonlinear transport from the Mutation++ module is second-order
  accurate in amplitude (first-order property expansion).
* Single-zone `.su2` meshes only; CGNS not read.

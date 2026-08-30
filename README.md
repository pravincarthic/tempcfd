# PSE solver for second Mack-mode instabilities in 11-species air

Linear parabolised stability equations (PSE) for hypersonic boundary layers in
chemically reacting, thermally non-equilibrium air, driven by base flows from
**OpenFOAM v1706 + hy2Foam / hypersonicfoam** (`hyStrath`).  Designed for a
Mach 10, 130 000 ft (US 1976 Standard Atmosphere) flight condition and for
strong scaling on ~384 CPU cores.

```
pip install -r requirements.txt
python3 -m pytest tests -q                     # verification suite
python3 scripts/run_pse.py config/mach10_130kft.json -o results/run1
mpirun -n 384 python3 scripts/run_pse.py config/mach10_130kft.json
```

## Physical model

State vector, 16 unknowns per wall-normal node:

```
q = [ rho_N2 rho_O2 rho_NO rho_N rho_O rho_N2+ rho_O2+ rho_NO+ rho_N+ rho_O+ rho_e-,
      u, v, w, T, Tv ]
```

* **Thermodynamics** (`thermo.py`) — Park two-temperature model: translational
  and rotational modes at `T`, vibrational + electronic + free-electron
  translational modes at `Tv = Te`.  Harmonic-oscillator vibration and
  tabulated electronic level sets; heats of formation at 0 K.  Electron
  pressure evaluated at `Tv`.
* **Kinetics** (`chemistry.py`) — Park (1993) 11-species air: 3 dissociation
  reactions over all 11 third bodies (31 steps), 2 Zeldovich exchange,
  3 associative-ionisation, 11 charge-exchange and 2 electron-impact
  ionisation reactions — **49 elementary reactions**.  Forward rates are
  Arrhenius in Park's rate-controlling temperature (`T^0.7 Tv^0.3` for
  heavy-impact dissociation, `Tv` for electron-driven steps, `T` otherwise);
  backward rates come from equilibrium constants built from the *same*
  partition functions as the thermodynamics, so the model is thermodynamically
  consistent and exactly mass- and charge-conserving.
  Vibrational relaxation is Millikan–White with Park's high-temperature
  limiting cross-section, plus non-preferential chemistry–vibration coupling
  and the electron-impact ionisation energy sink.
* **Transport** (`transport.py`) — Blottner species viscosities, Eucken
  conductivities split into translational-rotational and vibrational-electronic
  parts, Wilke mixing; mass diffusion through a constant-Lewis (or constant
  Schmidt) closure with the mass-conservation correction.  Below ~800 K the
  viscosity blends smoothly to Sutherland air, where the Blottner fits are
  ~10 % high.

## Stability formulation

The residual `R = dU/dt + dF/dx + dG/dy + dH/dz - S` is linearised about the
base flow with

```
q'(x,y,z,t) = qhat(x,y) exp( i ∫ alpha dx + i beta z - i omega t )
```

and parabolised (second streamwise derivatives of the amplitude dropped):

```
( L0 + L1 d/dy + L2 d²/dy² ) qhat + M0 dqhat/dx = 0
```

Every coefficient matrix is obtained from **exact complex-step Jacobians** of
the flux and source functions (`jacobians.py`).  The physics is written once,
in `fluxes.py`; nothing is hand-differentiated, and there is no subtractive
cancellation even though the species densities span twenty decades.
Base-flow `x`-derivatives (non-parallel terms) and the axisymmetric divergence
metric `(r_x/r) F + (r_y/r) G` are included.

**Boundary conditions.**  The 11 species-continuity equations sum exactly to
the mixture-continuity equation, which carries no diffusion and is therefore
first order in `y`.  The system is 15 second-order + 1 first-order equations
and admits **31** conditions: 15 at the wall (10 independent non-catalytic
zero-diffusive-flux — or super-catalytic — species conditions, no-slip,
`T` and `Tv`, with the mixture-continuity equation retained as the 16th wall
row) and **16 at the far field**.

The far-field rows are the *asymptotic decay* (Riccati) condition, not
Dirichlet.  The wall-normal characteristic exponents of the frozen far-field
operator are computed from the 32×32 quadratic eigenproblem
`(L0 + λ L1 + λ² L2) v = 0`; exactly 16 of the 31 finite exponents decay, and
requiring the solution at `y_max` to contain no growing characteristic gives
`dqhat/dy = V Λ V⁻¹ qhat`.  This removes the discretised continuous spectra,
which otherwise swamp the trapped acoustic mode — for this case the slow
acoustic branch sits at `c = 1 - 1/M_e = 0.877`, only 2 % away from the second
mode, and a Dirichlet far field makes Newton fall into it every time.  The
Riccati form is used (rather than an SVD basis of the complementary subspace)
because it varies smoothly with `alpha`, which the finite-difference
`dA/dalpha` in the Newton solve requires.

**Mode identification.**  Shift-invert Arnoldi on the `alpha`-quadratic pencil
with the far-field rows frozen at the shift, filtered on (i) far-field energy
fraction, (ii) wall pressure amplitude — the second mode is a trapped acoustic
wave with `|p̂|` maximum at the wall — and (iii) the interior `|p̂|` minimum
count, which is Mack's mode index.  The candidate is then polished by a
bordered Newton iteration with the fully `alpha`-dependent far-field
condition.  Downstream stations continue from the previous `alpha` and
eigenfunction, and the structure metrics are re-checked at every station so a
branch jump is reported rather than silently integrated.

**Marching.**  First- or second-order backward differences in `x`, with the
wavenumber updated after each solve from `⟨qhat, dqhat/dx⟩ = 0` (weighted by
reference values of the primitive variables) until `alpha` converges.  The
usual `Δx > 1/|alpha_r|` step-size restriction is checked and reported; an
optional Li–Malik streamwise-pressure damping is available.  N-factors are
reported both as `∫ -alpha_i dx` and including the amplitude-function growth,
`N = ∫ ( -alpha_i + d ln‖qhat‖_E/dx ) dx`.

**Discretisation.**  Wall-normal fourth-order finite differences on an
algebraically stretched grid (Fornberg weights, exact on the non-uniform
mesh), or Chebyshev collocation with the same mapping.  Base-flow profiles are
mapped with quintic splines: linear interpolation puts a train of delta
functions in `d²q/dy²` and produces spurious growing wall modes — this was the
single largest source of non-physical eigenvalues during development.

## Base flow

Two sources, selected by `baseflow.source`:

**`openfoam`** — reads `constant/polyMesh` (points/faces/owner/neighbour/
boundary, ASCII or binary points and labels) and computes cell centres and
volumes with OpenFOAM's own face-decomposition/pyramid algorithm, so the
sampled data sit exactly where the solver stored them.  Fields follow the
hy2Foam naming used by `hyStrath`:

| quantity | field |
|---|---|
| species mass fractions | `N2 O2 NO N O N2+ O2+ NO+ N+ O+ e-` |
| trans-rotational temperature | `Tt` |
| vibrational-electronic temperature | `Tv` |
| velocity, pressure, density | `U`, `p`, `rho` (optional) |

Alternative names go through `baseflow.field_map`.  Wall-normal rays are cast
from the wall-patch face centres along the patch normals and sampled by
Delaunay-linear interpolation with a KD-tree fallback; velocities are rotated
into the local (tangential, normal) frame; `plane: axisym` projects onto
(axial, radial) and activates the axisymmetric metric.
Binary `polyMesh/faces` is not supported — run `foamFormatConvert -ascii`.

**`similarity`** — a self-contained Levy–Lees compressible similarity solution
using the same 11-species thermodynamics and transport (frozen composition),
with the Mangler transformation for sharp cones.  Cone/wedge edge conditions
come from a Taylor–Maccoll or oblique-shock solve.  This is the verification
path: it needs no CFD input, and it is what the test suite runs on.

## Parallel design

Two levels, both in `parallel.py`.

1. **Mode level (primary).**  Each `(frequency, spanwise wavenumber)` march is
   independent.  Work is handed out through an MPI-3 RMA `Fetch_and_op`
   counter — lock-free, no rank sacrificed as a master, and robust to the
   3–5× spread in per-mode cost that a static decomposition cannot absorb.
   Modes are offered band-centre-first (`modes.schedule: center_out`), a
   longest-processing-time-first heuristic that cuts the tail.
2. **Wall-normal level.**  `parallel.group_size > 1` splits `COMM_WORLD` into
   groups.  Inside a group the wall-normal points are partitioned; the
   complex-step Jacobian assembly (the dominant per-station cost) is computed
   slab-wise and gathered, and every linear solve is done by an **exact
   Schur-complement substructuring solver** (`linalg.SchurSolver`): interiors
   are eliminated with a local sparse LU, the small interface system —
   `4·NV·(g−1)` unknowns — is assembled by one `Allreduce` and solved
   redundantly.  Verified to machine precision against the serial solve
   (`tests/mpi_check.py`, relative error 8e-16), and a 4-rank group reproduces
   the serial N-factor to six figures.

**Sizing for 384 cores.**  Aim for at least three work units per group.  The
shipped config uses 384 frequencies with `group_size: 4` → 96 groups × 4 modes
each, which keeps granularity healthy *and* speeds up each individual march.
If you sweep ≥ 3×384 modes (e.g. 384 frequencies × 4 spanwise wavenumbers),
`group_size: 1` is the better choice.  Always pin one BLAS thread per rank
(`PSE_THREADS=1`); nested threading only oversubscribes.

`scripts/submit_384.slurm` is a ready 8×48-rank job script.
`scripts/scaling_benchmark.py` reports wall time, per-group load balance,
scheduler efficiency and throughput.

## Configuration

Configs are plain JSON — `config/mach10_130kft.json` is the shipped example;
`pse/config.py` holds the defaults, which a config file only needs to
override in part (missing sections/keys fall back to the defaults).  Override
from the command line with `-o/--output`, `--foam-case`, `--nf`,
`--group-size`, or dump the resolved configuration with `--dump-config`.

Outputs land in `output.directory`:
`pse_modes.h5` (per-mode `x`, `alpha`, growth rate, N-factors, phase speed;
`.npz` if h5py is absent), `n_factor_envelope.csv`, and `baseflow.npz`.

## Flight condition

US 1976 at 130 000 ft (39 624 m geometric):
`T∞ = 249.3 K`, `p∞ = 302.1 Pa`, `ρ∞ = 4.222e-3 kg/m³`, `a∞ = 316.5 m/s`;
at Mach 10, `U∞ = 3165 m/s` and `Re/m = 8.4e5`.
Behind a 7° sharp cone shock (Taylor–Maccoll, `β = 9.45°`):
`M_e = 8.14`, `T_e = 367 K`, `p_e = 1028 Pa`, `u_e = 3134 m/s`.

At these conditions the boundary layer peaks near 1.5 kK with a 1000 K wall,
so the flow is chemically frozen — `physics.reacting` defaults to `false`
while thermal non-equilibrium stays on.  Set `reacting: true` for hotter
trajectories (blunt noses, lower altitude, higher Mach); the kinetics module
is fully coupled into the operator through the source Jacobian either way.

## Verification

`python3 -m pytest tests -q` (≈2 min) covers:

* US 1976 at sea level, 11/20/32 km and the 130 kft flight point;
* oblique-shock and Taylor–Maccoll against standard tables;
* thermodynamic round-trips, the classical high-`T` vibrational limit,
  mixture Prandtl number and the Sutherland blend;
* exact element and charge balance of all 49 reactions, mass-conserving source
  terms and the correct equilibrium ordering (O₂ dissociates before N₂,
  ionisation increases with `T`);
* grid convergence and integration weights for both discretisations;
* **inviscid Jacobians reproducing the characteristic speeds `u`, `u ± a`**
  with the frozen speed of sound to 1e-6 — this validates the thermodynamics,
  the flux definitions and the complex-step differentiation together;
* complex-step vs central differences on the reacting, two-temperature fluxes;
* the far-field split (31 finite exponents, 16 decaying);
* the second mode being trapped (`far-field energy < 1e-4`), wall-peaked in
  pressure, with `0.7 < c/u_e < 1.05`, and `alpha` grid-converged to 2 %
  between 81 and 121 wall-normal points;
* a full PSE march producing a physical amplified band, with the first station
  matching the local LST eigenvalue;
* the OpenFOAM reader end to end against a synthetic hy2Foam-format case whose
  mesh volume it reproduces to 1e-10 (`tests/foam_fixture.py`).

`mpirun -n 4 python3 tests/mpi_check.py` checks the distributed solver and the
scheduler.

## Known limitations

* Ionic electronic level sets are truncated to the levels Park tabulates; the
  ionised fraction is < 1e-7 at this condition.
* Ambipolar diffusion is modelled through the same Lewis/Schmidt closure as the
  neutrals rather than a separate ambipolar field.
* Transverse curvature is carried only through the axisymmetric divergence
  metric; the `1 + κy` Lamé factor is not included (thin-layer assumption).
* Binary `polyMesh/faces` is unsupported (see above).
* Linear PSE only: no nonlinear harmonic balance, no receptivity model.  The
  N-factor is an amplification ratio, not an amplitude.

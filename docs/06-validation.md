# Validation record

The evidence base for the reducer: what has been cross-checked, against which
independent engines, and where every number comes from. The reducer takes any
one-loop integral with an arbitrary polynomial numerator and reduces it to the
four scalar masters A0/B0/C0/D0 with coefficients rational in `d = 4 − 2ε` (see
[the overview](01-overview.md) and [the reduction algorithm](02-reduction.md)).
Everything below is a check on those coefficients and their assembled values —
never on hand-tuned numbers.

Two complementary tracks are reported. **(1)** A cross-engine harness that checks
generic families against independent scalar-master libraries, an analytic
library, a Feynman-parameter integrator, and two tensor oracles — this is the
*direct* validation of the reducer's output. **(2)** A MadLoop / MG5_aMC suite of
~110 real collider processes, which is a *reproduction record* of the process
space the reductions cover, not a per-process diff.

To run any of it, see
[`../crates/one-loop-reduce/benchmarks/README.md`](../crates/one-loop-reduce/benchmarks/README.md).

## Results at a glance

| Surface | Scope | Result |
|---|---|---|
| **Cross-engine families** | 132 integral families, 2 geometries, vs OneLOop + feynalg + scipy | **132 / 132 PASS** |
| **Master values** | every A0/B0/C0/D0 emitted, OneLOop vs feynalg | **944 / 944 agree** |
| High-rank mixed tensors | box → heptagon, distinct external directions, rank 3–6 (incl. dotted) | all PASS, pulls **0.0–1.6σ** |
| Singular-Gram tensors | heptagon (N=7) + coincident momenta, pseudo-inverse fix | reduces + validated, **0.4–0.7σ** |
| MadLoop reproduction | ~110 MG5_aMC v3.7.2 processes (96 fresh, 3 batches) | reproduced to MadLoop's ~14-digit accuracy |
| Full-amplitude assembly | gg→h, H→γγ (W loop) | [09](09-ggh-formfactor.md) (10⁻¹³), [10](10-hgammagamma.md) (~10⁻⁵) |
| Unit tests | `cargo test --workspace -- --test-threads=1` | 70 library + 7 binding pass, 1 ignored, 0 fail |

---

## 1. The cross-engine harness

`benchmarks/python/crosscheck.py` reads the reductions emitted by
`benchmarks/rust/emit_reductions.rs` and, for each family, forms the full Laurent
series

```
V_reduced(ε) = Σ_i c_i(d = 4 − 2ε) · M_i(ε)
```

where the `c_i` are the reducer's rational-in-`d` coefficients (series-expanded
in ε) and each `M_i` is a scalar master value. It then requires **(a)** the `1/ε`
and `1/ε²` poles cancel where the original integral is finite, and **(b)** the
assembled finite part matches an independent direct evaluation. The
renormalization scale is fixed to `μ² = 1` (`ob.set_renormalization_scale(1.0)`)
to match the bare scipy Feynman integrand.

| engine | role | what it validates |
|---|---|---|
| **OneLOop** (`avh_olo`, van Hameren) via `oneloop_bridge` | numeric scalar masters A0/B0/C0/D0, with poles | the master **values** `M_i` |
| **feynalg** `analytic_A0/B0/C0/D0` | analytic scalar masters | independent second check of master finite parts |
| **scipy** Feynman-parameter integration (`scipy_quad` deterministic quadrature N≤4; `scipy_direct` Monte-Carlo N≥5) | direct value of the *original* integral in `d = 4` | the **coefficients + topology selection**, end to end |
| **tensor oracles** (`_tensor_oracle.py`: symmetric-moment + covariant Minkowski-Gram) | direct value of tensor-numerator integrals | dotted / rank-raised numerators, under a **two-oracle** agreement requirement |

**Pass criteria.** Poles cancel to `< 1e-6` (algebraically they come out
`~1e-17`); the reduced finite part sits within 5σ of the scipy Monte-Carlo (3M
samples), or within `2e-3` relative for the deterministic N≤4 quadrature. A
separate line tallies OneLOop-vs-feynalg master agreement.

### Coverage

| Category | What it covers |
|---|---|
| Scalar N = 3…7 | triangle → heptagon; vNV / degenerate-Cayley for N>4. N≤4 exact vs OneLOop masters, N≥5 finite and checked against scipy MC |
| Dotted (raised powers) | FJT/Tarasov index-lowering: `[2,1,1,1]`, `[2,2,1]`, `[3,1]` |
| Isotropic tensor `(k²)ᵖ` | rank-2, rank-4, rank-6 `(k²)³`, finite **and** UV-divergent (compared on the full Laurent series — both the `1/ε` coefficient and the finite part) |
| Rank-1 external `k·qᵢ` | bubble … heptagon |
| Rank-2 mixed | `k·qᵢ k·qⱼ`, `k² k·qᵢ`, `(k·qᵢ)²` |
| **Rank-3–6 mixed** | distinct external directions, box → heptagon, incl. dotted — see §2 |
| **Singular-Gram tensor** | heptagon: >4 external momenta ⇒ rank-deficient Gram — see §2 |
| Massless internal line | IR-finite off-shell, massless masters. A near-massless line makes the integrand singular at the `x_l → 1` corner, so `scipy_direct` uses a corner-concentrated Dirichlet importance sampler; flat sampling would bias the estimate |
| Near-degenerate (tiny Gram) | `1/Gram ~ 1e5` catastrophic-cancellation stress |
| Timelike / above-threshold | explicit signed invariants injected via the `SINV` line. Above threshold the integral is complex and scipy cannot cross `F = 0`, so the `tlx_*` families are validated against the IBP mass-derivative identity `I[a_d = 2] = ± d/dm_d² I[scalar]`, the right-hand side finite-differenced from OneLOop's *complex* master — validating the coefficients in the complex plane |

**Result: 132 / 132 families PASS**, and **944 / 944** master finite parts agree
between OneLOop and feynalg.

---

## 2. Rank ≥ 3 mixed tensors, and the singular Gram

Real amplitudes produce numerators of high rank in genuinely *different* external
directions, not just the isotropic `(k²)ᵖ`. This was the last stated gap. It is
validated directly at UV-finite ranks (`r < 2(N−2)`, so the moment/scipy oracle
checks the finite part) on two spacelike geometries:

| Family | Topology | Numerator | Rank | pull (σ), both geometries |
|---|---|---|---|---|
| `mix_q1q2q3_N4` | box | `k·q1 · k·q2 · k·q3` | 3 | 1.6, 0.5 |
| `mix_q1q1q2_N4` | box | `(k·q1)² · k·q2` | 3 | 0.4, 0.1 |
| `mix_q1q2q3_dotN4` | box `[2,1,1,1]` | `k·q1 · k·q2 · k·q3` (dotted) | 3 | 0.0, 0.7 |
| `mix_llq1q2_dotN4` | box `[2,1,1,1]` | `k² · k·q1 · k·q2` (dotted) | 4 | 0.2, 0.2 |
| `mix_q1q2q3_N5` | pentagon | `k·q1 · k·q2 · k·q3` | 3 | 0.6, 0.2 |
| `mix_q1q2q3q4_N5` | pentagon | `k·q1 · k·q2 · k·q3 · k·q4` | 4 | 0.1, 0.8 |
| `mix_llq1q2_N5` | pentagon | `k² · k·q1 · k·q2` | 4 | 0.9, 0.4 |
| `mix_q1q2q3_N6` | hexagon | `k·q1 · k·q2 · k·q3` | 3 | 0.5, 0.1 |
| `mix_llq1q2q3_N6` | hexagon | `k² · k·q1 · k·q2 · k·q3` | 5 | 0.5, 0.0 |
| `mix_llq1q2q3q4_N6` | hexagon | `k² · k·q1 · k·q2 · k·q3 · k·q4` | 6 | 0.8, 0.7 |
| `mix_q1q2q3_N7` | **heptagon** | `k·q1 · k·q2 · k·q3` (singular Gram) | 3 | 0.4, 0.7 |

**22 / 22 PASS** (11 families × 2 geometries), poles cancelling to `~1e-17` and
finite parts matching the moment oracle + scipy to **0.0–1.6σ** — the mixed
transverse/reducible tensor machinery (RSP rules + Passarino–Veltman transverse
average) at rank up to 6 with distinct external directions, **including raised
propagator powers**.

### The singular-Gram case and its fix

A one-loop N-point has N−1 external momenta, and the tensor reduction inverts
their **Gram matrix**. In `d = 4` at most **four** momenta can be linearly
independent, so for **N ≥ 6** the Gram is rank-deficient and the naive inverse
crashes. (The same happens for coincident external momenta, `qᵢ = qⱼ`.) This is
unavoidable geometry, not a physics bug, and it only surfaces for tensor
numerators — scalar heptagons never invert a Gram.

**The fix is a pseudo-inverse.** Only four directions are independent; the rest
are linear combinations of them. `gram_solve` solves on a **maximal linearly
independent sub-Gram** and sets the redundant coefficients to zero. Because `rhs`
lies in the reducible span, this reproduces the full system exactly (`G·c = rhs`)
and gives the correct projection onto the external-momentum span. It lives in
[`reduce.rs`](../crates/one-loop-reduce/src/reduce.rs) (`gram_solve` →
`gram_solve_matrix` + `independent_gram_subset`), is unit-tested
(`gram_solve_matrix_handles_singular_gram`), and is validated above: the heptagon
rank-3 tensor reduces to **35 master terms** matching the oracle to 0.4σ / 0.7σ.
With it the reducer is complete — any N, any rank — and the last guarded panic in
the tensor path is retired.

*Note:* rank ≥ 3 on a **triangle** (N=3) is genuinely UV-divergent (needs `r < 2`),
so its finite part is not directly scipy-integrable; it is covered by the
reduces-cleanly unit tests and by the divergent isotropic `(k²)ᵖ` families
(`ll2d`, `ll3d`), checked via the Laurent tensor oracle.

---

## 3. MadLoop / MG5_aMC reproduction (~110 processes)

**Read this first.** The crate does the *reduction* of a loop integrand to
masters; it does **not** build the spinor-trace numerator (spenso / MadGraph),
evaluate the IR-divergent masters (OneLOop / analytic), or do the Born
interference and colour/spin sum (MadGraph). So this section is a **reproduction
record**: MG5_aMC was run on ~110 processes and produces self-consistent numbers
at its own ~14-digit accuracy, with the published anchors matching known values.
It demonstrates the *process space* the reductions cover and that the IR/colour
structure is textbook. **The direct validation of the reducer's output is §1** —
132/132 against OneLOop, the very master library MadLoop links.

Generated with **MG5_aMC v3.7.2** (python3.11, gfortran 15.2). ~21 processes were
reproduced in the first round; 96 more came fresh from three 2026-08-21 batches.

| topology | processes |
|---|---|
| triangle | e⁺e⁻ → dd̄ (**−8.936**), Drell-Yan uū → e⁺e⁻ (**= −8.936**, by crossing), uū → Z, ud̄ → W⁺, tt̄ → g, e⁺e⁻ → bb̄ (massive b) |
| box | uū → dd̄, dd̄ → dd̄, uū → gg = dd̄ → gg = gg → dd̄ (**−54.03**, crossing), gu → gu, uū → tt̄ (massive top) |
| pentagon | e⁺e⁻ → dd̄g |
| 4-gluon | gg → gg (123 loop diagrams, **−66.63**) |
| loop-induced | gg → h, gg → hh, **H → γγ** (28 W+top loops, **6.64e-2**), **γγ → γγ** (light-by-light, 186 loops, **1.05e-3**) |

### Reference points

- **e⁺e⁻ → a → dd̄ [virt=QCD]** — the primary benchmark, at the default PS point
  (`μ_R = M_Z = 91.188 GeV`, `s = 10⁶ GeV²`): Born `= 3.4754514769164148e-03`;
  virtual normalized by `Born · α_s/(2π)`: Finite `= -8.9363792407373648e+00`,
  single pole `= 8.7724371707012754e+00`, double pole `= -2.6666666666666670e+00`.
  The loop is the QCD vertex correction to the photon-quark-quark vertex — a
  **massless on-shell triangle** `C0(0,0,s;0,0,0)`, IR-divergent. The pole
  structure is understood analytically: double pole `-2 C_F = -8/3 = -2.66667`
  (exact), single pole `C_F[-3 - 2 ln(μ²/s)] ≈ 8.773` (matches `8.7724`).
- **e⁺e⁻ → tt̄ [virt=QCD]** (2026-08-21, rel. accuracy `5.1e-15`) — the massive-quark
  IR structure: **double pole exactly 0** (massive tops have no collinear
  singularity), the clean contrast with the massless anchor's `−8/3 = −2C_F`.
  Born `3.0630e-2`, virtual finite `+1.8566`, single pole `+6.5419`.
- **gg → hh / bb̄ [virt=QCD]** (loop-induced, top boxes + triangles): loop `|M|²`
  finite `= 3.4123262140874510e-05`, poles `= 0`, MG5 accuracy `5.2e-13`.
- **gg → h** (massive-top triangle, loop-induced): finite `9.37e-3`.
- **e⁺e⁻ → dd̄g** (pentagon, 5 external): 4 Born, 30(+4) loops, 18 R2, 52 UV;
  finite `= 2.4369`.
- **H → γγ [virt=QED]**: 0 Born, 28 W+top loops, finite `6.6360e-2`.
- **γγ → γγ [virt=QED]** (light-by-light): 0 Born, 186(+30) loops, finite `1.0538e-3`.

### The three batches

A **35-process batch** (5.5 min) covering V→qq̄ triangles, qq̄/gluon-initiated
boxes, the 4-gluon amplitude, loop-induced (`gg→h`, `H→γγ`, `gg→hh`) and the
`e⁺e⁻→ddg` pentagon. The **massive-vs-massless double pole** (0 for massive `b`/`t`,
`−8/3 = −2C_F` for light quarks) and the **exact colour-factor poles** (`−4C_F`,
`−4C_A = −12` for 4-gluon) fall out automatically; `e⁺e⁻→dd̄ = −8.93638` and
`gg→gg = −66.63` reproduce the prior anchors exactly.

A **second 40-process batch** (EW/QED — `e⁺e⁻→WW/ZZ/γγ`, `W→ℓν`, `z→ℓℓ`, plus new
QCD 2→2 and more pentagons) validated **40/40**, with the QED `e⁺e⁻` double pole
`−0.1315` constant across the electroweak channels and the both-massive `bb̄→tt̄`
double pole exactly 0.

A **third batch** added QCD 2→3 jets (double poles `−25/3`, `−35/3`) and
quark-initiated diboson (double pole scaling with quark charge²: `−0.0584` up vs
`−0.0146` down).

Full tables, and the regeneration commands, are in
[`../crates/one-loop-reduce/benchmarks/madloop_reference.md`](../crates/one-loop-reduce/benchmarks/madloop_reference.md).

### The gap this suite surfaced

**On-shell massless external legs** drive pinched sub-topologies degenerate. The
reducer is complete off-shell; a triangle tolerates one on-shell leg; boxes are
robust; only the ≥2-on-shell-leg triangle at rank ≥ 2 hits the degenerate-Gram
wall. The symbolic-δ wrapper (exact rational arithmetic cancels the `1/δ`
inverse-Gram poles automatically) recovers the correct value: gg→h rank-2 gives
`(1/20)[B0(0;1,1) − B0(2/5;1,1)] = −3.47483e-3` against an independent
Feynman-parameter average `−3.47483e-3` — `3.4e-7` from the numeric off-shell
extrapolation, `3.2e-10` symbolically. Full treatment in
[the frontier](04-frontier.md).

---

## 4. Application-level coverage (848 diagrams, 0 walls)

Recorded 2026-08-21 against the deployed gammaloop → reducer pipeline. The sweep
harness drove that deployment's HTTP API and is **not** part of this repo; the
numbers are kept here because they are the only large-sample evidence on *real*
physical numerators rather than constructed families.

| Metric | Value |
|---|---|
| Processes | 40 (self-energies, V→ff̄ triangles, Higgs, 4-fermion + QCD boxes, loop-induced) |
| Diagrams reduced | **848** |
| Reduced to masters | 738 |
| Graceful `zero_numerator` (numerator vanishes) | 109 |
| Degenerate-Cayley **walls** | **0** |
| Timeouts | 1 (a 36 MB light-by-light output — a resource limit, not a reduction failure) |
| D0 / box coverage | 15 of 40 processes |

Every master class A0/B0/C0/D0 was exercised on real diagrams, with **zero**
degenerate-Cayley failures across 848 physical numerators.

**Pentagons and beyond.** The graph → reducer bridge originally mapped externals
only up to `q3` (four-point); building them dynamically (`P(j) → q{j+1}`, up to
nine-point) extended it to 5+ point diagrams. Validated on **`e+e-→ddg`**: 184
one-loop diagrams, of 15 sampled **11 reduced, 4 graceful zero-numerator, 0 walls,
0 errors**, and **6 reductions reference `q4`** — the genuine five-point external
momentum the old bridge could not produce. Reductions reach `D0` and run to
~10 MB. The routing that does this now lives in
[`routing.rs`](../crates/one-loop-reduce/src/routing.rs).

---

## 5. Speed

Symbolic reduction time per topology + numerator, producing analytic
rational-in-`d` coefficients (measured 2026-07-31, generic massive kinematics):

| topology | scalar | rank-1 | rank-2 |
|---|---|---|---|
| triangle | 0.016 ms | 0.097 ms | 0.272 ms |
| box | 0.029 ms | 0.142 ms | 0.543 ms |
| pentagon | 0.504 ms | 0.727 ms | — |

Sub-millisecond across triangle / box / pentagon, scalar → rank-2. This is the
**absolute** reducer speed: the reduction is done **once** per topology + numerator
structure.

### Honest framing vs MadLoop

These numbers are **not** a raw-numerical-speed win over MadLoop, and it matters
not to claim one. The two tools do a different operation: the reducer does
**symbolic** exact-rational algebra in `d` and produces a *formula*; MadLoop does
optimized **float numerical** evaluation and produces a *number* at one
phase-space point. Symbolic reduction is inherently ~10–100× slower — the price of
analyticity, not a defect. A warmed-up per-point comparison (MadLoop's
`check_sa.f` timed over 2000 loop-ME evals after warmup, numbers still matching
`-8.9363792407`) gives **MadLoop = 13.6 µs/point**, faster than one symbolic
reduce; MadLoop is mature compiled code that paid its process-specific codegen
cost once at `output` time (~minutes).

| operation (triangle, rank-2) | this crate | MadLoop | ratio |
|---|---|---|---|
| symbolic reduction (once → analytic coeffs) | 0.27 ms | — (float codegen at build) | different task |
| per-point, re-reducing every point (current mode) | 0.27 ms | 0.0136 ms | ~20× slower (+1900%) |
| per-point, reduce-once + eval masters (projected) | ~0.003 ms | 0.0136 ms | ~4× faster (projected) |

The per-point floor is the scalar-master evaluation itself (avh_olo: C0 ≈ 0.5 µs),
which **MadLoop links too** — so a "reduce once symbolically in the invariants,
evaluate per point" mode would be master-dominated and competitive, but is not the
present differentiator.

**The value proposition is analyticity, not speed:** rational-in-`d` coefficients
times masters, produced once per topology — formulas, not numbers, reusable across
kinematics, human-readable, and composable. Report the absolute figures
("~0.02–0.7 ms/reduction, analytic output") and let the reader place them in the
one-loop landscape.

---

## Why this is trustworthy, and what it does not cover

- **Three independent master engines:** OneLOop (avh_olo, the library MadLoop
  links), feynalg (Denner-style analytic closed forms), and direct scipy
  Feynman-parameter integration of the *original* integral.
- **Two independent tensor oracles:** a symmetric-moment integrator and a
  covariant Minkowski-Gram integrator, on different computational paths.
- **Exact rational coefficients:** the reduction is symbolic in `d`, so pole
  cancellation is algebraic (`~1e-17`), not a numerical fit.

The crate performs the *reduction*; the numeric / ε-expansion of the masters comes
from the external evaluator. Genuinely singular and threshold configurations
(invariants vanishing at *different* rates) and exactly-singular Gram boxes at
all-massless kinematics remain guarded — a clean error, never garbage; see
[the frontier](04-frontier.md). Fully-massless one-loop (external *and* internal)
is scaleless = 0 in dim reg and needs no computation.

---

## Reproducing

```
cargo run --release --example emit_reductions -p one-loop-reduce > /tmp/oneloop_reductions.txt
python3 crosscheck.py /tmp/oneloop_reductions.txt
```

Prerequisites, and the rest of the harnesses, are in
[`../crates/one-loop-reduce/benchmarks/README.md`](../crates/one-loop-reduce/benchmarks/README.md).
For the algorithm being validated see [the reduction algorithm](02-reduction.md)
and [numerators](03-numerators.md); for the on-shell-massless frontier and its fix
see [the frontier](04-frontier.md); for the two full-amplitude assemblies see
[gg→h](09-ggh-formfactor.md) and [H→γγ](10-hgammagamma.md).

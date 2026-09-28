# one-loop-reduce

A symbolic one-loop IBP reducer. Given a one-loop integral family — N propagators,
their masses, the external invariants, and an arbitrary polynomial numerator in the
loop momentum — it returns `Σ cᵢ(d) · Mᵢ`, where each `Mᵢ` is one of the four scalar
master integrals `A0`/`B0`/`C0`/`D0` and each `cᵢ` is an *exact* rational function of
`d = 4 − 2ε` (for `N ≤ 4`; the `N ≥ 5` step is a four-dimensional identity, see
[below](#what-works-and-what-does-not)). The reductions are closed-form per-topology recursions implemented in
[Symbolica](https://symbolica.io); there is no Laporta engine.

**It reduces; it does not evaluate.** The masters come back as opaque function atoms —
nothing here assigns them a number or an ε-expansion. Putting values on `A0`/`B0`/`C0`/`D0`
is the job of the companion module **`oneloopmaster`**, or of any external evaluator
(`avh_olo`/OneLOop numerically, feynalg analytically). That split is the design: this half
produces formulas reusable across kinematics, the other turns them into numbers.

## Using it from Python

There is nothing here to `pip install`. `crates/one-loop-reduce-python` is a
[symbolica-community](https://github.com/symbolica-dev/symbolica-community) module: it
implements `symbolica::api::python::SymbolicaCommunityModule`, is linked into that root,
and ships inside the one Symbolica wheel. So it arrives with `pip install symbolica`, and
is imported as `symbolica.community.hep.oneloop`.

The reason is Symbolica's **global symbol table**. Symbols, and every `Expression` built
from them, live in one per-process table owned by one compiled copy of Symbolica. A
standalone extension module would link its own copy, get its own table, and its expressions
could not be combined with anything from `symbolica` itself — silently, with no error. So
the crate does not enable `pyo3/extension-module` or `pyo3/abi3` (those belong to the
consuming root) and declares `symbolica = "3.0"` as a plain crates.io requirement, so that
root's `[patch.crates-io]` can redirect it; a git dependency compiles that second copy.

Everything crosses the boundary as an `Expression`, never as a string:
`Propagator(mass_sq)`, `IntegralFamily(propagators, invariants, numerator=None,
exponents=None)`, and the returned `Reduction` (`.terms`, `.to_expression()`,
`.simplify()`) and `MasterIntegral` (`.kind`, `.head`, `.arguments`).

### A worked example

A triangle with three equal internal masses, one numerator insertion `k·q₁`:

```python
from symbolica import E, S
from symbolica.community.hep.oneloop import IntegralFamily, Propagator

dot, k, q1 = S("oneloopreduce::dot"), S("oneloopreduce::k"), S("oneloopreduce::q1")

triangle = IntegralFamily(
    propagators=[Propagator(E("msq"))] * 3,
    invariants=[E("p1sq"), E("s"), E("p2sq")],   # (r_i − r_j)², lexicographic i<j
    numerator=dot(k, q1),
)
print(triangle.reduce().simplify().to_expression())
# -1/2*p1sq*C0(p1sq,p2sq,s,msq,msq,msq)+1/2*B0(s,msq,msq)-1/2*B0(p2sq,msq,msq)
```

These coefficients are `d`-free — a property of this integral, not of the output in
general, where they are rational in the symbol `oneloopreduce::d`.

### Evaluating with oneloopmaster

The same `symbolica.community.hep.oneloop` module includes the scalar evaluator.
`master.to_oneloopmaster(mu_squared=None)` converts a reducer master to its
canonical `oneloopmaster::` head, preserves the AVH kinematic argument order, and
appends the squared renormalization scale (default `1`).
`reduction.to_oneloopmaster(...)` converts every term, preserving the exact
coefficient dependence on `oneloopreduce::d`. `to_expression()` retains the original
opaque `oneloopreduce::` heads.

```python
from symbolica.community.hep import oneloop

bubble = oneloop.IntegralFamily([oneloop.Propagator(E("0"))] * 2, [E("s")])
reduction = bubble.reduce()
_, master = reduction.terms[0]
finite, pole, double_pole = oneloop.master_coefficients(master.to_oneloopmaster())

# For a complete reduction, include d = 4 - 2 epsilon in its coefficients.
finite, pole, double_pole = oneloop.reduction_coefficients(reduction)
evaluator = oneloop.compile_native([finite, pole, double_pole], [E("s")])
print(evaluator.evaluate_complex([-2 + 0j]))
```

Do not substitute `d=4` before multiplying by the master Laurent series: terms
proportional to epsilon times a pole contribute to the finite answer. The host's
`reduction_coefficients` helper includes those contributions. Individual reduction
coefficients with poles at `d=4` need higher master orders than the evaluator
provides, and are rejected by that helper.

## The Rust API

`reduce(&IntegralFamily) -> Result<Reduction, OneLoopError>` returns a `Reduction` whose
`terms` is a `Vec<(Atom, MasterIntegral)>`; `amplitude(&IntegralFamily) -> Result<Atom,
OneLoopError>` folds those pairs into the single expression `Σ cᵢ · symbol(Mᵢ)`. The
Python example above, spelled in Rust:

```rust
use oneloopreduce::{Integral, IntegralFamily, Kinematics, Propagator, amplitude, symbols::S};

let fam = IntegralFamily {
    propagators: vec![Propagator { momentum: Atom::Zero, mass_sq: msq.clone() }; 3],
    isps: vec![],
    kinematics: Kinematics { invariants: vec![p1sq, s, p2sq] },
    targets: vec![Integral { propagator_exponents: vec![1; 3], isp_exponents: vec![] }],
    numerator: function!(S.dot, Atom::var(S.k), Atom::var(S.q1)),
};
println!("{}", amplitude(&fam)?);
```

**Two orderings, and they are not the same one.** Get these right or the answer is wrong
in a way nothing will flag. **Input** `kinematics.invariants` is the `C(N,2)` pairwise
invariants `(rᵢ − rⱼ)²` in **lexicographic** `i<j` order — `(0,1), (0,2), …, (0,N−1),
(1,2), …` — and the list must have exactly that length; `reduce()` rejects any other,
since inside the reducer a short one would read as "these legs are on shell". **Output** master arguments follow the
**AVH / OneLOop** convention, so the emitted atoms feed that evaluator directly:

    A0(m²)   B0(p², m₁², m₂²)   C0(p₁², p₂², p₁₂², m₁², m₂², m₃²)
    D0(p₁², p₂², p₃², p₄², s, t, m₁², m₂², m₃², m₄²)

So above, the lexicographic input `[p1sq, s, p2sq]` comes back out as
`C0(p1sq, p2sq, s, …)` — the third slot is `p₁₂²`, not the third invariant.

The numerator is a polynomial in the symmetric, linear `oneloopreduce::dot` over `dot(k, k)`
and `dot(k, qᵢ)`, where `qᵢ` runs over the family's own chain `q₁ … q_{N−1}` (a tadpole
accepts `q₁ … q₃`). Its coefficients may be anything free of `k`, including functions of
`d`. `reduce()` returns `OneLoopError::UnsupportedNumerator` for anything else — a
polarization vector's `dot(k, eps)`, a direction the family does not have, a bare `k`,
`1/dot(k, k)` — rather than carrying loop momentum into a coefficient; project those out
first.

`reduce()` validates the family before touching it and returns `InvalidFamily` for no
propagators, anything but exactly one target, exponent or invariant lists of the wrong
length, non-empty ISP fields (reserved), and any input symbol from the reducer's scratch
names in the `oneloopreduce` namespace: `reg_delta`, `xll`, `xq<n>`, `den<n>`,
`routing_tmp_q<n>`. Those are fixed, interned names, so a mass or prefactor called
`oneloopreduce::xll` would otherwise be silently read as the reducer's own variable.

## What works, and what does not

**Works.** Any `N`, numerators up to total degree `MAX_NUMERATOR_DEGREE = 20` in the
scalar products (each `dot(k,·)` counts one), arbitrary internal masses, and raised
propagator powers (`[2,1,1,1]`, `[3,1]`, …). Tensor reduction inverts the external Gram matrix, which
is necessarily singular for `N ≥ 6` (more than four independent external momenta in `d = 4`)
and for coincident momenta; that is handled exactly, by solving on a maximal independent
sub-Gram and zeroing the redundant directions, which are linear combinations of the kept
ones. Scalar reductions never invert a Gram at all.

**Degenerate kinematics.** The recursions divide by Gram and modified-Cayley determinants,
both of which vanish when external legs go on shell and massless — not an exotic corner but
the normal state of external gluons, photons and light quarks. Unaided, a triangle tolerates
exactly one on-shell leg and fails at two once the numerator reaches rank 2 or a propagator
power is raised; a box survives two on-shell legs at rank 2, since it pinches to triangles
rather than to null-leg bubbles. `reduce()` detects a vanishing invariant and routes to a
regularized path: replace each zero invariant with a symbolic off-shellness
`oneloopreduce::reg_delta`, run the unchanged core reducer, then take `reg_delta → 0` in
both coefficients and master arguments. This is **exact, not an approximation**:
the `1/δ` inverse-Gram poles cancel algebraically in the sum, and since the arithmetic is
exact rational there is no numerical cancellation to fight — the limit comes for free, and
the core reducer is untouched by any of it.

**Not handled: genuine thresholds.** The single shared `δ` is right when the integral is
continuous at the on-shell point. It can fail where different invariants must vanish at
*different rates* — a genuinely singular threshold, a vanishing Cayley determinant rather
than a spurious Gram one. That needs the systematic Denner–Dittmaier expansion about the
degenerate limit (Nucl. Phys. **B734** (2006) 62, hep-ph/0509141); nothing here attempts it.
What `reduce()` does guarantee is that such a case never comes back as a success: a
coefficient or master argument that ends up indeterminate or infinite is returned as
`OneLoopError::NonFiniteResult`.

**Coincident lines.** Two propagators with `(rᵢ − rⱼ)² = 0`, equal masses and equal
invariants against every other line are the same denominator as far as a scalar integral
can tell, and their Cayley determinant vanishes identically — a real singularity, which the
shared-`δ` limit gets wrong (a dotted bubble at zero momentum came back indeterminate). For
a scalar family with a raised power, `reduce()` merges such lines first, `Dᵢᵃ Dⱼᵇ = Dᵢᵃ⁺ᵇ`,
which is exact. It does not do so under a numerator, where an on-shell leg is not a zero
vector; that case is covered only by the non-finite check.

**`N ≥ 5` is exact in `d = 4`, not in `d = 4 − 2ε`.** An `N`-point scalar integral with
`N ≥ 5` is reduced by the van Neerven–Vermaseren relation, `I_N = Σᵢ cᵢ I_{N−1}⁽ⁱ⁾`, with
`cᵢ` from the bordered modified Cayley matrix. That identity holds in four dimensions. In
`d` dimensions the pentagon carries an extra `(d − 4) · I₅^(d+2)` term; the six-dimensional
pentagon is finite, so the dropped piece is `O(ε)`, and nothing here adds it back. Results
at `N ≥ 5` are therefore correct through `O(ε⁰)` when what multiplies the high-point step is
finite at `d = 4`, and not beyond. This has been validated numerically at `O(ε⁰)` (below),
not at higher orders; every `N ≤ 4` recursion is exact in `d`.

**`MAX_TOTAL_INDEX = 32`.** `reduce()` refuses any target whose propagator exponents are
negative or sum past 32. That bound bounds the *abort*, not the runtime: every recursion
level drops one unit of total index or one propagator, and overrunning the stack is a
`SIGABRT` no `catch_unwind` can intercept. It promises nothing about getting an answer
back. The tree branches `N(N−1)+1` ways per level, so a release-build dotted bubble takes
0.05 s at total index 11, 15 s at 17, and roughly 2.9× more per unit after that — about a
day at 25. A 2-propagator family with total index 32 is accepted and will not return. The
limit sits an order of magnitude below the overflow floor and far above anything that
would ever have finished.

**No ε expansion.** Coefficients stay exact in `d`; nothing is series-expanded. That is a
composition boundary, not a gap: a Laurent series in ε needs the masters' own expansions —
including the `1/ε` and `1/ε²` poles that cancel against the coefficients' `d`-dependence —
and those belong to the evaluator. Reduce here, expand in `oneloopmaster`.

## Validation

The reducer has been checked against independent engines rather than against itself.
**132 of 132** integral families — scalar and dotted, `N = 3…7`, ranks 1–6, over two
spacelike geometries — reduced and reassembled correctly, the masters evaluated by OneLOop
(`avh_olo`, the library MadLoop itself links) and feynalg, the result compared against
direct scipy Feynman-parameter integration of the original integral, and the `1/ε` and
`1/ε²` poles cancelling in every finite case. Across those families **944 of 944** master
values agreed between OneLOop and feynalg. Two full amplitudes were assembled end to end:
the gg→H massive-top form factor reproduces the closed-form `A_{1/2}(τ)` to a maximum
relative error of **3.23e-13** over six points, and the assembled `|M|²` sits within
**0.04 %** of MadLoop's `9.3702613e-3` at `α_s = 0.1114`; the rank-6 H→γγ W-boson loop —
both photon legs on shell, the hardest case the regularized path handles — gives
**Γ = 9.102 keV** against an SM LO value of about 9.1. A MadLoop/MG5_aMC suite of roughly
110 processes was reproduced separately.

The harnesses are no longer here. The seven Cargo examples under
`crates/one-loop-reduce/benchmarks/` are the emitting half and still run; the Python
drivers and the full validation record are archived, with the repository, commit and
`git show` commands to recover them recorded in [STATUS.md](STATUS.md).

## Building and testing

```bash
cargo build --workspace
SYMBOLICA_HIDE_BANNER=1 cargo test --workspace -- --test-threads=1
```

Expect **78 library + 9 binding tests**, 1 ignored (a slow dotted heptagon that passes in
release). `--test-threads=1` is a requirement, not a preference: an unlicensed Symbolica
allows one instance per process and *aborts* the moment it is touched from a second thread,
and the test binaries share a process. Every Symbolica-using test therefore calls
`ensure_symbolica_license()` first, which activates a key from `SYMBOLICA_LICENSE` if one
is set and is otherwise a no-op — the suite passes either way.

Each example's `//!` header says what it checks and how to invoke it; several take their
kinematics as integer argv pairs — for instance
`cargo run --release -p one-loop-reduce --example ggh_formfactor`.

`python/tests/test_oneloopreduce.py` is the only coverage of the FFI boundary and sits
outside CI, since it needs the module built into a symbolica-community root: add this
crate as a dependency of that root with `default-features = false`, forward the
root's `native`/`wasm` features, and call
`oneloopreduce_python::CommunityModule::register_module` on the existing
`symbolica.community.hep.oneloop` module. The host must also call its
`CommunityModule::initialize()` during the HEP initialization hook. This registers
`oneloopreduce::dot` with its `Symmetric, Linear` attributes before user code can
mention it and fix them to defaults. Both the reducer and evaluator share the
host's one Symbolica engine and symbol table. Native backend features select
GMP/MPFR and code generation explicitly, leaving allocator selection to the host.

The checked-in facade at `python/symbolica/community/oneloopreduce` provides a
legacy import alias to the combined HEP module. Regenerate the reducer's checked-in
public `.pyi` with `./scripts/gen_stubs.sh`; the community host's combined stubs also
include the evaluator and its composition helpers.

## CI and the license

`.github/workflows/ci.yml` runs `cargo fmt --all --check`, `cargo build`, `cargo clippy
--workspace --all-targets -D warnings` and the single-threaded test suite, matrixed over
Symbolica `main` and `dev` (the branch gammaloop tracks) by `sed`-ing the root patch table.

**No licence key is needed.** CI runs Symbolica restricted by default — one instance, one
thread — exactly what `--test-threads=1` already assumes, so the full suite passes. The job
logs a `::notice::` naming the mode it ran in, every time.

To run licensed instead, add a repository secret — *Settings → Secrets and variables →
Actions → New repository secret* — named exactly **`SYMBOLICA_LICENSE`**. The name matters:
that is what Symbolica's `src/license.rs` reads. symbolica-community's own workflow sets
`SYMBOLICA_LICENSE_KEY`, which nothing reads at all; its CI runs restricted too.

# One-loop reduction of shared HEP integral families

This library reduces `hep.IntegralFamily` directly to the primitive
`oneloopmaster::A0`, `B0`, `C0`, and `D0` symbols. The family, scalar products,
kinematic assumptions, and symbolic dimension are the same Feynkit objects used
by the other HEP backends. There is no separate one-loop family, propagator class,
or dot-product function to construct or translate in user code.

Native master symbols and their numerical hooks come from
[oneloopmaster](https://github.com/alphal00p/oneloopmaster). Untagged calls carry
kinematics followed by the squared scale and support exact expression inspection.
The Python exports `oneloop.A0`, `B0`, `dB0`, `C0`, and `D0` are these callable
primitive symbols. Tagged calls evaluate through their direct native Rust hooks;
lowercase `a0`, `b0`, `db0`, `c0`, and `d0` are numerical convenience functions.
Tagged calls with numeric kinematics and at least one floating-point argument
evaluate during construction at the supplied precision, for example
`oneloop.A0(0, Float("2", decimal_digits=50), 1)` with `Float` imported from
Symbolica. Untagged and exact-only calls remain symbolic.

## Python example

The binding is linked into the shared Symbolica community wheel and imported as
`symbolica.community.hep.oneloop`. It uses native Symbolica expressions throughout.

```python
from symbolica import E, S
from symbolica.community import hep
from symbolica.community.hep import oneloop

D, k, p, q, m2, s1, s2, s = S("D", "k", "p", "q", "m2", "s1", "s2", "s")
kin = (hep.Kinematics(D, momenta=[k, p, q])
       .with_scalar_product(p, p, s1)
       .with_scalar_product(q, q, s2)
       .with_scalar_product(p, q, (s-s1-s2)/2))
family = hep.IntegralFamily(
    [k], [p, q],
    [kin.scalar_product(v, v) - m2 for v in [k, k+p, k+p+q]],
    kinematics=kin,
)
reduction = oneloop.reduce(family, [1, 1, 1], numerator=kin.scalar_product(k, p))
assert reduction.dimension == D
mu2 = S("mu2")
print(reduction.to_expression(mu_squared=mu2))
# [B0(s,m2,m2,mu2) - B0(s2,m2,m2,mu2)
#  - s1*C0(s1,s2,s,m2,m2,m2,mu2)] / 2
# Every master head above belongs to oneloopmaster::.
```

Powers are explicit and follow `family.denominators`. Positive powers include
propagators, zero powers omit them, and negative powers put their denominator
expressions in the numerator. This also works with auxiliary entries appended by
`family.complete()`. The same family can be passed to the general HEP IBP backend.
`hep.Propagator` remains Feynkit's existing model propagator object; integral-family
inputs are inverse-denominator expressions.

Use a **symbolic dimension**, such as `hep.Kinematics(S("D"))`. A concrete
four-dimensional context is rejected because replacing D by 4 before multiplication
by master Laurent series loses finite contributions from epsilon times poles.

## Native evaluation and exact inspection

```python
coefficients = oneloop.reduction_coefficients(reduction, mu_squared=mu2)
point = {m2: 2, s1: -1, s2: -2, s: -3, mu2: 1}
print([coefficient.evaluate(point) for coefficient in coefficients])
```

The coefficient order is `[finite, 1/epsilon, 1/epsilon²]` in OneLoopMaster's
normalization. `reduction_coefficients` expands the shared dimension at
`D=4-2*epsilon`, then convolves the rational prefactors with tagged primitive
master calls. `Expression.evaluate` invokes their existing native Rust callbacks;
no manual master evaluator or function map is needed.

`master.to_expression(mu_squared=None)` and
`reduction.to_expression(mu_squared=None)` append the squared renormalization
scale (default 1). The untagged native master arities are:

- `A0(m², mu²)`
- `B0(p², m0², m1², mu²)`
- `C0(p1², p2², p3², m0², m1², m2², mu²)`
- `D0(p1², p2², p3², p4², s12, s23, m0², m1², m2², m3², mu²)`

A leading tag 0, -1, or -2 selects the finite, simple-pole or double-pole numerical
coefficient. For a nontrivial exact formula and a selected analytic branch:

```python
from symbolica import N, Replacement

psq = S("triangle::s", is_real=True)
mass2 = S("triangle::m2", is_positive=True)
primitive = oneloop.C0(0, 0, psq, 0, mass2, 0, 1)
all_branches = oneloop.get_expression(primitive)
selected = oneloop.select_branch(all_branches, [
    Replacement(psq, N(-2)), Replacement(mass2, N(1)),
])
print(selected[0])  # Equivalent to (pi²/6-polylog(2,1+psq/mass2))/psq.
assert selected[1:] == (N(0), N(0))
```

Branch probes are used only to resolve conditional branches. The returned formula
retains its symbolic kinematics and applies in the selected analytic region;
select again when crossing a branch cut.

## Shared-family adapter and limits

The adapter borrows the existing Feynkit family. Its quadratic-denominator
extraction owns the relation `D_i = a_i*(k+r_i)² + remainder_i`. Reduction retains
`a_i^(-power_i)`, shifts the numerator with the loop momentum, and computes all
pairwise invariants using the shared kinematics. Missing external numerator
directions are represented by auxiliary quadratic lines of power zero, preserving
the entire external Gram matrix. Dependent denominators are partial-fractioned
with the family's existing exact method.

The input must have one loop and a polynomial scalar numerator. Tensor indices
must be contracted first. Positive-power eikonal denominators are unsupported;
negative auxiliary powers may represent those scalar products in the numerator.
Propagator shifts require real coefficients. The family describes algebraic
quadratic denominators with the conventional Feynman prescription; custom
prescriptions and contour-changing complex shifts are not inferred.

The recurrence accepts nonnegative internal propagator powers with total at
most 32 and polynomial degree at most 20 in each loop scalar product. These are
bounds on implementation support, not performance guarantees: highly dotted
families can be very slow. The existing high-point and degenerate-kinematics
limitations remain; see the historical [integration review](COMMUNITY_INTEGRATION_REVIEW.md).
Reduction coefficients singular at D=4 need higher master epsilon orders than
OneLoopMaster supplies and are rejected by `reduction_coefficients`.

## Rust and builds

The public entry point is
`oneloopreduce::reduce_family(&feynkit_graph::IntegralFamily, powers, numerator)`.
The private recurrence chart stores masses, invariants, powers, and a numerator;
family definitions and quadratic decomposition belong to the shared HEP layer.
`OneLoopMasters.symbol_with_scale(&master, &mu_squared)` constructs primitive
calls, while the `MasterBasis::symbol` implementation defaults the scale to 1.

Local builds use sibling checkouts `../oneloopmaster` and
`../gammaloop/symbolica-301-citations`. The latter must include `IntegralFamily.quadratic_denominator` and
`PyIntegralFamily.as_family`; these shared-family additions are currently local
changes and must be published together with this integration before a clean
remote CI checkout can reproduce the build.
The root uses released Symbolica and Numerica 3.0.1; consuming roots must select
one shared kernel. Native numerical dependencies are excluded for WebAssembly builds.

```sh
cargo build --workspace --all-targets
cargo test --workspace -- --test-threads=1
cargo fmt --all --check
```

Python boundary tests run against the community wheel built from these checkouts:

```sh
SYMBOLICA_HIDE_BANNER=1 pytest python/tests/test_oneloopreduce.py
```

The historical validation record and its archived numerical drivers remain in
[STATUS.md](STATUS.md). The standalone integration review preserves the patches
and findings from the earlier API; its embedded patches are historical artifacts.

## Citations and upstream integration

After reduction, `symbolica.get_citations()` includes Elijah Cavan's package
credit. Merely importing the module does not add it. Scalar master use adds the
OneLoopMaster package credit and the papers requested by its README. Citation
reporting is cumulative across the process; individual HEP modules do not expose
separate Python getters.

The current reduction algorithms include upstream `main` at
`53db2ab7e6efa5e60ac31207ced0d9b578a03a5c`, including validation, coincident-line
merging and exact on-shell IBP fallback. See `UPSTREAM_PATCHES.typ` for the
remaining shared-HEP integration delta to send to the reducer author.

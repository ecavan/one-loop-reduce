# one-loop-reduce

A symbolic **one-loop IBP reducer**: it reduces any one-loop Feynman integral
with an arbitrary polynomial numerator to the four standard scalar master
integrals — **A0** (tadpole), **B0** (bubble), **C0** (triangle), **D0** (box) —
with coefficients **rational in `d = 4 − 2ε`**. The reductions are closed-form
per-topology recursions implemented in [Symbolica](https://symbolica.io); there
is no Laporta engine (that is scoped to the future ≥2-loop effort).

This crate is the **REDUCE** step only. *Evaluating* the master integrals
(putting numbers on A0/B0/C0/D0) is delegated to OneLOopBridge (avh_olo,
numeric) and feynalg (analytic), driven from the [`benchmarks/`](crates/one-loop-reduce/benchmarks/)
harnesses.

## Documentation

**Start with the [one-page summary & document map](docs/00-summary.md)** (status,
results at a glance, and where to find each thing). The full set builds on one
another — read in order:

1. [Overview](docs/01-overview.md) — what it is, the REDUCE-vs-EVALUATE scope, current status
2. [The reduction algorithm](docs/02-reduction.md) — the masters, the per-topology IBP recursions, N>4 via bordered-Cayley
3. [Tensor & dotted numerators](docs/03-numerators.md) — the `dot(k, qᵢ)` reduction and the gammaloop→family bridge
4. [The on-shell massless-leg frontier](docs/04-frontier.md) — why massless legs are hard, the off-shell-δ fix, and what's left
5. [The validation record](docs/06-validation.md) — the cross-engine method and every number (132/132 cross-engine; ~110 MadLoop processes; speed)
6. [gg→h](docs/09-ggh-formfactor.md) and [H→γγ](docs/10-hgammagamma.md) — the two full-amplitude assemblies

See also the runnable
[benchmarks guide](crates/one-loop-reduce/benchmarks/README.md).

## Quick start

```bash
cargo build
SYMBOLICA_HIDE_BANNER=1 cargo test -- --test-threads=1
cargo run -p one-loop-reduce --example golden_master   # a validation harness
```

Symbolica has a **one-instance-per-process** constraint: without a license it
allows a single instance and aborts the moment it is touched from a second
thread. So call `crate::ensure_symbolica_license()` at the top of any
Symbolica-using test, and always run the suite with `--test-threads=1`.
`ensure_symbolica_license()` activates a key from the `SYMBOLICA_LICENSE`
environment variable when one is present, and is a no-op otherwise — the suite
passes either way.

## Continuous integration

`.github/workflows/ci.yml` runs `cargo fmt --check`, `build`, `clippy -D warnings`
and the single-threaded test suite against symbolica **`main`** and **`dev`** (the
branch gammaloop tracks), by `sed`-ing the root `[patch.crates-io]` table.

**No licence key is required.** By default CI runs Symbolica in restricted mode —
one instance, one thread — which is exactly what `--test-threads=1` already
assumes, so the full suite passes. The job logs a `::notice::` saying which mode
it ran in, every time.

To run licensed instead, add a repository secret (**Settings → Secrets and
variables → Actions → New repository secret**) named `SYMBOLICA_LICENSE`. The
name matters: Symbolica reads `SYMBOLICA_LICENSE` (`src/license.rs`).
symbolica-community's own workflow sets `SYMBOLICA_LICENSE_KEY`, which nothing
reads — its CI runs restricted too.

`python/tests/test_oneloopreduce.py` is the only coverage of the FFI boundary and
is deliberately **not** in CI — it needs the module built into a
symbolica-community root. Run it by hand:

```bash
SYMBOLICA_HIDE_BANNER=1 pytest python/tests/test_oneloopreduce.py
```

## Python bindings

`crates/one-loop-reduce-python` is **not a pip-installable package**. It is a
[symbolica-community](https://github.com/symbolica-dev/symbolica-community)
module: it implements `symbolica::api::python::SymbolicaCommunityModule` and is
linked into that root, which publishes it as `symbolica.community.oneloopreduce`
and builds the one wheel through maturin. There is nothing here to `pip install`
on its own, and this crate deliberately does **not** enable
`pyo3/extension-module` or `pyo3/abi3` — those belong to the consuming root, and
Cargo's feature unification would push them onto every other consumer.

### The surface

Everything crosses the boundary as a Symbolica `Expression`, never as a string.

| | |
| --- | --- |
| `Propagator(mass_sq)` | a loop line; `.mass_sq` |
| `IntegralFamily(propagators, invariants, numerator=None, exponents=None)` | `.propagators` `.invariants` `.numerator` `.exponents` `.reduce()` |
| `Reduction` | `.terms` `.to_expression()` `.simplify()` `len()` |
| `MasterIntegral` | `.kind` `.head` `.arguments` `.to_expression()` `==` |

```python
from symbolica import E, S
from symbolica.community.oneloopreduce import IntegralFamily, Propagator

dot, k, q1 = S("oneloopreduce::dot"), S("oneloopreduce::k"), S("oneloopreduce::q1")

triangle = IntegralFamily(
    propagators=[Propagator(E("msq"))] * 3,
    invariants=[E("p1sq"), E("s"), E("p2sq")],   # (r_i - r_j)^2, lexicographic i<j
    numerator=dot(k, q1),
)
print(triangle.reduce().simplify().to_expression())
# -1/2*p1sq*C0(p1sq,p2sq,s,msq,msq,msq)+1/2*B0(s,msq,msq)-1/2*B0(p2sq,msq,msq)
```

### Wiring it into symbolica-community

Add the dependency to that root's `Cargo.toml` and register it in its
`core` `#[pymodule]`. The Rust crate name is `oneloopreduce_python` (the package
is `one-loop-reduce-python`):

```rust
register_module!(m, oneloopreduce_python::CommunityModule);
```

Then copy `python/symbolica/community/oneloopreduce/` into that repo's
`python/` tree. `__init__.py` is the usual two-line facade:

```python
from ..oneloopreduce_native import *

initialize_module()
```

`initialize_module()` is not decoration. `CommunityModule::initialize` forces the
`oneloopreduce::symbols::S` `LazyLock`, and `dot` is declared
`symbol!("oneloopreduce::dot"; Symmetric, Linear)`. Symbolica fixes a symbol's
attributes the first time it is mentioned, so if user code parsed
`oneloopreduce::dot(k, q1)` before that ran, `dot` would already exist with
default attributes and the `symbol!` would panic with *"Symbol redefined with new
attributes"*.

### Type stubs

`python/symbolica/community/oneloopreduce/__init__.pyi` is generated, and is
checked in for editor support. Regenerate it with `./scripts/gen_stubs.sh`.
Once the crate is wired into a symbolica-community checkout, that root's own
`cargo run --bin stub_gen --features python_stubgen` is the authoritative path.

## Layout

```
crates/one-loop-reduce/         the reducer — the mergeable library
  src/                          the reduction engine
  benchmarks/                   runnable validation harnesses: rust/ (Cargo
                                examples) + python/ (cross-engine oracles)
                                + reference records
crates/one-loop-reduce-python/  Symbolica-community Python bindings (module
                                name `oneloopreduce`)
python/                         the Python facade + generated type stubs, to be
                                merged into symbolica-community's python/ tree
python/tests/                   FFI-boundary tests (need a built module; not CI)
scripts/gen_stubs.sh            regenerates the .pyi
docs/                           the documentation set linked above
```

Names, fixed and used consistently:

| thing | name |
| --- | --- |
| Cargo package | `one-loop-reduce` |
| Rust library (`use ...`) | `oneloopreduce` |
| Symbolica symbol namespace | `oneloopreduce::` |
| Python module | `oneloopreduce` |

`symbolica` is declared as a plain crates.io version requirement so a consuming
workspace root can redirect it through `[patch.crates-io]`; the git redirect
for local builds lives in this repo's workspace-root `Cargo.toml`.

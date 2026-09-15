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
5. [The FeynmanEngine app](docs/05-app.md) — the live "Reduce to masters" button, end to end
6. [Validation & benchmarks](docs/06-benchmarks.md) — the cross-engine method, in prose
7. [Benchmark report](docs/07-benchmark-report.md) — the results and numbers (132/132 cross-engine; ~110 MadLoop processes)

See also the [CHANGELOG](docs/CHANGELOG.md) and the runnable
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

## Layout

```
crates/one-loop-reduce/         the reducer — the mergeable library
  src/                          the reduction engine
  benchmarks/                   runnable validation harnesses: rust/ (Cargo
                                examples) + python/ (cross-engine oracles)
                                + reference records
crates/one-loop-reduce-python/  Symbolica-community Python bindings (module
                                name `oneloopreduce`)
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

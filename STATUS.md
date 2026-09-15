# Status

Running record of where this repo is. Newest entries at the top.

---

## 2026-09-15 — the unbounded recursion is fixed in the library

`reduce()` now returns `Result<Reduction, OneLoopError>` and refuses, before entering the
recursion, any target whose propagator indices are negative or total more than
`MAX_TOTAL_INDEX = 32`. The guard the Python constructor was carrying is gone; the binding
forwards the library's error, so the bound lives in one place and the Rust API is no longer
exposed.

### Why a bound rather than a deeper base case

`reduce_cayley` descends depth-first and every level drops either one unit of total index
or one propagator, so stack depth is bounded by `sum(exponents) + N` — and a negative index
never bottoms out at all. Overrunning the stack is an *abort*: `catch_unwind` cannot
intercept it and neither can pyo3's trampoline.

Measured on this reducer, macOS arm64:

| | |
|---|---|
| `reduce_cayley` frame, debug | 3408 bytes (exact, from stack-pointer deltas) |
| `reduce_cayley` frame, release | 560 bytes |
| overflow depth, 2 MiB thread stack, debug | 486 — SIGABRT |
| overflow depth, 8 MiB thread stack, debug | 2323 — SIGABRT |
| observed depth | exactly the total index, confirmed on bubble/triangle/N-gon |

Runtime hits the wall far sooner, because the tree branches `N(N-1)+1` ways per level. A
release-build dotted massless bubble: total index 11 → 0.05 s, 13 → 0.23 s, 15 → 1.84 s,
17 → 15.3 s, a factor ~2.9 per unit after that (≈ a day at 25). So 32 sits an order of
magnitude below the overflow floor and far above anything that would ever have returned —
it rejects only input that was never going to finish, and truncates nothing.

### Cost

`reduce`'s signature change touched 66 call sites across tests and the benchmark examples
(mechanical `.unwrap()`), and `amplitude()` became `Result` with it. That was worth it over
a second, checked entry point: one entry point means the abort is not reachable at all.

---

## 2026-09-14 — extracted from gammaloop, Python module works

The crate is standalone and the Symbolica-community binding runs end to end.

### What happened

Lifted out of `alphal00p/gammaloop` (`crates/oneloop`, PR #86) via `git subtree split`,
so all 49 original commits carry over with their authors and dates. Restructured into a
two-crate workspace, cut the gammaloop dependency, renamed the symbol namespace, and
added the Python module.

| | |
|---|---|
| Library | `crates/one-loop-reduce` — lib name `oneloopreduce` |
| Python module | `crates/one-loop-reduce-python` — registers as `symbolica.community.oneloopreduce` |
| Symbol namespace | `oneloopreduce::` (was `oneloop::`) |
| Deps | `symbolica 2.2`, `thiserror 2.0`. Nothing else. |

### Verified

Run on 2026-09-14, macOS arm64, Symbolica restricted mode (no `SYMBOLICA_LICENSE` set).

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` from clean | 0 errors, 49 s |
| `cargo test --workspace -- --test-threads=1` | 62 library + 11 binding, 0 failed, 1 ignored |
| ignored slow test (`dotted_heptagon…`) | passes in release, 4.8 s |
| `cargo clippy --workspace --all-targets` | 0 warnings |
| `cargo fmt --all --check` | clean |
| `cargo tree -d` | one symbolica, one numerica, one graphica |

Test-function names were diffed against gammaloop's `crates/oneloop/src`: 63 on both
sides, identical name for name. Nothing regressed through the rename.

### Physics still reproduces

The point of the extraction was that nothing about the physics should change. It didn't —
these are the same numbers gammaloop produces.

| Validation | Result |
|---|---|
| gg→H form factor `A_{1/2}(τ)`, six points | max rel. err **3.23e-13** |
| gg→H assembled \|M\|² vs MadLoop `9.3702613e-3` | ratio **0.9996** at α_s = 0.1114 |
| H→γγ, W loop, rank 6 | PASS, Γ = **9.102 keV** (SM LO ≈ 9.1) |
| RSP identity, 2000 random pentagons | PASS |
| Heptagon power-lowering, degenerate | PASS |
| Pentagon reduction, dotted pentagon | PASS |

Requires `oneloop_bridge` (avh_olo) on the Python path. See
`crates/one-loop-reduce/benchmarks/README.md`.

### The Python surface

`symbolica.community.oneloopreduce` — typed objects throughout, no strings.

```python
from symbolica import E
from symbolica.community.oneloopreduce import IntegralFamily, Propagator

fam = IntegralFamily(
    [Propagator(E("msq"))] * 3,
    [E("p1sq"), E("s"), E("p2sq")],
    numerator=E("oneloopreduce::dot(oneloopreduce::k, oneloopreduce::q1)"),
)
print(fam.reduce().simplify().to_expression())
# -1/2*p1sq*C0(p1sq,p2sq,s,msq,msq,msq) + 1/2*B0(s,msq,msq) - 1/2*B0(p2sq,msq,msq)
```

`Propagator`, `IntegralFamily`, `Reduction`, `MasterIntegral`. Coefficients come back as
Symbolica `Expression`s, so they compose with everything else in the shared kernel.

Verified against a symbolica-community-shaped root built with maturin — the module
imports, the symbols keep their `Symmetric, Linear` attributes, and the reduction above
runs in a real interpreter.

### Decisions

| Decision | Choice | Status |
|---|---|---|
| Repo location | Personal (`ecavan/one-loop-reduce`) | Ben's suggestion — "it is your project" |
| Name | `oneloopreduce` | **Needs Ben + Cedric to confirm.** Cedric's `oneloopmaster` currently returns `get_name() -> "oneloop"`; two modules with the same name silently clobber each other in `sys.modules`. |
| gammaloop PR #86 | Leave open | It gave the work access to gammaloop; theirs to close |
| `oneloop` branch in gammaloop | Keep | The deployed app needs `--reduce` on a buildable ref |
| History | Preserved via `subtree split` | 49 commits, original authors and dates |
| `symbolica` dependency | Plain version requirement, `"2.2"` | **Not** a git dep — see below |
| `Cargo.lock` | Tracked | The patch table points at a moving branch |

### Why symbolica must not be a git dependency

`[patch.crates-io]` at a consuming root only rewrites dependencies declared as
crates.io dependencies. Declare `symbolica = { git = … }` in a member crate and Cargo
treats it as a separate source, compiles a second copy of Symbolica, and produces two
disjoint global symbol tables — expressions from one cannot be used with the other, with
no error. Both members use `symbolica = "2.2"`; the git redirect lives only in the root
`[patch.crates-io]`, where it governs local builds and is inert when symbolica-community
consumes the crate.

Confirmed empirically: a simulated community root resolves to one symbolica, and
`cargo tree -d` reports no duplicates.

### Known issues

**`momentum` is not exposed on `Propagator`.** The field exists in the Rust struct but
`reduce` never reads it; external offsets come from `invariants`. Exposing a field the
reducer ignores would be a trap. `bridge.rs` is the only consumer.

### Deviations from the original plan

- `default-features = false` on symbolica does not build. Symbolica declares
  `numerica = { default-features = false }`, so dropping symbolica's defaults leaves
  numerica with neither an integer nor a float backend (28 errors). Naming a backend
  explicitly is not portable either — crates.io 2.2.0 spells them `gmp`/`no_gmp` while
  `symbolica-dev/symbolica@main` spells them `integer-gmp`/`float-mpfr`. Plain
  `symbolica = "2.2"` with defaults is the only spelling that resolves against both, and
  is what `example_extension` uses.
- `LicenseManager` is at `symbolica::license::LicenseManager` on `main`, not the crate
  root.

### For gammaloop

gammaloop's workspace requests symbolica's `gmp` feature, which no longer exists on
`symbolica-dev/symbolica@main` (it is `integer-gmp` there). That will break whoever moves
gammaloop off the `dev` branch.

### Not verified

- No build against the **real** `symbolica-community` — the check used a faithful but
  minimal root (no vakint/idenso/spynso3, and `set_python_integration_functions` was not
  called). Feature unification across all community modules at once is untested.
- abi3 build not exercised through the community root's `module` feature path.
- Restricted mode only; `SYMBOLICA_LICENSE` is not set here.
- The `.pyi` has not been run through a type checker.
- No Python test suite committed — the interpreter tests were run in a scratch directory.
- No CI in this repo yet.

### Next

1. Get the name confirmed by Ben and Cedric before anything is registered upstream.
2. CI — GitHub Environment gating `SYMBOLICA_LICENSE`, matrix over symbolica `main` and
   `dev`. Note: Symbolica reads `SYMBOLICA_LICENSE`; symbolica-community's own workflow
   sets `SYMBOLICA_LICENSE_KEY`, which is read nowhere, so that leg runs restricted.
3. Commit the Python tests so the interpreter-level behaviour is covered permanently.
4. Fix the recursion bound in `reduce.rs` properly, rather than guarding at the boundary.
5. Promote the routing logic out of `bridge.rs` into a model-agnostic module.
6. `reduce_diagram(FeynmanDiagram)` — most of `bridge.rs` deletes itself once FeynKit
   hands over `MomentumSignature.integer_coefficients()` directly.
7. Sign the Ruijl Research CLA (commits must be authored from the signature email), then
   PR to `symbolica-dev/symbolica-community`.

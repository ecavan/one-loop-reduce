//! Symbolica-community Python bindings for [`oneloopreduce`].
//!
//! This crate is **not** a standalone Python package. It implements
//! [`SymbolicaCommunityModule`] and is linked into the
//! [symbolica-community](https://github.com/symbolica-dev/symbolica-community)
//! root, which registers it with its `register_module!` macro and publishes it
//! as `symbolica.community.oneloopreduce`. See `README.md` for the wiring.
//!
//! The surface is deliberately typed: every kinematic input and every
//! coefficient crosses the boundary as a Symbolica `Expression`
//! ([`PythonExpression`]), never as a string that has to be re-parsed.

use std::panic;

use oneloopreduce::family::{
    Integral, IntegralFamily as RsIntegralFamily, Kinematics, Propagator as RsPropagator,
};
use oneloopreduce::masters::{MasterBasis, MasterIntegral as RsMasterIntegral, OneLoopMasters};
use oneloopreduce::reduce::{Reduction as RsReduction, reduce as rs_reduce};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use symbolica::api::python::{PythonExpression, SymbolicaCommunityModule};
use symbolica::atom::Atom;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{
    define_stub_info_gatherer,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
};

/// The Python module name. Every `#[pyclass]` below must declare
/// `module = "symbolica.community.oneloopreduce"` to match.
const MODULE_NAME: &str = "oneloopreduce";

/// The symbolica-community entry point.
pub struct CommunityModule;

impl SymbolicaCommunityModule for CommunityModule {
    fn get_name() -> String {
        MODULE_NAME.to_string()
    }

    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // No Symbolica symbols may be created here -- that is what `initialize`
        // is for. Registering classes is symbol-free.
        m.add_class::<Propagator>()?;
        m.add_class::<IntegralFamily>()?;
        m.add_class::<Reduction>()?;
        m.add_class::<MasterIntegral>()?;
        Ok(())
    }

    fn initialize(_py: Python) -> PyResult<()> {
        // Force the `oneloopreduce::symbols::S` `LazyLock` *now*, while we still
        // own the registration order.
        //
        // `dot` is declared `symbol!("oneloopreduce::dot"; Symmetric, Linear)`.
        // Symbolica interns a symbol on first mention and fixes its attributes
        // there and then, so if a user parsed `oneloopreduce::dot(k, q1)` before
        // this ran, `dot` would already exist with default attributes and the
        // `symbol!` inside the `LazyLock` would panic with
        // "Symbol redefined with new attributes". Touching one field forces the
        // whole block, so all eleven symbols are registered here.
        let _ = oneloopreduce::symbols::S.dot;
        Ok(())
    }
}

/// Run `f`, turning a Rust panic into a message instead of letting it cross the
/// FFI boundary (where it would be undefined behaviour).
///
/// The reducer asserts on malformed input -- most of those cases are rejected
/// with a clean `ValueError` in the constructors below, but this is the backstop
/// for the rest. A thread-local flag keeps the default hook from dumping a
/// panic message and backtrace to stderr on the way out; panics raised anywhere
/// else, including on other threads, still print as usual.
fn catch_panic<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    use std::cell::Cell;
    use std::sync::Once;

    thread_local! {
        static SILENT: Cell<bool> = const { Cell::new(false) };
    }
    static INSTALL: Once = Once::new();

    INSTALL.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !SILENT.with(|s| s.get()) {
                previous(info);
            }
        }));
    });

    SILENT.with(|s| s.set(true));
    let out = panic::catch_unwind(panic::AssertUnwindSafe(f));
    SILENT.with(|s| s.set(false));

    out.map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "the reducer panicked".to_string())
    })
}

fn reduction_failed(message: String) -> pyo3::PyErr {
    PyValueError::new_err(format!("one-loop reduction failed: {message}"))
}

/// Check the shape of an integral family and resolve its propagator exponents.
///
/// Split out of [`IntegralFamily::new`] so the refusals can be unit-tested
/// without standing up a Python interpreter.
fn resolve_exponents(
    propagators: usize,
    invariants: usize,
    exponents: Option<Vec<i32>>,
) -> Result<Vec<i32>, String> {
    let n = propagators;
    if n == 0 {
        return Err("an integral family needs at least one propagator".to_string());
    }

    // The reducer indexes `invariants` by a hard-coded permutation of the C(n,2)
    // lexicographic slots and asserts on the length. Catch it here so the user
    // gets a sentence instead of an assertion.
    let expected = n * (n - 1) / 2;
    if invariants != expected {
        return Err(format!(
            "a {n}-point family needs {expected} pairwise invariants \
             (r_i - r_j)^2 in lexicographic i<j order, got {invariants}"
        ));
    }

    // The values are the reducer's business: it refuses a negative or unbounded
    // index itself, out of `reduce`, so there is one bound in one place.
    match exponents {
        Some(e) if e.len() != n => Err(format!(
            "a {n}-point family needs {n} propagator exponents, got {}",
            e.len()
        )),
        Some(e) => Ok(e),
        None => Ok(vec![1; n]),
    }
}

// ---------------------------------------------------------------------------
// Propagator
// ---------------------------------------------------------------------------

/// A loop propagator `1 / ((k + r)^2 - mass_sq)`.
///
/// Only the mass is carried here. The external offset `r` never enters the
/// reduction directly: the reducer works from the pairwise invariants
/// `(r_i - r_j)^2` that you hand to `IntegralFamily`.
///
/// ## Examples
/// ```python
/// from symbolica import E
/// from symbolica.community.oneloopreduce import Propagator
///
/// massless = Propagator(E("0"))
/// massive = Propagator(E("mt^2"))
/// ```
///
/// Parameters
/// ----------
/// mass_sq : Expression
///     The propagator mass squared. Use `E("0")` for a massless line.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    name = "Propagator",
    module = "symbolica.community.oneloopreduce"
)]
#[derive(Clone)]
pub struct Propagator {
    mass_sq: Atom,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl Propagator {
    #[new]
    #[pyo3(signature = (mass_sq))]
    fn new(mass_sq: PythonExpression) -> Self {
        Propagator {
            mass_sq: mass_sq.expr,
        }
    }

    /// The propagator mass squared.
    #[getter]
    fn mass_sq(&self) -> PythonExpression {
        self.mass_sq.clone().into()
    }

    fn __repr__(&self) -> String {
        format!("Propagator({})", self.mass_sq)
    }
}

// ---------------------------------------------------------------------------
// IntegralFamily
// ---------------------------------------------------------------------------

/// A one-loop integral family: N propagators, their external kinematics, and a
/// numerator polynomial in `dot(k, ...)`.
///
/// ## Examples
/// ```python
/// from symbolica import E, S
/// from symbolica.community.oneloopreduce import IntegralFamily, Propagator
///
/// # A massless bubble with an off-shell external leg and a unit numerator.
/// family = IntegralFamily(
///     propagators=[Propagator(E("0")), Propagator(E("0"))],
///     invariants=[E("s")],
/// )
/// reduction = family.reduce()
/// print(reduction.to_expression())
///
/// # A rank-one massless triangle numerator.
/// dot, k, q1 = S("oneloopreduce::dot"), S("oneloopreduce::k"), S("oneloopreduce::q1")
/// triangle = IntegralFamily(
///     propagators=[Propagator(E("0"))] * 3,
///     invariants=[E("p1sq"), E("s"), E("p2sq")],
///     numerator=dot(k, q1),
/// )
/// ```
///
/// Parameters
/// ----------
/// propagators : Sequence[Propagator]
///     The N loop propagators, in the order that fixes the labelling `r_0 .. r_{N-1}`.
/// invariants : Sequence[Expression]
///     The `C(N, 2)` pairwise invariants `(r_i - r_j)^2`, in lexicographic `i < j`
///     order: `(0,1), (0,2), ..., (0,N-1), (1,2), ...`. A zero entry is treated as
///     an on-shell leg and is off-shell regularized internally.
/// numerator : Optional[Expression]
///     A polynomial in the symmetric linear dot product `oneloopreduce::dot`, built
///     from `dot(k, k)` and `dot(k, q_i)`. Defaults to `1` (a scalar integral).
/// exponents : Optional[Sequence[int]]
///     The power of each propagator. Defaults to `[1] * N`. Must be non-negative
///     and sum to at most `oneloopreduce::MAX_TOTAL_INDEX`, which `reduce()`
///     enforces.
///
/// Raises
/// ------
/// ValueError
///     If `propagators` is empty, or if `invariants` or `exponents` has the wrong
///     length for an N-point family.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    name = "IntegralFamily",
    module = "symbolica.community.oneloopreduce"
)]
#[derive(Clone)]
pub struct IntegralFamily {
    inner: RsIntegralFamily,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl IntegralFamily {
    #[new]
    #[pyo3(signature = (propagators, invariants, numerator = None, exponents = None))]
    fn new(
        propagators: Vec<Propagator>,
        invariants: Vec<PythonExpression>,
        numerator: Option<PythonExpression>,
        exponents: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        let propagator_exponents =
            resolve_exponents(propagators.len(), invariants.len(), exponents)
                .map_err(PyValueError::new_err)?;

        Ok(IntegralFamily {
            inner: RsIntegralFamily {
                propagators: propagators
                    .into_iter()
                    .map(|p| RsPropagator {
                        // The reducer takes the external offsets from
                        // `invariants`; this field is only used by the
                        // (Rust-only) gammaloop graph bridge.
                        momentum: Atom::Zero,
                        mass_sq: p.mass_sq,
                    })
                    .collect(),
                isps: vec![],
                kinematics: Kinematics {
                    invariants: invariants.into_iter().map(|i| i.expr).collect(),
                },
                targets: vec![Integral {
                    propagator_exponents,
                    isp_exponents: vec![],
                }],
                numerator: numerator.map(|n| n.expr).unwrap_or(Atom::num(1)),
            },
        })
    }

    /// The propagators of the family.
    #[getter]
    fn propagators(&self) -> Vec<Propagator> {
        self.inner
            .propagators
            .iter()
            .map(|p| Propagator {
                mass_sq: p.mass_sq.clone(),
            })
            .collect()
    }

    /// The `C(N, 2)` pairwise invariants, in lexicographic `i < j` order.
    #[getter]
    fn invariants(&self) -> Vec<PythonExpression> {
        self.inner
            .kinematics
            .invariants
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// The numerator polynomial.
    #[getter]
    fn numerator(&self) -> PythonExpression {
        self.inner.numerator.clone().into()
    }

    /// The power of each propagator.
    #[getter]
    fn exponents(&self) -> Vec<i32> {
        self.inner.targets[0].propagator_exponents.clone()
    }

    /// Reduce the family to the scalar masters `A0`, `B0`, `C0` and `D0`.
    ///
    /// ## Examples
    /// ```python
    /// reduction = family.reduce()
    /// for coefficient, master in reduction.terms:
    ///     print(master.kind, coefficient)
    /// ```
    ///
    /// Returns
    /// -------
    /// Reduction
    ///     The coefficient of each master integral.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the reduction fails -- on kinematics the reducer cannot handle, or
    ///     on a propagator index that is negative or whose total is beyond the
    ///     depth the recursion can reach. The Rust-side message is included.
    ///
    /// Notes
    /// -----
    /// The GIL is held for the whole call. Symbolica aborts the process when an
    /// unlicensed instance is touched from a second thread, so releasing it
    /// would turn a concurrent call into a crash rather than a speed-up.
    fn reduce(&self) -> PyResult<Reduction> {
        let reduction = catch_panic(|| rs_reduce(&self.inner))
            .map_err(reduction_failed)?
            .map_err(|e| reduction_failed(e.to_string()))?;
        Ok(Reduction {
            terms: reduction.terms,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "IntegralFamily({} propagators, numerator={})",
            self.inner.propagators.len(),
            self.inner.numerator
        )
    }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

/// The result of reducing an `IntegralFamily`: a linear combination of scalar
/// master integrals.
///
/// ## Examples
/// ```python
/// reduction = family.reduce()
///
/// len(reduction)
/// # 2
///
/// for coefficient, master in reduction.terms:
///     print(f"{coefficient} * {master.to_expression()}")
///
/// reduction.simplify().to_expression()
/// ```
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    name = "Reduction",
    module = "symbolica.community.oneloopreduce"
)]
#[derive(Clone)]
pub struct Reduction {
    terms: Vec<(Atom, RsMasterIntegral)>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl Reduction {
    /// The `(coefficient, master)` pairs of the reduction.
    #[getter]
    fn terms(&self) -> Vec<(PythonExpression, MasterIntegral)> {
        self.terms
            .iter()
            .map(|(coefficient, master)| {
                (
                    coefficient.clone().into(),
                    MasterIntegral {
                        inner: master.clone(),
                    },
                )
            })
            .collect()
    }

    /// Contract the reduction into a single expression over the `A0`/`B0`/`C0`/`D0`
    /// heads.
    ///
    /// ## Examples
    /// ```python
    /// family.reduce().to_expression()
    /// # oneloopreduce::B0(s,0,0)
    /// ```
    ///
    /// Returns
    /// -------
    /// Expression
    ///     `sum(coefficient * master.to_expression())` over every term.
    fn to_expression(&self) -> PyResult<PythonExpression> {
        let basis = OneLoopMasters;
        catch_panic(|| {
            self.terms
                .iter()
                .fold(Atom::Zero, |acc, (coefficient, master)| {
                    acc + coefficient * basis.symbol(master)
                })
        })
        .map(Into::into)
        .map_err(reduction_failed)
    }

    /// Cancel every coefficient down to lowest terms.
    ///
    /// ## Examples
    /// ```python
    /// reduction = family.reduce().simplify()
    /// ```
    ///
    /// Returns
    /// -------
    /// Reduction
    ///     A new reduction; the receiver is left untouched.
    fn simplify(&self) -> PyResult<Reduction> {
        let terms = catch_panic(|| {
            RsReduction {
                terms: self.terms.clone(),
            }
            .simplify()
            .terms
        })
        .map_err(reduction_failed)?;
        Ok(Reduction { terms })
    }

    fn __len__(&self) -> usize {
        self.terms.len()
    }

    fn __repr__(&self) -> String {
        format!("Reduction({} terms)", self.terms.len())
    }
}

// ---------------------------------------------------------------------------
// MasterIntegral
// ---------------------------------------------------------------------------

/// One of the four scalar one-loop master integrals.
///
/// Instances come out of `Reduction.terms`; there is no public constructor.
///
/// ## Examples
/// ```python
/// _, master = family.reduce().terms[0]
///
/// master.kind
/// # 'bubble'
///
/// master.arguments
/// # [s, 0, 0]
///
/// master.to_expression()
/// # oneloopreduce::B0(s,0,0)
/// ```
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    eq,
    name = "MasterIntegral",
    module = "symbolica.community.oneloopreduce"
)]
#[derive(Clone, PartialEq)]
pub struct MasterIntegral {
    inner: RsMasterIntegral,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl MasterIntegral {
    /// The topology: `'tadpole'`, `'bubble'`, `'triangle'` or `'box'`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            RsMasterIntegral::Tadpole { .. } => "tadpole",
            RsMasterIntegral::Bubble { .. } => "bubble",
            RsMasterIntegral::Triangle { .. } => "triangle",
            RsMasterIntegral::Box { .. } => "box",
        }
    }

    /// The head symbol of the master: `'A0'`, `'B0'`, `'C0'` or `'D0'`.
    #[getter]
    fn head(&self) -> &'static str {
        match self.inner {
            RsMasterIntegral::Tadpole { .. } => "A0",
            RsMasterIntegral::Bubble { .. } => "B0",
            RsMasterIntegral::Triangle { .. } => "C0",
            RsMasterIntegral::Box { .. } => "D0",
        }
    }

    /// The kinematic arguments, in the order the head takes them.
    ///
    /// - `A0(m_sq)`
    /// - `B0(p_sq, m1_sq, m2_sq)`
    /// - `C0(p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)`
    /// - `D0(p1_sq, p2_sq, p3_sq, p4_sq, s, t, m1_sq, m2_sq, m3_sq, m4_sq)`
    #[getter]
    fn arguments(&self) -> Vec<PythonExpression> {
        let args: Vec<&Atom> = match &self.inner {
            RsMasterIntegral::Tadpole { m_sq } => vec![m_sq],
            RsMasterIntegral::Bubble { p_sq, m1_sq, m2_sq } => vec![p_sq, m1_sq, m2_sq],
            RsMasterIntegral::Triangle {
                p1_sq,
                p2_sq,
                p12_sq,
                m1_sq,
                m2_sq,
                m3_sq,
            } => vec![p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq],
            RsMasterIntegral::Box {
                p1_sq,
                p2_sq,
                p3_sq,
                p4_sq,
                s,
                t,
                m1_sq,
                m2_sq,
                m3_sq,
                m4_sq,
            } => vec![p1_sq, p2_sq, p3_sq, p4_sq, s, t, m1_sq, m2_sq, m3_sq, m4_sq],
        };
        args.into_iter().cloned().map(Into::into).collect()
    }

    /// The master as a Symbolica function call on its head.
    ///
    /// ## Examples
    /// ```python
    /// master.to_expression()
    /// # oneloopreduce::C0(p1sq,p2sq,s,0,0,0)
    /// ```
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The `A0`/`B0`/`C0`/`D0` head, in the `oneloopreduce` namespace, applied
    ///     to `arguments`.
    fn to_expression(&self) -> PyResult<PythonExpression> {
        catch_panic(|| OneLoopMasters.symbol(&self.inner))
            .map(Into::into)
            .map_err(reduction_failed)
    }

    fn __repr__(&self) -> String {
        let args: Vec<String> = self
            .arguments()
            .into_iter()
            .map(|a| a.expr.to_string())
            .collect();
        format!("MasterIntegral({}({}))", self.head(), args.join(", "))
    }
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);

#[cfg(test)]
mod tests {
    use super::resolve_exponents;

    #[test]
    fn defaults_the_exponents_to_all_ones() {
        assert_eq!(resolve_exponents(3, 3, None).unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn accepts_dotted_propagators() {
        assert_eq!(
            resolve_exponents(2, 1, Some(vec![3, 2])).unwrap(),
            vec![3, 2]
        );
    }

    #[test]
    fn rejects_an_empty_family() {
        assert!(
            resolve_exponents(0, 0, None)
                .unwrap_err()
                .contains("at least one propagator")
        );
    }

    #[test]
    fn rejects_the_wrong_invariant_count() {
        let err = resolve_exponents(3, 1, None).unwrap_err();
        assert!(err.contains("needs 3 pairwise invariants"), "{err}");
        assert!(err.contains("got 1"), "{err}");
    }

    #[test]
    fn rejects_the_wrong_exponent_count() {
        let err = resolve_exponents(2, 1, Some(vec![1, 1, 1])).unwrap_err();
        assert!(err.contains("needs 2 propagator exponents"), "{err}");
    }

    /// A zero exponent is a pinched line, not an error -- the reducer deletes the
    /// row and column and carries on.
    #[test]
    fn allows_a_pinched_line() {
        assert_eq!(
            resolve_exponents(2, 1, Some(vec![0, 1])).unwrap(),
            vec![0, 1]
        );
    }

    /// Values are the reducer's problem now: the constructor takes them as given
    /// and `reduce()` is what refuses the ones the recursion cannot walk down.
    #[test]
    fn leaves_the_index_bound_to_the_reducer() {
        assert_eq!(
            resolve_exponents(2, 1, Some(vec![-1, 5000])).unwrap(),
            vec![-1, 5000]
        );
    }
}

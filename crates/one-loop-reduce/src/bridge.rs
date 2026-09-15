//! rewrite a gammaloop *scalar* numerator into oneloop's dot convention.
//!
//! When gammaloop is asked for
//! `save dot --output-full-numerator --do-gamma-algebra --do-color-algebra`, the emitted numerator
//! is a polynomial in Minkowski dot products of the loop momentum `K(0,·)` and the external momenta
//! `P(i,·)`. A dot product `a·b` is written as a shared-index product
//! `a(mink(4,idx)) b(mink(4,idx))` — Einstein summation on the repeated `mink(4,idx)` — rather than
//! as a `dot(...)` function. The oneloop reducer instead consumes a polynomial in the symmetric-linear
//! [`crate::symbols`] `dot(k, q_i)` / `dot(k, k)`.
//!
//! Momentum map: loop `K(0,·)` -> `oneloopreduce::k`; externals `P(j,·)` -> `oneloopreduce::q{j+1}` (built
//! dynamically up to `MAX_MOMENTUM_ID`, so pentagons and beyond are handled).
//!
//! Only the tensor half lives here; once the offsets are signed sums of the `q` symbols the
//! rest is model-agnostic integer combinatorics in [`crate::routing`].

use symbolica::atom::{Atom, AtomCore, Symbol};
use symbolica::{function, symbol};

use crate::error::OneLoopError;
use crate::family::{Integral, IntegralFamily, Kinematics, Propagator};
use crate::routing::{self, MAX_MOMENTUM_ID};
use crate::symbols::S;

/// The names gammaloop's `reduce_bridge.rs` still imports.
pub use self::{
    LoopEdge as GammaloopEdge, TensorHeads as GammaloopHeads,
    family_from_tensor_numerator as family_from_gammaloop,
};

#[derive(Clone, Copy)]
pub struct TensorHeads {
    pub loop_mom: Symbol,
    pub external_mom: Symbol,
    pub index: Symbol,
    pub metric: Symbol,
}

fn bare_momentum(head: Symbol, id: i64, oneloop_sym: Atom, index: Symbol) -> (Atom, Atom) {
    let tensor = function!(head, Atom::num(id), function!(index, Atom::num(4)));
    (tensor, oneloop_sym)
}

/// The loop/external momenta in bare form (loop `K(0)` -> k, externals `P(0..)` -> q1..).
fn bare_momenta(heads: &TensorHeads) -> Vec<(Atom, Atom)> {
    let mut moms = vec![bare_momentum(
        heads.loop_mom,
        0,
        Atom::var(S.k),
        heads.index,
    )];
    for (j, q) in routing::external_syms().into_iter().enumerate() {
        moms.push(bare_momentum(heads.external_mom, j as i64, q, heads.index));
    }
    moms
}

/// A gammaloop momentum tensor `head(id, index(4, idx_))`
fn known_momentum(head: Symbol, id: i64, oneloop_sym: Atom, index: Symbol) -> (Atom, Atom) {
    let idx = Atom::var(symbol!("idx_"));
    let tensor = function!(head, Atom::num(id), function!(index, Atom::num(4), idx));
    (tensor, oneloop_sym)
}

/// The loop/external momenta the bridge recognizes (loop `K(0)` -> k, externals `P(0..)` -> q1..),
/// built dynamically up to `MAX_MOMENTUM_ID` so pentagons and beyond (`P(3,·)`, …) are handled.
fn known_momenta(heads: &TensorHeads) -> Vec<(Atom, Atom)> {
    let mut moms = vec![known_momentum(
        heads.loop_mom,
        0,
        Atom::var(S.k),
        heads.index,
    )];
    for (j, q) in routing::external_syms().into_iter().enumerate() {
        moms.push(known_momentum(heads.external_mom, j as i64, q, heads.index));
    }
    moms
}

/// Rewrite the shared-index momentum contractions of a gammaloop scalar numerator into oneloop
/// `dot(...)` form. Self-contractions `a·a` (which Symbolica stores as squares) and contractions
/// `a·b` between two distinct momenta (shared-index products) both become `dot(...)`.
pub fn numerator_to_dot_form(num: &Atom, heads: &TensorHeads) -> Atom {
    let moms = known_momenta(heads);
    let mut out = num.clone();
    // Self-contractions `a·a` appear as squares `mom(mink(4,i))^2` (the shared index makes the
    // two factors identical, so Symbolica collects them into a power).
    for (tensor, sym) in &moms {
        let square = tensor * tensor;
        let dot = function!(S.dot, sym.clone(), sym.clone());
        out = out.replace(square.to_pattern()).with(dot);
    }
    // Contractions `a·b` between two distinct momenta appear as shared-index products.
    for (i, (tensor_i, sym_i)) in moms.iter().enumerate() {
        for (j, (tensor_j, sym_j)) in moms.iter().enumerate() {
            if i == j {
                continue;
            }
            let contraction = tensor_i * tensor_j;
            let dot = function!(S.dot, sym_i.clone(), sym_j.clone());
            out = out.replace(contraction.to_pattern()).with(dot);
        }
    }
    // Metric-dot form `g(a, b)` produced by `simplify_metrics` (both self `a·a` and distinct `a·b`).
    let bare = bare_momenta(heads);
    for (tensor_i, sym_i) in &bare {
        for (tensor_j, sym_j) in &bare {
            let g = function!(heads.metric, tensor_i.clone(), tensor_j.clone());
            let dot = function!(S.dot, sym_i.clone(), sym_j.clone());
            out = out.replace(g.to_pattern()).with(dot);
        }
    }
    out
}

/// The external-momentum offset of a propagator, extracted from its gammaloop `lmb_rep`
pub fn external_offset_from_lmb_rep(lmb_rep: &Atom, heads: &TensorHeads) -> Atom {
    // A single wildcard that swallows the whole `mink(4, idx)` argument.
    let any_index = Atom::var(symbol!("midx_"));
    let mut offset = lmb_rep.clone();
    for l in 0..MAX_MOMENTUM_ID {
        let loop_mom = function!(heads.loop_mom, Atom::num(l), any_index.clone());
        offset = offset.replace(loop_mom.to_pattern()).with(Atom::Zero);
    }
    for j in 0..MAX_MOMENTUM_ID {
        let external = function!(heads.external_mom, Atom::num(j), any_index.clone());
        let q = routing::q(j as usize);
        offset = offset.replace(external.to_pattern()).with(q);
    }
    offset
}

pub struct LoopEdge {
    pub lmb_rep: Atom,
    pub mass_sq: Atom,
}

/// Is there still a tensor with this head in the expression?
fn has_residual_head(expr: &Atom, head: Symbol) -> bool {
    let args = Atom::var(symbol!("bridge_residual_args___"));
    let marker = Atom::var(symbol!("oneloopreduce::bridge_residual_marker"));
    let pat = function!(head, args);
    expr.replace(pat.to_pattern()).with(marker) != *expr
}

/// The reducer treats anything that is not `dot(k,k)` or `dot(k,q_a)` as a *constant*
/// coefficient (`numerator_to_monos`). That is sound exactly when the numerator's entire
/// dependence on the loop momentum has been captured in `dot(...)` symbols -- i.e. when no
/// `loop_mom` tensor survives the translation.
///
/// A surviving one means the loop momentum is carried by an uncontracted Lorentz index, and
/// the reducer would pull it *out of the integral*. That happens whenever
///
/// * the loop momentum is contracted with something that is not a momentum, e.g. a
///   polarization vector: `dot(k, eps)` is not in the Gram basis, so the caller must
///   contract the polarizations away first (the gg->h form factor does this by applying
///   the transverse projector `g^{mu nu} - (q2^mu q1^nu)/(q1.q2)` to the numerator, which
///   leaves only `dot(k,k)` and `dot(k,q_a)`);
/// * it is contracted with an external whose id is at or beyond `MAX_MOMENTUM_ID`, so the
///   pair was never rewritten; or
/// * `heads.metric` / `heads.index` are stale after a spenso rename, so the `g(k, .)` form
///   was never recognized.
///
/// Leftover *external-only* tensors are deliberately not an error: they are genuinely
/// constant with respect to the loop momentum, so absorbing them into a coefficient is
/// correct.
fn check_loop_momentum_is_contracted(
    expr: &Atom,
    heads: &TensorHeads,
    what: &str,
) -> Result<(), String> {
    if has_residual_head(expr, heads.loop_mom) {
        return Err(format!(
            "{what} still contains an untranslated `{}` (loop momentum) tensor: `{expr}`. \
             Either a Lorentz index is uncontracted (e.g. the loop momentum is contracted \
             with a polarization vector), an external id is at or beyond \
             MAX_MOMENTUM_ID={MAX_MOMENTUM_ID}, or the expected head symbols are stale.",
            heads.loop_mom
        ));
    }
    Ok(())
}

/// Assemble a reducer [`IntegralFamily`] from a gammaloop one-loop numerator and its internal
/// edges. The numerator is translated to dot form and *relabelled into the reducer's chain
/// basis*; each edge contributes a massive propagator; and the external kinematics are the
/// pairwise invariants of the edges' external offsets.
///
/// Fails rather than degrading whenever the gammaloop routing cannot be expressed in the
/// reducer's conventions -- see [`routing::chain_slots`] and
/// [`routing::check_numerator_directions`].
pub fn family_from_tensor_numerator(
    numerator: &Atom,
    edges: &[LoopEdge],
    heads: &TensorHeads,
) -> Result<IntegralFamily, OneLoopError> {
    let fail = |reason: String| OneLoopError::ExtractionFailed { reason };
    if edges.is_empty() {
        return Err(fail(
            "a one-loop family needs at least one propagator".into(),
        ));
    }
    let offsets: Vec<Atom> = edges
        .iter()
        .map(|e| external_offset_from_lmb_rep(&e.lmb_rep, heads))
        .collect();
    for offset in &offsets {
        check_loop_momentum_is_contracted(offset, heads, "a propagator offset").map_err(fail)?;
    }
    let dirs: Vec<Vec<i32>> = offsets
        .iter()
        .map(routing::offset_dirs)
        .collect::<Result<_, _>>()
        .map_err(fail)?;
    // gammaloop hands the propagators over in `iter_edges()` order, which is not the loop
    // order; put them into the reducer's chain first, then read the slots off that.
    let order = routing::chain_order(&dirs).map_err(fail)?;
    let dirs: Vec<Vec<i32>> = order.iter().map(|&i| dirs[i].clone()).collect();
    let offsets: Vec<Atom> = order.iter().map(|&i| offsets[i].clone()).collect();
    let edges: Vec<&LoopEdge> = order.iter().map(|&i| &edges[i]).collect();
    let slots = routing::chain_slots(&dirs).map_err(fail)?;

    let dotted = numerator_to_dot_form(numerator, heads);
    check_loop_momentum_is_contracted(&dotted, heads, "the translated numerator").map_err(fail)?;
    let n = edges.len();
    routing::check_numerator_directions(&dotted, &slots, n).map_err(fail)?;
    let numerator = routing::relabel_numerator(&dotted, &slots);

    Ok(IntegralFamily {
        propagators: edges
            .iter()
            .map(|e| Propagator {
                momentum: Atom::Zero,
                mass_sq: e.mass_sq.clone(),
            })
            .collect(),
        isps: vec![],
        kinematics: Kinematics {
            invariants: routing::invariants_from_offsets(&offsets),
        },
        targets: vec![Integral {
            propagator_exponents: vec![1; n],
            isp_exponents: vec![],
        }],
        numerator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mink4(idx: i64) -> Atom {
        function!(symbol!("mink"), Atom::num(4), Atom::num(idx))
    }

    /// Standalone gammaloop heads for tests (self-consistent with the `K`/`P`/`mink` inputs built
    /// above; the real glue passes gammalooprs's `GS.loop_mom`/`GS.external_mom`/`spenso::mink`).
    fn heads() -> TensorHeads {
        TensorHeads {
            loop_mom: symbol!("K"),
            external_mom: symbol!("P"),
            index: symbol!("mink"),
            metric: symbol!("g"),
        }
    }

    #[test]
    fn contracts_metric_dot_form_into_dot() {
        crate::ensure_symbolica_license();
        // `simplify_metrics` can leave a scalar product as `g(a, b)` (bare momenta): map it too.
        let bare = |head: &str, id: i64| {
            function!(
                symbol!(head),
                Atom::num(id),
                function!(symbol!("mink"), Atom::num(4))
            )
        };
        let g = |a: Atom, b: Atom| function!(symbol!("g"), a, b);
        // g(K,K) = k·k ; g(K,P) = k·q1
        let input =
            &g(bare("K", 0), bare("K", 0)) + &(Atom::num(2) * g(bare("K", 0), bare("P", 0)));
        let got = numerator_to_dot_form(&input, &heads());
        let kk = function!(S.dot, Atom::var(S.k), Atom::var(S.k));
        let kq1 = function!(S.dot, Atom::var(S.k), Atom::var(S.q1));
        let want = (&kk + &(Atom::num(2) * &kq1)).expand();
        assert_eq!(got.expand(), want);
    }

    #[test]
    fn contracts_loop_external_product_into_dot() {
        crate::ensure_symbolica_license();

        let input = &function!(symbol!("K"), Atom::num(0), mink4(5))
            * &function!(symbol!("P"), Atom::num(0), mink4(5));
        let got = numerator_to_dot_form(&input, &heads());
        let want = function!(S.dot, Atom::var(S.k), Atom::var(S.q1));
        assert_eq!(got, want);
    }

    #[test]
    fn contracts_loop_self_square_into_dot_kk() {
        crate::ensure_symbolica_license();

        let kmom = function!(symbol!("K"), Atom::num(0), mink4(2));
        let input = &kmom * &kmom;
        let got = numerator_to_dot_form(&input, &heads());
        let want = function!(S.dot, Atom::var(S.k), Atom::var(S.k));
        assert_eq!(got, want);
    }

    #[test]
    fn contracts_mixed_rank2_numerator() {
        crate::ensure_symbolica_license();

        let k = |i: i64| function!(symbol!("K"), Atom::num(0), mink4(i));
        let p0 = |i: i64| function!(symbol!("P"), Atom::num(0), mink4(i));
        let p1 = |i: i64| function!(symbol!("P"), Atom::num(1), mink4(i));
        let input = Atom::num(2) * &k(1) * &p0(1) * &k(2) * &k(2) - &p0(3) * &p1(3);
        let got = numerator_to_dot_form(&input, &heads());
        let kk = function!(S.dot, Atom::var(S.k), Atom::var(S.k));
        let kq1 = function!(S.dot, Atom::var(S.k), Atom::var(S.q1));
        let q1q2 = function!(S.dot, Atom::var(S.q1), Atom::var(S.q2));
        let want = (Atom::num(2) * &kq1 * &kk - &q1q2).expand();
        assert_eq!(got.expand(), want);
    }

    #[test]
    fn maps_high_externals_for_pentagon_and_beyond() {
        crate::ensure_symbolica_license();
        // A pentagon (5-point) carries externals up to P(3); the bridge must map P(3) -> q4
        // (and higher). K·P(3) -> dot(k, q4).
        let input = &function!(symbol!("K"), Atom::num(0), mink4(7))
            * &function!(symbol!("P"), Atom::num(3), mink4(7));
        let got = numerator_to_dot_form(&input, &heads());
        let want = function!(
            S.dot,
            Atom::var(S.k),
            Atom::var(symbol!("oneloopreduce::q4"))
        );
        assert_eq!(got, want);
    }

    #[test]
    fn external_offset_drops_loop_and_maps_externals() {
        crate::ensure_symbolica_license();
        let k = function!(symbol!("K"), Atom::num(0), mink4(9));
        let p0 = function!(symbol!("P"), Atom::num(0), mink4(9));
        assert_eq!(external_offset_from_lmb_rep(&k, &heads()), Atom::Zero);
        let offset = external_offset_from_lmb_rep(&(&k - &p0), &heads());
        assert_eq!(offset, Atom::num(-1) * Atom::var(S.q1));
    }

    #[test]
    fn scalar_massless_bubble_reduces_to_b0() {
        crate::ensure_symbolica_license();

        let k = function!(symbol!("K"), Atom::num(0), mink4(0));
        let p0 = function!(symbol!("P"), Atom::num(0), mink4(0));
        let edges = vec![
            LoopEdge {
                lmb_rep: k.clone(),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &k - &p0,
                mass_sq: Atom::Zero,
            },
        ];
        let fam = family_from_tensor_numerator(&Atom::num(1), &edges, &heads()).unwrap();
        let r = crate::reduce::reduce(&fam).unwrap();
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, crate::masters::MasterIntegral::Bubble { .. })),
            "expected a B0 master, got {:?}",
            r.terms.iter().map(|(_, m)| m).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rank1_bubble_numerator_reduces_through_the_bridge() {
        crate::ensure_symbolica_license();

        let k = |i: i64| function!(symbol!("K"), Atom::num(0), mink4(i));
        let p0 = |i: i64| function!(symbol!("P"), Atom::num(0), mink4(i));
        let numerator = &k(1) * &p0(1);
        let edges = vec![
            LoopEdge {
                lmb_rep: k(0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &k(0) - &p0(0),
                mass_sq: Atom::Zero,
            },
        ];
        let fam = family_from_tensor_numerator(&numerator, &edges, &heads()).unwrap();
        // NOTE the minus sign. The propagators are `k` and `k - P(0)`, so the reducer's chain
        // direction is `q1^reducer = r_2 - r_1 = -P(0)`, and the gammaloop numerator `k.P(0)`
        // is `-dot(k, q1)` in the reducer's own basis. This assertion used to read
        // `+dot(k, q1)`, i.e. it asserted the bug (see
        // `ggh_rank1_through_the_bridge_respects_the_propagator_routing`).
        assert_eq!(
            fam.numerator,
            -function!(S.dot, Atom::var(S.k), Atom::var(S.q1))
        );
        // Value, not just "a Bubble appeared". With `m1 = m2 = 0`, the bubble RSP rule
        // `k.w = (D2 - D1 - m1 + m2 - p^2)/2` gives
        //   int (k.P(0)) / (D1 D2) = int -(k.w)/(D1 D2)
        //                          = (1/2)[A0(0) - A0(0) + p^2 B0(p^2,0,0)]
        //                          = (p^2/2) B0(p^2, 0, 0),
        // the massless tadpoles vanishing in dim reg. Here `p^2 = dot(q1,q1)`.
        let p_sq = function!(S.dot, Atom::var(S.q1), Atom::var(S.q1));
        let r = crate::reduce::reduce(&fam).unwrap();
        assert_terms(
            &r,
            &[(
                &p_sq / Atom::num(2),
                crate::masters::MasterIntegral::Bubble {
                    p_sq: p_sq.clone(),
                    m1_sq: Atom::Zero,
                    m2_sq: Atom::Zero,
                },
            )],
        );
    }

    // ---------------------------------------------------------------------------------------
    // End-to-end: known-good numbers driven through `family_from_tensor_numerator`.
    //
    // Everything above tests the dot-product rewriting in isolation, and every benchmark in
    // `benchmarks/` hand-builds its `IntegralFamily`. These tests instead push the *validated*
    // gg->h numbers through the translation layer, so a bridge bug can no longer hide
    // behind a correct reducer. Those numbers are the massive-top gg->h triangle at
    // m_H = 125, m_t = 173: the reduction agrees with OneLOop and with the closed-form
    // A_{1/2}(tau) to 1e-13 over six (m_H, m_t) points.
    //
    // The three closed forms the tests below pin, for the top triangle with
    // invariants (0, s, 0) and all three internal masses m:
    //
    //     1              ->  C0
    //     k^2            ->  m^2 * C0 + B0(0)
    //     (k.q1)(k.q2)   ->  (s/4) B0(0) - (s/8) B0(s)
    // ---------------------------------------------------------------------------------------

    /// `m_H^2` for `m_H = 125` -- the first of the six validated (m_H, m_t) points.
    const GGH_S: i64 = 15625;
    /// `m_t^2` for `m_t = 173`.
    const GGH_MTSQ: i64 = 29929;

    fn kk(idx: i64) -> Atom {
        function!(symbol!("K"), Atom::num(0), mink4(idx))
    }
    fn pp(j: i64, idx: i64) -> Atom {
        function!(symbol!("P"), Atom::num(j), mink4(idx))
    }
    fn dot(a: &Atom, b: &Atom) -> Atom {
        function!(S.dot, a.clone(), b.clone())
    }
    fn q(a: usize) -> Atom {
        Atom::var(symbol!(format!("oneloopreduce::q{a}")))
    }

    /// The reducer-side family `benchmarks/rust/ggh_formfactor.rs` hand-builds: three top
    /// propagators, invariants `(q1^2, (q1+q2)^2, q2^2) = (0, s, 0)`.
    fn ggh_handbuilt(numerator: Atom) -> IntegralFamily {
        IntegralFamily {
            propagators: (0..3)
                .map(|_| Propagator {
                    momentum: Atom::Zero,
                    mass_sq: Atom::num(GGH_MTSQ),
                })
                .collect(),
            isps: vec![],
            kinematics: Kinematics {
                invariants: vec![Atom::Zero, Atom::num(GGH_S), Atom::Zero],
            },
            targets: vec![Integral {
                propagator_exponents: vec![1, 1, 1],
                isp_exponents: vec![],
            }],
            numerator,
        }
    }

    /// The gammaloop-side edges for the same triangle.
    ///
    /// `sign = +1` routes the propagators as `k, k+P(0), k+P(0)+P(1)`; `sign = -1` as
    /// `k, k-P(0), k-P(0)-P(1)`, which is what gammaloop's LMB actually emits -- verified by
    /// `gammalooprs::graph::parse::tests::test_load`, whose one-loop triangle prints
    /// `lmb_rep="-1*P(0,a___)+-1*P(1,a___)+K(0,a___)"`.
    fn ggh_edges(sign: i64) -> Vec<LoopEdge> {
        let s = Atom::num(sign);
        let m = Atom::num(GGH_MTSQ);
        vec![
            LoopEdge {
                lmb_rep: kk(0),
                mass_sq: m.clone(),
            },
            LoopEdge {
                lmb_rep: &kk(0) + &(&s * &pp(0, 0)),
                mass_sq: m.clone(),
            },
            LoopEdge {
                lmb_rep: &kk(0) + &(&s * &pp(0, 0)) + &(&s * &pp(1, 0)),
                mass_sq: m,
            },
        ]
    }

    /// On-shell gluons: `q1^2 = q2^2 = 0`, `q1.q2 = s/2`. The bridge builds the invariants
    /// *symbolically* from the offsets, so this is what turns them into the benchmark's
    /// numeric `(0, s, 0)` and puts both families on the same reduction path.
    fn ggh_on_shell(a: &Atom) -> Atom {
        a.replace(dot(&q(1), &q(1)).to_pattern())
            .with(Atom::Zero)
            .replace(dot(&q(2), &q(2)).to_pattern())
            .with(Atom::Zero)
            .replace(dot(&q(1), &q(2)).to_pattern())
            .with(Atom::num(GGH_S) / Atom::num(2))
            .expand()
    }

    /// Build the gg->h family through the bridge and put it on the benchmark's kinematic point.
    fn ggh_through_bridge(sign: i64, numerator: &Atom) -> IntegralFamily {
        let mut fam = family_from_tensor_numerator(numerator, &ggh_edges(sign), &heads())
            .expect("the gg->h triangle must translate");
        fam.kinematics.invariants = fam.kinematics.invariants.iter().map(ggh_on_shell).collect();
        fam
    }

    /// The reduction as a sorted, expanded `(master, coeff)` list, so two reductions can be
    /// compared without depending on term order.
    fn normalized(r: &crate::reduce::Reduction) -> Vec<(String, Atom)> {
        let mut out: Vec<(String, Atom)> = r
            .terms
            .iter()
            .map(|(c, m)| (format!("{m:?}"), c.expand()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn c0_ggh() -> crate::masters::MasterIntegral {
        crate::masters::MasterIntegral::Triangle {
            p1_sq: Atom::Zero,
            p2_sq: Atom::Zero,
            p12_sq: Atom::num(GGH_S),
            m1_sq: Atom::num(GGH_MTSQ),
            m2_sq: Atom::num(GGH_MTSQ),
            m3_sq: Atom::num(GGH_MTSQ),
        }
    }
    fn b0_ggh(p_sq: i64) -> crate::masters::MasterIntegral {
        crate::masters::MasterIntegral::Bubble {
            p_sq: Atom::num(p_sq),
            m1_sq: Atom::num(GGH_MTSQ),
            m2_sq: Atom::num(GGH_MTSQ),
        }
    }

    /// The whole reduction as an explicit `[(coeff, master)]` set, for exact assertions.
    fn assert_terms(r: &crate::reduce::Reduction, want: &[(Atom, crate::masters::MasterIntegral)]) {
        let mut got = normalized(r);
        let mut expect: Vec<(String, Atom)> = want
            .iter()
            .map(|(c, m)| (format!("{m:?}"), c.expand()))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        expect.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, expect);
    }

    #[test]
    fn ggh_scalar_triangle_through_the_bridge_is_the_unit_c0() {
        crate::ensure_symbolica_license();
        let fam = ggh_through_bridge(1, &Atom::num(1));
        // The bridge must reproduce the benchmark's hand-built kinematics exactly.
        assert_eq!(
            fam.kinematics.invariants,
            vec![Atom::Zero, Atom::num(GGH_S), Atom::Zero],
            "bridge invariants must match the hand-built (0, s, 0)"
        );
        assert_terms(
            &crate::reduce::reduce(&fam).unwrap(),
            &[(Atom::num(1), c0_ggh())],
        );
    }

    #[test]
    fn ggh_ksq_through_the_bridge_matches_the_documented_closed_form() {
        crate::ensure_symbolica_license();
        // The closed form for the gg->h top triangle: `k^2 -> m^2 * C0 + B0(0)`.
        let numerator = &kk(1) * &kk(1);
        let fam = ggh_through_bridge(1, &numerator);
        assert_eq!(fam.numerator, dot(&Atom::var(S.k), &Atom::var(S.k)));
        assert_terms(
            &crate::reduce::reduce(&fam).unwrap(),
            &[(Atom::num(GGH_MTSQ), c0_ggh()), (Atom::num(1), b0_ggh(0))],
        );
    }

    #[test]
    fn ggh_kq1_kq2_through_the_bridge_matches_the_documented_closed_form() {
        crate::ensure_symbolica_license();
        // The closed form for the gg->h top triangle:
        // `(k.q1)(k.q2) -> (s/4) B0(0) - (s/8) B0(s)`.
        let numerator = (&kk(1) * &pp(0, 1)) * (&kk(2) * &pp(1, 2));
        let fam = ggh_through_bridge(1, &numerator);
        assert_eq!(
            fam.numerator,
            &dot(&Atom::var(S.k), &q(1)) * &dot(&Atom::var(S.k), &q(2))
        );
        assert_terms(
            &crate::reduce::reduce(&fam).unwrap(),
            &[
                (Atom::num(GGH_S) / Atom::num(4), b0_ggh(0)),
                (Atom::num(-GGH_S) / Atom::num(8), b0_ggh(GGH_S)),
            ],
        );
    }

    #[test]
    fn ggh_rank1_through_the_bridge_matches_the_handbuilt_family() {
        crate::ensure_symbolica_license();
        // `k.q1` on the `+P` routing: the bridge's q_a and the reducer's chain
        // `r_i = q1 + ... + q_{i-1}` coincide, so this must equal the hand-built reduction.
        let numerator = &kk(1) * &pp(0, 1);
        let got = crate::reduce::reduce(&ggh_through_bridge(1, &numerator)).unwrap();
        let want = crate::reduce::reduce(&ggh_handbuilt(dot(&Atom::var(S.k), &q(1)))).unwrap();
        assert_eq!(normalized(&got), normalized(&want));
        // and pin the value itself, not just the agreement
        assert_terms(
            &got,
            &[
                (Atom::num(1) / Atom::num(2), b0_ggh(GGH_S)),
                (Atom::num(-1) / Atom::num(2), b0_ggh(0)),
            ],
        );
    }

    #[test]
    fn ggh_rank1_through_the_bridge_respects_the_propagator_routing() {
        crate::ensure_symbolica_license();
        // THE test the bridge was missing. gammaloop routes the triangle as
        // `k, k-P(0), k-P(0)-P(1)`, so the reducer's chain directions are
        // `q_a^red = r_{a+1} - r_a = -P(a-1)`. The numerator `k.P(0)` is therefore
        // `-dot(k, q1^red)` in the reducer's own basis -- i.e. the hand-built family with a
        // NEGATED rank-1 numerator. Anything else is an odd-rank sign error.
        let numerator = &kk(1) * &pp(0, 1);
        let got = crate::reduce::reduce(&ggh_through_bridge(-1, &numerator)).unwrap();
        let want = crate::reduce::reduce(&ggh_handbuilt(-dot(&Atom::var(S.k), &q(1)))).unwrap();
        assert_eq!(
            normalized(&got),
            normalized(&want),
            "the `-P` routing gammaloop actually emits must flip the sign of an odd-rank \
             numerator relative to the `+P` routing"
        );
        assert_terms(
            &got,
            &[
                (Atom::num(-1) / Atom::num(2), b0_ggh(GGH_S)),
                (Atom::num(1) / Atom::num(2), b0_ggh(0)),
            ],
        );
    }

    #[test]
    fn ggh_rank2_through_the_bridge_is_routing_blind() {
        crate::ensure_symbolica_license();
        // Even-rank monomials are invariant under `k -> -k`, so both routings must agree --
        // this is the control that says the rank-1 failure above is a *sign* bug and not a
        // general breakage of the bridge.
        let numerator = (&kk(1) * &pp(0, 1)) * (&kk(2) * &pp(1, 2));
        let plus = crate::reduce::reduce(&ggh_through_bridge(1, &numerator)).unwrap();
        let minus = crate::reduce::reduce(&ggh_through_bridge(-1, &numerator)).unwrap();
        assert_eq!(normalized(&plus), normalized(&minus));
    }

    // ---------------------------------------------------------------------------------------
    // Box and pentagon through the bridge.
    //
    // `invariants_from_offsets` emits the C(n,2) pairwise invariants in *lexicographic* pair
    // order, and the reducer reads them back with two different hard-coded permutations:
    // `base_gram_from_pairwise`'s `idx(i,j)` for N > 4, and `box_numerator(inv(0), inv(3),
    // inv(5), inv(2), inv(1), inv(4))` for the box-with-numerator branch. Nothing pinned
    // either agreement with a number -- and the hand-built
    // `reduce::tests::box_with_linear_numerator_reduces_to_masters` actually feeds *physics*
    // order into the lexicographic slot, which its kind-only assertions cannot see.
    // ---------------------------------------------------------------------------------------

    /// Edges `k, k - P(0), ..., k - P(0) - ... - P(n-2)`: the routing gammaloop emits.
    fn chain_edges(masses_sq: &[i64]) -> Vec<LoopEdge> {
        let mut edges = Vec::new();
        let mut rep = kk(0);
        for (i, &m) in masses_sq.iter().enumerate() {
            if i > 0 {
                rep = &rep - &pp(i as i64 - 1, 0);
            }
            edges.push(LoopEdge {
                lmb_rep: rep.clone(),
                mass_sq: Atom::num(m),
            });
        }
        edges
    }

    /// Substitute a numeric external Gram `dot(q_i, q_j) -> value` into an expression.
    fn substitute_gram(a: &Atom, gram: &[((usize, usize), Atom)]) -> Atom {
        let mut out = a.clone();
        for ((i, j), v) in gram {
            out = out
                .replace(dot(&q(*i), &q(*j)).to_pattern())
                .with(v.clone());
        }
        out.expand()
    }

    /// The Gram whose chain `r_i = q1 + ... + q_{i-1}` reproduces the lexicographic pairwise
    /// invariants `[1, 2, 3, 4, 5, 6]` of `reduce::tests::scalar_box_reduces_to_unit_d0`
    /// (which independently pins `s = 2`, `t = 5` for that list).
    ///
    /// `q1^2 = s01 = 1`, `q2^2 = s12 = 4`, `q3^2 = s23 = 6`,
    /// `(q1+q2)^2 = s02 = 2 => q1.q2 = -3/2`, `(q2+q3)^2 = s13 = 5 => q2.q3 = -5/2`,
    /// `(q1+q2+q3)^2 = s03 = 3 => q1.q3 = 0`.
    fn box_gram() -> Vec<((usize, usize), Atom)> {
        let half = |n: i64| Atom::num(n) / Atom::num(2);
        vec![
            ((1, 1), Atom::num(1)),
            ((2, 2), Atom::num(4)),
            ((3, 3), Atom::num(6)),
            ((1, 2), half(-3)),
            ((2, 3), half(-5)),
            ((1, 3), Atom::Zero),
        ]
    }

    /// The Gram behind `reduce::tests::scalar_pentagon_reduces_to_five_boxes`' invariant list
    /// `[3, 5, 7, 9, 4, 6, 8, 5, 7, 6]`; solved from the chain and verified to reconstruct
    /// all ten entries exactly.
    fn pentagon_gram() -> Vec<((usize, usize), Atom)> {
        let half = |n: i64| Atom::num(n) / Atom::num(2);
        vec![
            ((1, 1), Atom::num(3)),
            ((2, 2), Atom::num(4)),
            ((3, 3), Atom::num(5)),
            ((4, 4), Atom::num(6)),
            ((1, 2), Atom::num(-1)),
            ((1, 3), Atom::Zero),
            ((1, 4), Atom::Zero),
            ((2, 3), half(-3)),
            ((2, 4), Atom::Zero),
            ((3, 4), Atom::num(-2)),
        ]
    }

    fn family_on_gram(
        numerator: &Atom,
        edges: &[LoopEdge],
        gram: &[((usize, usize), Atom)],
    ) -> IntegralFamily {
        let mut fam =
            family_from_tensor_numerator(numerator, edges, &heads()).expect("must translate");
        fam.kinematics.invariants = fam
            .kinematics
            .invariants
            .iter()
            .map(|a| substitute_gram(a, gram))
            .collect();
        fam
    }

    #[test]
    fn box_invariants_through_the_bridge_land_in_lexicographic_order() {
        crate::ensure_symbolica_license();
        let fam = family_on_gram(&Atom::num(1), &chain_edges(&[0, 0, 0, 0]), &box_gram());
        assert_eq!(
            fam.kinematics.invariants,
            (1..=6).map(Atom::num).collect::<Vec<_>>(),
            "the bridge's pairwise invariants must come out in lexicographic pair order"
        );
        // Same anchor as `reduce::tests::scalar_box_reduces_to_unit_d0`: lex [1..6] has
        // Mandelstam diagonals s = s02 = 2, t = s13 = 5.
        assert_terms(
            &crate::reduce::reduce(&fam).unwrap(),
            &[(
                Atom::num(1),
                crate::masters::MasterIntegral::Box {
                    p1_sq: Atom::num(1),
                    p2_sq: Atom::num(4),
                    p3_sq: Atom::num(6),
                    p4_sq: Atom::num(3),
                    s: Atom::num(2),
                    t: Atom::num(5),
                    m1_sq: Atom::Zero,
                    m2_sq: Atom::Zero,
                    m3_sq: Atom::Zero,
                    m4_sq: Atom::Zero,
                },
            )],
        );
    }

    #[test]
    fn box_numerator_through_the_bridge_has_the_exact_d0_coefficient() {
        crate::ensure_symbolica_license();
        // Masses chosen so the D0 coefficient below is non-degenerate.
        let masses = [1i64, 5, 3, 4];
        let edges = chain_edges(&masses);
        let fam = family_on_gram(&(&kk(1) * &pp(0, 1)), &edges, &box_gram());
        // gammaloop routes `k, k-P(0), ...`, so `q1^reducer = -P(0)` and the numerator
        // `k.P(0)` is `-dot(k, q1)` in the reducer's basis.
        assert_eq!(fam.numerator, -dot(&Atom::var(S.k), &q(1)));

        // `box_topo`'s RSP rule is `k.q1 = (D2 - D1 - m1 + m2 - p1)/2`, so
        //   int -(k.q1) / (D1 D2 D3 D4)
        //     = (1/2)[int 1/(D2 D3 D4) - int 1/(D1 D3 D4)] - (m2 - m1 - p1)/2 * D0.
        // With p1 = inv(0) = 1 (lexicographic!), m1 = 1, m2 = 5 the D0 coefficient is
        // -(5 - 1 - 1)/2 = -3/2. Reading the invariants in physics order instead would put
        // p4 = 3 or s = 2 in that slot and give -1/2 or -1.
        let r = crate::reduce::reduce(&fam).unwrap();
        let d0 = crate::masters::MasterIntegral::Box {
            p1_sq: Atom::num(1),
            p2_sq: Atom::num(4),
            p3_sq: Atom::num(6),
            p4_sq: Atom::num(3),
            s: Atom::num(2),
            t: Atom::num(5),
            m1_sq: Atom::num(masses[0]),
            m2_sq: Atom::num(masses[1]),
            m3_sq: Atom::num(masses[2]),
            m4_sq: Atom::num(masses[3]),
        };
        let got = r.terms.iter().find(|(_, m)| *m == d0).unwrap_or_else(|| {
            panic!(
                "no D0 with the expected arguments; got {:?}",
                r.terms.iter().map(|(_, m)| m).collect::<Vec<_>>()
            )
        });
        assert_eq!(got.0.expand(), Atom::num(-3) / Atom::num(2));
        // and the two triangles the (D2 - D1) piece pinches to, with coefficient +/- 1/2
        for (skip, sign) in [(0usize, 1i64), (1usize, -1i64)] {
            let keep: Vec<usize> = (0..4).filter(|&i| i != skip).collect();
            let tri = r
                .terms
                .iter()
                .filter_map(|(c, m)| match m {
                    crate::masters::MasterIntegral::Triangle {
                        m1_sq,
                        m2_sq,
                        m3_sq,
                        ..
                    } if [m1_sq, m2_sq, m3_sq]
                        == [
                            &Atom::num(masses[keep[0]]),
                            &Atom::num(masses[keep[1]]),
                            &Atom::num(masses[keep[2]]),
                        ] =>
                    {
                        Some(c)
                    }
                    _ => None,
                })
                .next()
                .unwrap_or_else(|| panic!("no triangle pinching propagator {}", skip + 1));
            assert_eq!(tri.expand(), Atom::num(sign) / Atom::num(2));
        }
    }

    #[test]
    fn scalar_pentagon_through_the_bridge_reproduces_the_exact_rationals() {
        crate::ensure_symbolica_license();
        // The pentagon is the only topology that exercises `q4` end to end, and the only one
        // whose invariants are consumed via `base_gram_from_pairwise`'s `idx(i,j)`.
        let edges = chain_edges(&[1, 2, 3, 4, 5]);
        let fam = family_on_gram(&Atom::num(1), &edges, &pentagon_gram());
        assert_eq!(
            fam.kinematics.invariants,
            [3, 5, 7, 9, 4, 6, 8, 5, 7, 6]
                .iter()
                .map(|&x| Atom::num(x))
                .collect::<Vec<_>>(),
            "the bridge must reproduce the pentagon invariant list of \
             `reduce::tests::scalar_pentagon_reduces_to_five_boxes`"
        );
        let r = crate::reduce::reduce(&fam).unwrap();
        assert_eq!(r.terms.len(), 5);
        // van Neerven-Vermaseren coefficients, identical to the hand-built family.
        let want_c = [
            Atom::num(-1088) / Atom::num(639),
            Atom::num(-212) / Atom::num(639),
            Atom::num(-44) / Atom::num(213),
            Atom::num(-191) / Atom::num(639),
            Atom::num(-341) / Atom::num(639),
        ];
        for ((coeff, _), want) in r.terms.iter().zip(&want_c) {
            assert_eq!(coeff, want);
        }
        assert_eq!(
            r.terms[4].1,
            crate::masters::MasterIntegral::Box {
                p1_sq: Atom::num(3),
                p2_sq: Atom::num(4),
                p3_sq: Atom::num(5),
                p4_sq: Atom::num(7),
                s: Atom::num(5),
                t: Atom::num(6),
                m1_sq: Atom::num(1),
                m2_sq: Atom::num(2),
                m3_sq: Atom::num(3),
                m4_sq: Atom::num(4),
            }
        );
    }

    #[test]
    fn pentagon_numerator_through_the_bridge_uses_the_q4_chain_direction() {
        crate::ensure_symbolica_license();
        // `dot(k, q4)` only exists at all for a pentagon; it is also the direction the
        // `MAX_MOMENTUM_ID` mapping and `square_external_momentum` have to reach.
        let edges = chain_edges(&[1, 2, 3, 4, 5]);
        let fam = family_on_gram(&(&kk(1) * &pp(3, 1)), &edges, &pentagon_gram());
        assert_eq!(fam.numerator, -dot(&Atom::var(S.k), &q(4)));
        let r = crate::reduce::reduce(&fam).unwrap();

        // Pin the *value*, from the RSP rule rather than from whatever the code printed.
        // For an N-gon, `k.q4 = (D5 - D4 - r5^2 + r4^2 + m5 - m4)/2`, so with the bridge's
        // numerator `-k.q4`,
        //
        //   int -(k.q4)/(D1..D5) = -(1/2)[ I(pinch 5) - I(pinch 4) ] - lambda * I(pentagon),
        //   lambda = (m5 - m4 - r5^2 + r4^2)/2 = (5 - 4 - 9 + 7)/2 = -1/2,
        //
        // reading `r5^2 = inv(1,5) = 9` and `r4^2 = inv(1,4) = 7` off the lexicographic
        // invariant list `[3,5,7,9,4,6,8,5,7,6]` pinned in the test above. Substituting the
        // pentagon's own five vNV coefficients `c_i` (also pinned above) therefore gives
        // `[c1/2, c2/2, c3/2, 1/2 + c4/2, -1/2 + c5/2]` -- an identity between two
        // independently asserted reductions, not a transcription of the output.
        let half = |n: i64, d: i64| Atom::num(n) / Atom::num(d);
        let scalar_c = [
            half(-1088, 639),
            half(-212, 639),
            half(-44, 213),
            half(-191, 639),
            half(-341, 639),
        ];
        let two = Atom::num(2);
        let want: Vec<Atom> = scalar_c
            .iter()
            .enumerate()
            .map(|(i, c)| match i {
                3 => (c / &two + half(1, 2)).expand(),
                4 => (c / &two - half(1, 2)).expand(),
                _ => (c / &two).expand(),
            })
            .collect();
        assert_eq!(r.terms.len(), 5);
        let got: Vec<Atom> = r.terms.iter().map(|(c, _)| c.expand()).collect();
        assert_eq!(got, want, "rank-1 pentagon violates the q4 RSP identity");

        // The identity above is only meaningful if terms 4 and 5 really are the boxes that
        // pinch propagators 4 and 5 -- i.e. the ones keeping masses (1,2,3,5) and (1,2,3,4).
        let masses_of = |i: usize| match &r.terms[i].1 {
            crate::masters::MasterIntegral::Box {
                m1_sq,
                m2_sq,
                m3_sq,
                m4_sq,
                ..
            } => vec![m1_sq.clone(), m2_sq.clone(), m3_sq.clone(), m4_sq.clone()],
            other => panic!("expected a box, got {other:?}"),
        };
        let nums = |v: &[i64]| v.iter().map(|&n| Atom::num(n)).collect::<Vec<_>>();
        assert_eq!(
            masses_of(3),
            nums(&[1, 2, 3, 5]),
            "term 4 must pinch line 4"
        );
        assert_eq!(
            masses_of(4),
            nums(&[1, 2, 3, 4]),
            "term 5 must pinch line 5"
        );
    }

    // ---------------------------------------------------------------------------------------
    // Guards: the bridge must fail loudly rather than silently degrade.
    // ---------------------------------------------------------------------------------------

    fn translation_error(numerator: &Atom, edges: &[LoopEdge], h: &TensorHeads) -> String {
        match family_from_tensor_numerator(numerator, edges, h) {
            Ok(f) => panic!(
                "expected a translation failure, got numerator `{}` invariants {:?}",
                f.numerator,
                f.kinematics
                    .invariants
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
            ),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn reorders_a_triangle_handed_over_out_of_loop_order() {
        crate::ensure_symbolica_license();
        // `k, k - P(0) - P(1), k - P(0)` is a perfectly ordinary triangle listed out of loop
        // order: propagator 2 differs from propagator 1 by two externals only because
        // propagator 3 belongs between them. gammaloop hands the propagators over in
        // `iter_edges()` order, which is NOT loop order, so demanding the chain outright
        // would reject real graphs whose scalar reduction is routing-independent and was
        // correct before. Reorder instead -- permuting denominators only relabels the family.
        let edges = vec![
            LoopEdge {
                lmb_rep: kk(0),
                mass_sq: Atom::num(1),
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(0, 0) - &pp(1, 0),
                mass_sq: Atom::num(2),
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(0, 0),
                mass_sq: Atom::num(3),
            },
        ];
        let fam = family_from_tensor_numerator(&(&kk(1) * &pp(0, 1)), &edges, &heads()).unwrap();
        // Chain order is [0, 2, 1], so the masses follow the propagators into that order.
        assert_eq!(
            fam.propagators
                .iter()
                .map(|p| p.mass_sq.clone())
                .collect::<Vec<_>>(),
            vec![Atom::num(1), Atom::num(3), Atom::num(2)]
        );
        // Reordered offsets are `[0, -q1, -q1-q2]`, so `q1^reducer = -P(0)` as usual.
        assert_eq!(fam.numerator, -dot(&Atom::var(S.k), &q(1)));
    }

    #[test]
    fn reorders_a_bubble_whose_zero_offset_propagator_is_not_first() {
        crate::ensure_symbolica_license();
        // The LMB basis edge (the one with zero offset) need not come first in
        // `iter_edges()` order. It is the reducer's `r_1 = 0`, so it has to be moved there,
        // taking its mass with it.
        let edges = vec![
            LoopEdge {
                lmb_rep: &kk(0) - &pp(0, 0),
                mass_sq: Atom::num(5),
            },
            LoopEdge {
                lmb_rep: kk(0),
                mass_sq: Atom::num(7),
            },
        ];
        let fam = family_from_tensor_numerator(&(&kk(1) * &pp(0, 1)), &edges, &heads()).unwrap();
        assert_eq!(
            fam.propagators
                .iter()
                .map(|p| p.mass_sq.clone())
                .collect::<Vec<_>>(),
            vec![Atom::num(7), Atom::num(5)]
        );
        assert_eq!(fam.numerator, -dot(&Atom::var(S.k), &q(1)));
    }

    #[test]
    fn rejects_a_routing_no_reordering_can_repair() {
        crate::ensure_symbolica_license();
        // Genuinely not a one-loop chain: no walk from the zero-offset propagator reaches the
        // others one distinct external at a time.
        let edges = vec![
            LoopEdge {
                lmb_rep: kk(0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(0, 0) - &pp(1, 0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(0, 0) - &pp(1, 0) - &pp(2, 0) - &pp(3, 0),
                mass_sq: Atom::Zero,
            },
        ];
        let e = translation_error(&Atom::num(1), &edges, &heads());
        assert!(
            e.contains("do not form the reducer's chain"),
            "unexpected error: {e}"
        );
    }

    #[test]
    fn rejects_a_family_with_no_zero_offset_propagator() {
        crate::ensure_symbolica_license();
        // Without an `r_i = 0` the chain cannot start without shifting the loop momentum,
        // which would also shift the numerator. Fail loudly rather than guess.
        let edges = vec![
            LoopEdge {
                lmb_rep: &kk(0) - &pp(0, 0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(1, 0),
                mass_sq: Atom::Zero,
            },
        ];
        let e = translation_error(&Atom::num(1), &edges, &heads());
        assert!(e.contains("zero external offset"), "unexpected error: {e}");
    }

    #[test]
    fn rejects_a_non_unit_external_coefficient() {
        crate::ensure_symbolica_license();
        let edges = vec![
            LoopEdge {
                lmb_rep: kk(0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &kk(0) - &(Atom::num(2) * pp(0, 0)),
                mass_sq: Atom::Zero,
            },
        ];
        let e = translation_error(&Atom::num(1), &edges, &heads());
        assert!(
            e.contains("signed sum of distinct"),
            "unexpected error: {e}"
        );
    }

    #[test]
    fn rejects_a_numerator_direction_outside_the_chain() {
        crate::ensure_symbolica_license();
        // A triangle has only two chain directions, but the reducer hard-codes `n_ext = 3`
        // and would project `dot(k, q3)` against a *fabricated* Gram whose `q3.q3` is 0.
        let edges = chain_edges(&[0, 0, 0]);
        let e = translation_error(&(&kk(1) * &pp(2, 1)), &edges, &heads());
        assert!(e.contains("not one of the"), "unexpected error: {e}");
    }

    #[test]
    fn rejects_an_external_id_at_or_beyond_max_momentum_id() {
        crate::ensure_symbolica_license();
        // `P(8)` is outside the 0..MAX_MOMENTUM_ID window, so `numerator_to_dot_form` leaves
        // `K(0,i) P(8,i)` untranslated. Absorbing that into a coefficient would be silently
        // wrong, since it still depends on the loop momentum.
        let edges = chain_edges(&[0, 0, 0]);
        let e = translation_error(&(&kk(1) * &pp(MAX_MOMENTUM_ID, 1)), &edges, &heads());
        assert!(e.contains("untranslated"), "unexpected error: {e}");
    }

    #[test]
    fn rejects_a_wrong_metric_head() {
        crate::ensure_symbolica_license();
        // If a spenso rename made `heads.metric` stale, `g(K, K)` would survive untouched and
        // `numerator_to_monos` would treat the whole thing as a constant.
        let stale = TensorHeads {
            metric: symbol!("not_the_metric"),
            ..heads()
        };
        let bare = function!(
            symbol!("K"),
            Atom::num(0),
            function!(symbol!("mink"), Atom::num(4))
        );
        let numerator = function!(symbol!("g"), bare.clone(), bare);
        let e = translation_error(&numerator, &chain_edges(&[0, 0, 0]), &stale);
        assert!(e.contains("untranslated"), "unexpected error: {e}");
    }

    #[test]
    fn rejects_an_uncontracted_loop_momentum_index() {
        crate::ensure_symbolica_license();
        // A polarization vector leaves `dot(k, eps)`, which is not in the reducer's Gram
        // basis -- the caller has to project the polarizations out before reducing -- so
        // the loop momentum tensor survives and the guard must reject it.
        let numerator = &kk(1) * &function!(symbol!("eps"), Atom::num(0), mink4(1));
        let e = translation_error(&numerator, &chain_edges(&[0, 0, 0]), &heads());
        assert!(e.contains("untranslated"), "unexpected error: {e}");
    }

    #[test]
    fn an_external_only_tensor_survives_as_a_constant_coefficient() {
        crate::ensure_symbolica_license();
        // The guard is deliberately about the LOOP momentum only. A leftover tensor built
        // purely from externals and polarizations is genuinely k-independent, so absorbing it
        // into the coefficient is correct and must not be rejected.
        let opaque = &function!(symbol!("eps"), Atom::num(0), mink4(7))
            * &function!(symbol!("eps"), Atom::num(1), mink4(7));
        let numerator = &opaque * &(&kk(1) * &pp(0, 1));
        let fam =
            family_from_tensor_numerator(&numerator, &chain_edges(&[0, 0, 0]), &heads()).unwrap();
        assert_eq!(
            fam.numerator.expand(),
            (-&opaque * dot(&Atom::var(S.k), &q(1))).expand()
        );
        assert!(!crate::reduce::reduce(&fam).unwrap().terms.is_empty());
    }

    #[test]
    fn a_tadpole_may_still_carry_external_isps() {
        crate::ensure_symbolica_license();
        // A tadpole has no chain direction at all, but the reducer handles `dot(k, q_a)` there
        // with a fully *symbolic* Gram in the same labels the bridge emits, so it stays legal
        // (up to the hard-coded n_ext = 3).
        let edges = vec![LoopEdge {
            lmb_rep: kk(0),
            mass_sq: Atom::num(7),
        }];
        let fam = family_from_tensor_numerator(&(&kk(1) * &pp(0, 1)), &edges, &heads()).unwrap();
        assert_eq!(fam.numerator, dot(&Atom::var(S.k), &q(1)));
        assert!(fam.kinematics.invariants.is_empty());
        let e = translation_error(&(&kk(1) * &pp(3, 1)), &edges, &heads());
        assert!(e.contains("beyond the"), "unexpected error: {e}");
    }

    #[test]
    fn relabels_a_permuted_external_assignment() {
        crate::ensure_symbolica_license();
        // The LMB keeps one external as a dependent "dummy carrier", so the externals that
        // reach the bridge need not be `P(0), P(1)` in order. Here the chain is
        // `r = [0, -q2, -q2-q1]`, i.e. slot 1 is q2 and slot 2 is q1 -- a permutation the
        // relabelling has to perform simultaneously to avoid clobbering.
        let edges = vec![
            LoopEdge {
                lmb_rep: kk(0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(1, 0),
                mass_sq: Atom::Zero,
            },
            LoopEdge {
                lmb_rep: &kk(0) - &pp(1, 0) - &pp(0, 0),
                mass_sq: Atom::Zero,
            },
        ];
        let numerator = &(&kk(1) * &pp(1, 1)) + &(Atom::num(3) * (&kk(2) * &pp(0, 2)));
        let fam = family_from_tensor_numerator(&numerator, &edges, &heads()).unwrap();
        // P(1) -> slot 1 with eps = -1, P(0) -> slot 2 with eps = -1.
        assert_eq!(
            fam.numerator.expand(),
            (-dot(&Atom::var(S.k), &q(1)) - Atom::num(3) * dot(&Atom::var(S.k), &q(2))).expand()
        );
    }
}

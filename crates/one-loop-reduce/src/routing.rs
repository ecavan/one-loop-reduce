//! Momentum routing: putting propagator offsets into the chain `r_i = q1 + ... + q_{i-1}` the
//! reducer assumes, and reading the numerator's labels and the invariants off it.
//!
//! Integer combinatorics over the externals `q1 .. q{MAX_MOMENTUM_ID}` -- no tensor heads and
//! no model. Any caller that can write its offsets as signed sums of the [`q`] symbols drives
//! the reducer through here; [`crate::bridge`] is one such caller.

use symbolica::atom::{Atom, AtomCore, Symbol};
use symbolica::{function, symbol};

use crate::symbols::S;

/// How many external momentum ids the routing recognizes (`q1 .. q{MAX_MOMENTUM_ID}`).
pub const MAX_MOMENTUM_ID: i64 = 8;

fn q_sym(j: usize) -> Symbol {
    symbol!(format!("oneloopreduce::q{}", j + 1))
}

/// The 0-based `j`-th external momentum, `q{j+1}`. Both the caller's labels and the reducer's
/// chain slots are drawn from this one set -- [`relabel_numerator`] maps between them.
pub fn q(j: usize) -> Atom {
    Atom::var(q_sym(j))
}

/// Every external the routing recognizes.
pub(crate) fn external_syms() -> Vec<Atom> {
    (0..MAX_MOMENTUM_ID as usize).map(q).collect()
}

fn dot_kq(qa: &Atom) -> Atom {
    function!(S.dot, Atom::var(S.k), qa.clone())
}

/// `0`, `+1` or `-1`; anything else (including a non-numeric coefficient) is rejected, because
/// a one-loop propagator offset is always a signed sum of *distinct* external momenta.
fn unit_int(a: &Atom) -> Option<i32> {
    if *a == Atom::Zero {
        Some(0)
    } else if *a == Atom::num(1) {
        Some(1)
    } else if *a == Atom::num(-1) {
        Some(-1)
    } else {
        None
    }
}

/// Does the numerator actually depend on `dot(k, q_{j+1})`?
fn depends_on_dot_kq(num: &Atom, j: usize) -> bool {
    let probe = symbol!("oneloopreduce::routing_probe");
    num.replace(dot_kq(&q(j)).to_pattern())
        .with(Atom::var(probe))
        .derivative(probe)
        != Atom::Zero
}

/// A propagator offset as integer coefficients over the externals `q1..q{MAX_MOMENTUM_ID}`.
pub fn offset_dirs(offset: &Atom) -> Result<Vec<i32>, String> {
    let mut dirs = vec![0i32; MAX_MOMENTUM_ID as usize];
    let mut residue = offset.clone();
    for (j, slot) in dirs.iter_mut().enumerate() {
        let qs = q_sym(j);
        let c = offset.derivative(qs);
        let c = unit_int(&c).ok_or_else(|| {
            format!(
                "propagator offset `{offset}` is not a signed sum of distinct external \
                 momenta: the coefficient of q{} is `{c}`",
                j + 1
            )
        })?;
        *slot = c;
        residue -= Atom::num(i64::from(c)) * Atom::var(qs);
    }
    let residue = residue.expand();
    if residue != Atom::Zero {
        return Err(format!(
            "propagator offset `{offset}` has an untranslatable remainder `{residue}`"
        ));
    }
    Ok(dirs)
}

/// Validate that the edge offsets form the chain the reducer assumes, and return, for each
/// reducer slot `a`, the caller's external it corresponds to and its sign.
///
/// The reducer hard-codes its propagator offsets as `r_i = q_1 + ... + q_{i-1}` (see
/// `reduce.rs`'s `triangle_topo`/`box_topo`/`ngon_numerator`, and the RSP rule
/// `k.w1 = (D2 - D1 - m1 + m2 - s1)/2` that pins the *plus* sign). A caller's routing -- a
/// gammaloop LMB, say -- matches that only up to
///
/// * **which** external sits in which slot -- the LMB keeps one external as a dependent
///   "dummy carrier", so the externals that appear need not be `P(0), P(1), ...`; and
/// * a **per-slot sign** -- gammaloop routes a triangle as `k, k-P(0), k-P(0)-P(1)`
///   (verified in `gammalooprs::graph::parse::tests::test_load`), i.e. `r_a - r_{a-1} = -q_a`.
///
/// Both are basis-independent for the *invariants* `(r_i - r_j)^2` and for the Cayley matrix,
/// which is why the scalar benchmarks never saw them -- but `dot(k, q_a)` in a numerator is
/// not basis-independent, so the numerator must be relabelled with these slots.
///
/// This runs on propagators already put in chain order by [`chain_order`].
pub fn chain_slots(offsets: &[Vec<i32>]) -> Result<Vec<(usize, i32)>, String> {
    if offsets[0].iter().any(|&c| c != 0) {
        return Err(format!(
            "the first loop propagator must carry no external offset (the reducer's r_1 = 0), \
             got coefficients {:?}",
            offsets[0]
        ));
    }
    let mut slots: Vec<(usize, i32)> = Vec::new();
    for a in 1..offsets.len() {
        let w: Vec<i32> = offsets[a]
            .iter()
            .zip(&offsets[a - 1])
            .map(|(x, y)| x - y)
            .collect();
        let nz: Vec<usize> = (0..w.len()).filter(|&j| w[j] != 0).collect();
        let [j] = nz[..] else {
            return Err(format!(
                "propagator {} does not differ from propagator {} by a single external \
                 momentum (the reducer's chain r_i = q1 + ... + q_{{i-1}}); difference is over \
                 {} externals",
                a + 1,
                a,
                nz.len()
            ));
        };
        if w[j].abs() != 1 {
            return Err(format!(
                "propagator {} differs from propagator {} by {}*q{}, not a single external",
                a + 1,
                a,
                w[j],
                j + 1
            ));
        }
        if slots.iter().any(|&(used, _)| used == j) {
            return Err(format!(
                "external q{} appears in more than one chain slot; the reducer's chain \
                 directions must be independent",
                j + 1
            ));
        }
        slots.push((j, w[j]));
    }
    Ok(slots)
}

/// `r_j - r_i` as a single signed external, or `None` if it is not one.
fn single_step(dirs: &[Vec<i32>], i: usize, j: usize) -> Option<(usize, i32)> {
    let mut found = None;
    for (k, (to, from)) in dirs[j].iter().zip(&dirs[i]).enumerate() {
        let d = to - from;
        if d == 0 {
            continue;
        }
        if d.abs() != 1 || found.is_some() {
            return None;
        }
        found = Some((k, d));
    }
    found
}

/// Depth-first extension of a chain, neighbours in index order so the result is deterministic.
fn extend_chain(
    dirs: &[Vec<i32>],
    order: &mut Vec<usize>,
    visited: &mut [bool],
    used_ext: &mut Vec<usize>,
) -> bool {
    if order.len() == dirs.len() {
        return true;
    }
    let last = *order.last().expect("the chain always starts somewhere");
    for j in 0..dirs.len() {
        if visited[j] {
            continue;
        }
        let Some((k, _)) = single_step(dirs, last, j) else {
            continue;
        };
        if used_ext.contains(&k) {
            continue;
        }
        visited[j] = true;
        order.push(j);
        used_ext.push(k);
        if extend_chain(dirs, order, visited, used_ext) {
            return true;
        }
        used_ext.pop();
        order.pop();
        visited[j] = false;
    }
    false
}

/// Put the loop propagators into the order the reducer's chain `r_i = q1 + ... + q_{i-1}`
/// assumes, returning the permutation to apply to the edges and their offsets.
///
/// A caller need not hand the propagators over in loop order, and its offsets need not be the
/// reducer's chain even up to sign. A real one-loop box comes out of gammaloop's LMB as
///
/// ```text
/// r = [ -q2 + q3,  q3,  0,  -q1 - q2 + q3 ]
/// ```
///
/// (`gammalooprs::reduce_bridge::tests`), because exactly one propagator is the LMB basis edge
/// (offset `0`) and one leg of the polygon is the *dependent* external, which momentum
/// conservation expands into a sum of the others. Read in `iter_edges()` order that is neither
/// a chain nor a permutation of one -- but walking the polygon from the zero-offset edge,
/// `[0, q3, -q2 + q3, -q1 - q2 + q3]`, is exactly the reducer's chain with the dependent leg as
/// the wrap-around step, which the chain never needs.
///
/// Permuting propagators only relabels the family -- the integral is symmetric in its
/// denominators, and the invariants and masses are permuted with them -- so this is a faithful
/// translation. Rejecting these instead (as demanding the chain outright does) would send every
/// real box and pentagon down the `unsupported` path, even though their *scalar* reduction is
/// routing-independent and was correct before.
pub fn chain_order(dirs: &[Vec<i32>]) -> Result<Vec<usize>, String> {
    let n = dirs.len();
    let start = (0..n)
        .find(|&i| dirs[i].iter().all(|&c| c == 0))
        .ok_or_else(|| {
            format!(
                "no loop propagator carries a zero external offset, so the reducer's r_1 = 0 \
                 cannot be reached without shifting the loop momentum; offsets are {dirs:?}"
            )
        })?;
    let mut order = vec![start];
    let mut visited = vec![false; n];
    visited[start] = true;
    let mut used_ext = Vec::new();
    if !extend_chain(dirs, &mut order, &mut visited, &mut used_ext) {
        return Err(format!(
            "the loop propagators do not form the reducer's chain r_i = q1 + ... + q_{{i-1}} \
             under any ordering: no walk from the zero-offset propagator visits all {n} of \
             them one distinct external at a time; offsets are {dirs:?}"
        ));
    }
    Ok(order)
}

/// Rewrite `dot(k, q_{j_a}) -> eps_a * dot(k, q_a)` so the numerator speaks the reducer's
/// chain basis. Done in two passes through a scratch namespace so a permutation of slots
/// (e.g. q2 -> q1 and q1 -> q2) cannot collide.
pub fn relabel_numerator(num: &Atom, slots: &[(usize, i32)]) -> Atom {
    let tmp = |a: usize| Atom::var(symbol!(format!("oneloopreduce::routing_tmp_q{}", a + 1)));
    let mut out = num.clone();
    for (a, &(j, eps)) in slots.iter().enumerate() {
        let to = Atom::num(i64::from(eps)) * dot_kq(&tmp(a));
        out = out.replace(dot_kq(&q(j)).to_pattern()).with(to);
    }
    for a in 0..slots.len() {
        out = out
            .replace(dot_kq(&tmp(a)).to_pattern())
            .with(dot_kq(&q(a)));
    }
    out
}

/// The reducer hard-codes `n_ext = 3` at the tadpole/bubble/triangle/box entry points, so a
/// tadpole may legitimately carry `dot(k, q1..q3)` as irreducible scalar products (there it
/// uses a fully *symbolic* Gram, in the same labels this module emits, so no relabelling is
/// needed or wanted).
const TADPOLE_N_EXT: usize = 3;

/// Reject any `dot(k, q_j)` the reducer cannot faithfully interpret. For `n >= 2` the only
/// legitimate directions are the `n-1` chain directions: anything else would be projected
/// against a *fabricated* Gram (`base_gram_box` pads the unused slots with zeros) and
/// silently absorbed into a coefficient.
pub fn check_numerator_directions(
    num: &Atom,
    slots: &[(usize, i32)],
    n: usize,
) -> Result<(), String> {
    for j in 0..MAX_MOMENTUM_ID as usize {
        if !depends_on_dot_kq(num, j) {
            continue;
        }
        if n == 1 {
            if j < TADPOLE_N_EXT {
                continue;
            }
            return Err(format!(
                "tadpole numerator contracts the loop momentum with q{}, beyond the \
                 reducer's n_ext={TADPOLE_N_EXT} irreducible-scalar-product basis",
                j + 1
            ));
        }
        if !slots.iter().any(|&(slot, _)| slot == j) {
            return Err(format!(
                "numerator contracts the loop momentum with q{}, which is not one of the \
                 {} chain directions of this {n}-propagator topology; the reducer would \
                 project it against a fabricated Gram",
                j + 1,
                slots.len()
            ));
        }
    }
    Ok(())
}

fn square_external_momentum(momentum: &Atom) -> Atom {
    let qs = external_syms();
    let mut out = (momentum * momentum).expand();
    // `q_a^2 -> dot(q_a, q_a)` (squares) then `q_a*q_b -> dot(q_a, q_b)`
    for qa in &qs {
        out = out
            .replace((qa * qa).to_pattern())
            .with(function!(S.dot, qa.clone(), qa.clone()));
    }
    for a in 0..qs.len() {
        for b in (a + 1)..qs.len() {
            out = out.replace((&qs[a] * &qs[b]).to_pattern()).with(function!(
                S.dot,
                qs[a].clone(),
                qs[b].clone()
            ));
        }
    }
    out
}

/// The `C(n,2)` pairwise invariants `(r_i - r_j)^2`
pub fn invariants_from_offsets(offsets: &[Atom]) -> Vec<Atom> {
    let mut invariants = Vec::new();
    for i in 0..offsets.len() {
        for j in (i + 1)..offsets.len() {
            invariants.push(square_external_momentum(&(&offsets[i] - &offsets[j])));
        }
    }
    invariants
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An offset as coefficients over `q1..q{MAX_MOMENTUM_ID}`, from its non-zero entries.
    fn dirs(nz: &[(usize, i32)]) -> Vec<i32> {
        let mut v = vec![0; MAX_MOMENTUM_ID as usize];
        for &(j, c) in nz {
            v[j] = c;
        }
        v
    }

    #[test]
    fn chain_order_walks_the_real_box_lmb_routing() {
        // The box routing in `chain_order`'s doc comment, `r = [-q2+q3, q3, 0, -q1-q2+q3]`:
        // read in edge order it is not a chain, but walking the polygon from the zero-offset
        // edge is. Until this module existed the case could only be exercised by building a
        // full tensor numerator, so nothing here pinned it.
        let offs = [
            dirs(&[(1, -1), (2, 1)]),
            dirs(&[(2, 1)]),
            dirs(&[]),
            dirs(&[(0, -1), (1, -1), (2, 1)]),
        ];
        let order = chain_order(&offs).unwrap();
        assert_eq!(order, vec![2, 1, 0, 3]);
        let chain: Vec<Vec<i32>> = order.iter().map(|&i| offs[i].clone()).collect();
        assert_eq!(chain_slots(&chain).unwrap(), vec![(2, 1), (1, -1), (0, -1)]);
    }

    #[test]
    fn chain_slots_rejects_what_chain_order_cannot_produce() {
        // `chain_order` already guarantees a zero first offset and one *distinct* external
        // per step, so these two branches are unreachable through it -- they are the
        // contract for a caller that assembles the chain itself.
        let e = chain_slots(&[dirs(&[(0, 1)]), dirs(&[(0, 1), (1, 1)])]).unwrap_err();
        assert!(e.contains("must carry no external offset"), "{e}");
        let e = chain_slots(&[dirs(&[]), dirs(&[(0, 1)]), dirs(&[])]).unwrap_err();
        assert!(e.contains("more than one chain slot"), "{e}");
    }

    #[test]
    fn offset_dirs_reads_signed_units_and_rejects_the_rest() {
        crate::ensure_symbolica_license();
        assert_eq!(
            offset_dirs(&(&q(0) - &q(2))).unwrap(),
            dirs(&[(0, 1), (2, -1)])
        );
        let e = offset_dirs(&(Atom::num(2) * q(0))).unwrap_err();
        assert!(e.contains("signed sum of distinct"), "{e}");
    }

    #[test]
    fn relabel_numerator_permutes_two_slots_without_clobbering() {
        crate::ensure_symbolica_license();
        // Slot 1 is q2 and slot 2 is q1, both with eps = -1: a swap, which a single pass
        // would collapse onto one label.
        let num = &dot_kq(&q(1)) + &(Atom::num(3) * dot_kq(&q(0)));
        let got = relabel_numerator(&num, &[(1, -1), (0, -1)]);
        let want = (-dot_kq(&q(0)) - Atom::num(3) * dot_kq(&q(1))).expand();
        assert_eq!(got.expand(), want);
    }

    #[test]
    fn bubble_invariants_from_offsets() {
        crate::ensure_symbolica_license();
        let offsets = vec![Atom::Zero, Atom::num(-1) * q(0)];
        assert_eq!(
            invariants_from_offsets(&offsets),
            vec![function!(S.dot, q(0), q(0))]
        );
    }

    #[test]
    fn invariants_handle_high_externals() {
        crate::ensure_symbolica_license();
        // A pentagon edge offset of q4 must square to `dot(q4, q4)`, i.e.
        // `square_external_momentum` has to reach past q1..q3.
        let offsets = vec![Atom::Zero, q(3)];
        assert_eq!(
            invariants_from_offsets(&offsets),
            vec![function!(S.dot, q(3), q(3))]
        );
    }
}

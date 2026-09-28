//! Exact IBP reduction of a one-loop family at fixed kinematics, by Gaussian
//! elimination over its integration-by-parts identities (Laporta).
//!
//! Slower than the closed-form recursions, so it is only the fallback for
//! kinematics where those cannot take the on-shell limit. It needs no Gram
//! inverse and no regulator: at a degenerate point it finds the smaller set of
//! masters by itself (a massless triangle with two on-shell legs becomes a
//! bubble times `1/(d-4)`), and integrals without a scale reduce to zero.

use std::collections::{BTreeMap, HashMap};

use symbolica::atom::{Atom, AtomCore};
use symbolica::domains::integer::Z;
use symbolica::domains::rational::Q;

use super::{Rp, add_scaled, den_symbol, emit_master, extract_monomials, loop_dots_to_vars};
use super::{delete_row_col, modified_cayley, reduce_cayley};
use crate::family::IntegralFamily;
use crate::masters::MasterIntegral;
use crate::symbols::S;

/// An integral `I(a) = ∫ 1/(D_1^a_1 ... D_N^a_N)`; `a_i <= 0` is a numerator.
type Index = Vec<i32>;

/// Integrals sorted by complexity: lines, then dots, then numerator powers.
/// A master is the simplest integral of its sector: all lines to the power 1.
type Key = (usize, i32, i32, Index);

fn key(a: &[i32]) -> Key {
    let lines = a.iter().filter(|&&e| e > 0).count();
    let dots = a.iter().filter(|&&e| e > 0).map(|&e| e - 1).sum();
    let numerator = a.iter().filter(|&&e| e < 0).map(|&e| -e).sum();
    (lines, dots, numerator, a.to_vec())
}

fn rp(a: &Atom) -> Rp {
    a.to_rational_polynomial(&Q, &Z, None)
}

/// A linear combination of integrals.
type Row = BTreeMap<Key, Rp>;

fn add_to(row: &mut Row, a: Index, c: Rp) {
    // Without a positive index there is no scale: the integral is zero.
    if c.is_zero() || a.iter().all(|&e| e <= 0) {
        return;
    }
    let k = key(&a);
    let sum = match row.remove(&k) {
        Some(old) => &old + &c,
        None => c,
    };
    if !sum.is_zero() {
        row.insert(k, sum);
    }
}

/// The IBP identity from `∂/∂l · (l + r_k)` on `I(a)`:
/// `(d - Σa) I(a) - Σ_i a_i I(a + e_i - e_k) - Σ_i a_i Y_ki I(a + e_i) = 0`.
fn identity(a: &[i32], k: usize, y: &[Vec<Rp>]) -> Row {
    let total: i32 = a.iter().sum();
    let mut row = Row::new();
    add_to(
        &mut row,
        a.to_vec(),
        rp(&(Atom::var(S.d) - Atom::num(i64::from(total)))),
    );
    for (i, &ai) in a.iter().enumerate().filter(|&(_, &ai)| ai != 0) {
        let ai = rp(&Atom::num(i64::from(ai)));
        let mut shifted = a.to_vec();
        shifted[i] += 1;
        add_to(&mut row, shifted.clone(), -(&ai * &y[k][i]));
        shifted[k] -= 1;
        add_to(&mut row, shifted, -ai);
    }
    row
}

/// Seeds: every sector, with up to `dots` extra powers and `num` numerator powers.
fn seeds(n: usize, dots: i32, num: i32) -> Vec<Index> {
    fn spread(slots: &[usize], total: i32, sign: i32, base: &mut Index, out: &mut Vec<Index>) {
        let Some((&first, rest)) = slots.split_first() else {
            if total == 0 {
                out.push(base.clone());
            }
            return;
        };
        for x in 0..=total {
            base[first] += sign * x;
            spread(rest, total - x, sign, base, out);
            base[first] -= sign * x;
        }
    }
    let mut out = Vec::new();
    for mask in 1..(1u32 << n) {
        let on = |i: usize| (mask & (1 << i)) != 0;
        let lines: Vec<usize> = (0..n).filter(|&i| on(i)).collect();
        let others: Vec<usize> = (0..n).filter(|&i| !on(i)).collect();
        for r in 0..=dots {
            for s in 0..=num {
                let mut base: Index = (0..n).map(|i| i32::from(on(i))).collect();
                let mut with_dots = Vec::new();
                spread(&lines, r, 1, &mut base, &mut with_dots);
                for mut b in with_dots {
                    spread(&others, s, -1, &mut b, &mut out);
                }
            }
        }
    }
    out.sort_by_key(|a| key(a));
    out
}

/// Substitute solved integrals until only unsolved ones remain.
fn reduce_row(mut row: Row, rules: &HashMap<Key, Row>) -> Row {
    while let Some(k) = row.keys().rev().find(|k| rules.contains_key(*k)).cloned() {
        let c = row.remove(&k).unwrap();
        for (j, cj) in &rules[&k] {
            add_to(&mut row, j.3.clone(), &c * cj);
        }
    }
    row
}

/// Reduce `target` (a combination of integrals) to masters, or `None` if the
/// system with these seeds leaves something that is not a master.
fn solve(target: &Row, n: usize, y: &[Vec<Rp>], dots: i32, num: i32) -> Option<Row> {
    let mut rules: HashMap<Key, Row> = HashMap::new();
    for a in seeds(n, dots, num) {
        for k in 0..n {
            let mut row = reduce_row(identity(&a, k, y), &rules);
            let Some((pivot, c)) = row.pop_last() else {
                continue;
            };
            let rhs = row.into_iter().map(|(j, cj)| (j, -(&cj / &c))).collect();
            rules.insert(pivot, rhs);
        }
    }
    let result = reduce_row(target.clone(), &rules);
    let is_master = |k: &Key| k.1 == 0 && k.2 == 0;
    result.keys().all(is_master).then_some(result)
}

/// Reduce `family` exactly, or `None` if it has no direction to rewrite its
/// numerator in (a tadpole) or the elimination does not close.
pub(super) fn reduce_exact(family: &IntegralFamily) -> Option<Vec<(Atom, MasterIntegral)>> {
    let n = family.propagators.len();
    let exps = &family.targets[0].propagator_exponents;
    if n < 2 {
        return None;
    }
    let masses: Vec<Atom> = family
        .propagators
        .iter()
        .map(|p| p.mass_sq.clone())
        .collect();
    let invariants = &family.kinematics.invariants;
    let y_atoms = modified_cayley(&masses, invariants);
    let y: Vec<Vec<Rp>> = y_atoms
        .iter()
        .map(|row| row.iter().map(rp).collect())
        .collect();

    // With the chain r_1 = 0, r_i = q_1 + ... + q_{i-1}: k.k = D_1 + m_1^2 and
    // 2 k.r_i = D_i - D_1 - r_i^2 + m_i^2 - m_1^2, where r_i^2 = (r_1 - r_i)^2.
    let den: Vec<Atom> = (0..n).map(|i| Atom::var(den_symbol(i))).collect();
    let k_dot_r = |i: usize| {
        if i == 0 {
            return Atom::Zero;
        }
        (&den[i] - &den[0] - &invariants[i - 1] + &masses[i] - &masses[0]) / Atom::num(2)
    };
    let (poly, vars) = loop_dots_to_vars(&family.numerator, n - 1);
    let mut numerator = poly
        .replace(Atom::var(vars[0]).to_pattern())
        .with(&den[0] + &masses[0]);
    // vars[j] is k.q_j = k.r_{j+1} - k.r_j, i.e. lines j and j - 1 (0-based).
    for (j, xq) in vars.iter().enumerate().skip(1) {
        let dot_q = k_dot_r(j) - k_dot_r(j - 1);
        numerator = numerator.replace(Atom::var(*xq).to_pattern()).with(dot_q);
    }
    let den_syms: Vec<_> = (0..n).map(den_symbol).collect();
    let terms = extract_monomials(&numerator.expand(), &den_syms);

    // Solve each integral of the numerator with unit weight; the external
    // coefficients (which may hold any k-free expression) multiply at the end.
    let integrals: Vec<Index> = terms
        .iter()
        .map(|(powers, _)| exps.iter().zip(powers).map(|(a, p)| a - p).collect())
        .collect();
    let dots = integrals.iter().map(|a| key(a).1).max().unwrap_or(0);
    let num = integrals.iter().map(|a| key(a).2).max().unwrap_or(0);
    let mut out = Vec::new();
    for (a, (_, coefficient)) in integrals.iter().zip(&terms) {
        let mut target = Row::new();
        add_to(&mut target, a.clone(), rp(&Atom::num(1)));
        let solved = (0..=2).find_map(|extra| solve(&target, n, &y, dots + extra, num + extra))?;
        for (master, c) in solved {
            let c = coefficient * c.to_expression();
            let lines: Vec<usize> = (0..n).filter(|&i| master.3[i] > 0).collect();
            let sub_masses: Vec<Atom> = lines.iter().map(|&i| masses[i].clone()).collect();
            let mut sub_y = y_atoms.clone();
            for i in (0..n).rev().filter(|i| !lines.contains(i)) {
                sub_y = delete_row_col(&sub_y, i, i);
            }
            let masters = match lines.len() {
                1 => vec![(
                    Atom::num(1),
                    MasterIntegral::Tadpole {
                        m_sq: sub_masses[0].clone(),
                    },
                )],
                2..=4 => emit_master(&sub_y, &sub_masses),
                m => reduce_cayley(&sub_y, &sub_masses, &vec![1; m]),
            };
            add_scaled(&mut out, &c, masters);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::reduce_exact;
    use crate::family::{Integral, IntegralFamily, Kinematics, Propagator};
    use crate::masters::MasterIntegral;
    use crate::reduce::reduce;
    use crate::symbols::S;
    use symbolica::atom::{Atom, AtomCore};
    use symbolica::{function, symbol};

    fn n(x: i64) -> Atom {
        Atom::num(x)
    }

    fn family(
        masses: Vec<Atom>,
        invariants: Vec<Atom>,
        exps: Vec<i32>,
        num: Atom,
    ) -> IntegralFamily {
        IntegralFamily {
            propagators: (masses.into_iter())
                .map(|mass_sq| Propagator {
                    momentum: Atom::Zero,
                    mass_sq,
                })
                .collect(),
            isps: vec![],
            kinematics: Kinematics { invariants },
            targets: vec![Integral {
                propagator_exponents: exps,
                isp_exponents: vec![],
            }],
            numerator: num,
        }
    }

    /// Compare two reductions, writing `B0(0,m,m) = (d-2)/(2m^2) A0(m)` first:
    /// the exact solver sees that a zero-momentum equal-mass bubble is a
    /// tadpole, the recursion keeps it as a bubble.
    fn assert_same(what: &str, a: &[(Atom, MasterIntegral)], b: &[(Atom, MasterIntegral)]) {
        let normal = |t: &[(Atom, MasterIntegral)]| -> Vec<(Atom, MasterIntegral)> {
            let mut out = Vec::new();
            for (c, m) in t {
                let term = match m {
                    MasterIntegral::Bubble { p_sq, m1_sq, m2_sq }
                        if p_sq.is_zero() && m1_sq == m2_sq && !m1_sq.is_zero() =>
                    {
                        let factor = (Atom::var(S.d) - n(2)) / (n(2) * m1_sq);
                        (
                            c * factor,
                            MasterIntegral::Tadpole {
                                m_sq: m1_sq.clone(),
                            },
                        )
                    }
                    _ => (c.clone(), m.clone()),
                };
                crate::reduce::add_scaled(&mut out, &n(1), vec![term]);
            }
            out.retain(|(c, _)| !c.together().expand().is_zero());
            out
        };
        let (a, b) = (normal(a), normal(b));
        assert_eq!(a.len(), b.len(), "{what}: {a:?}\nvs {b:?}");
        for (c, m) in &a {
            let (c2, _) = b
                .iter()
                .find(|(_, m2)| m2 == m)
                .unwrap_or_else(|| panic!("{what}: {m:?}"));
            assert!(
                (c - c2).together().expand().is_zero(),
                "{what}, {m:?}: {c} vs {c2}"
            );
        }
    }

    fn dot(a: &Atom, b: &Atom) -> Atom {
        function!(S.dot, a, b)
    }

    fn q(i: usize) -> Atom {
        Atom::var(symbol!(format!("oneloopreduce::q{i}")))
    }

    /// Away from degenerate points both methods are exact, so they must agree.
    #[test]
    fn matches_the_recursion_at_generic_kinematics() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let box_inv = || vec![n(-2), n(-7), n(-3), n(-4), n(-5), n(-6)];
        let cases = [
            family(vec![n(1), n(2)], vec![n(-3)], vec![2, 1], n(1)),
            family(vec![n(0), n(0)], vec![n(-5)], vec![3, 2], n(1)),
            family(
                vec![n(1), n(2), n(3)],
                vec![n(-1), n(-2), n(-3)],
                vec![2, 1, 1],
                n(1),
            ),
            family(
                vec![n(1); 3],
                vec![n(-1), n(-2), n(-3)],
                vec![1; 3],
                dot(&k, &q(1)) * dot(&k, &q(2)),
            ),
            family(vec![n(0); 4], box_inv(), vec![2, 1, 1, 1], n(1)),
            family(
                vec![n(1); 4],
                box_inv(),
                vec![1; 4],
                dot(&k, &q(1)) * dot(&k, &k),
            ),
            family(vec![n(1); 4], box_inv(), vec![1, 2, 1, 1], dot(&k, &q(3))),
        ];
        for (i, fam) in cases.iter().enumerate() {
            let exact = reduce_exact(fam).unwrap();
            assert_same(&format!("case {i}"), &exact, &reduce(fam).unwrap().terms);
        }
    }

    /// With two on-shell legs the massless triangle is not a master:
    /// `C0(0,0,s) = -2(d-3)/((d-4) s) B0(s,0,0)`, from the ratio of their
    /// Gamma-function closed forms.
    #[test]
    fn the_on_shell_massless_triangle_is_a_bubble() {
        crate::ensure_symbolica_license();
        let s = Atom::var(S.psq);
        let d = Atom::var(S.d);
        let fam = family(vec![n(0); 3], vec![n(0), s.clone(), n(0)], vec![1; 3], n(1));
        let want = -(n(2) * (&d - n(3))) / ((&d - n(4)) * &s);
        let bubble = MasterIntegral::Bubble {
            p_sq: s,
            m1_sq: Atom::Zero,
            m2_sq: Atom::Zero,
        };
        assert_same("C0(0,0,s)", &reduce_exact(&fam).unwrap(), &[(want, bubble)]);
    }

    /// Where the shared-delta limit exists, the exact reduction at the on-shell
    /// point agrees with it -- an independent check of that limit.
    #[test]
    fn matches_the_on_shell_limit_where_it_exists() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let two_legs = || vec![n(0), n(-7), n(-3), n(0), n(-5), n(-2)];
        let four_legs = || vec![n(0), n(-7), n(0), n(0), n(-5), n(0)];
        let cases = [
            family(vec![n(1); 4], two_legs(), vec![2, 1, 1, 1], n(1)),
            family(vec![n(1); 4], four_legs(), vec![1, 2, 1, 1], n(1)),
            family(
                vec![n(1); 4],
                two_legs(),
                vec![1; 4],
                dot(&k, &q(1)) * dot(&k, &q(3)) * dot(&k, &k),
            ),
            family(
                vec![n(1); 3],
                vec![n(0), n(2) / n(5), n(0)],
                vec![2, 1, 1],
                n(1),
            ),
        ];
        for (i, fam) in cases.iter().enumerate() {
            assert_same(
                &format!("case {i}"),
                &reduce_exact(fam).unwrap(),
                &reduce(fam).unwrap().terms,
            );
        }
    }

    /// The two configurations the shared delta cannot do: `reduce` falls back
    /// to the exact solver and returns a finite result.
    #[test]
    fn reduces_where_the_on_shell_limit_does_not_exist() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let cases = [
            family(
                vec![n(0); 4],
                vec![n(0), n(-7), n(-3), n(0), n(-5), n(-2)],
                vec![2, 1, 1, 1],
                n(1),
            ),
            family(
                vec![n(1); 3],
                vec![n(0), n(-3), n(0)],
                vec![2, 2, 1],
                dot(&k, &q(2)),
            ),
        ];
        for (i, fam) in cases.iter().enumerate() {
            let r = reduce(fam).unwrap_or_else(|e| panic!("case {i}: {e}"));
            assert_same(&format!("case {i}"), &r.terms, &reduce_exact(fam).unwrap());
            assert!(!r.terms.is_empty(), "case {i}");
        }
    }
}

//! Reduce every dot(k,.) monomial (rank <= 4) of a delta-regularized on-shell massless box.
//!
//! The four external legs are on-shell and massless, which collapses a row of the box's
//! Gram matrix; the two vanishing invariants are carried as an explicit off-shellness
//! `delta` instead, so the 1/delta inverse-Gram poles stay rational and cancel when the
//! caller takes delta -> 0. Prints, per monomial `(k^2)^a (k.q1)^b1 (k.q2)^b2 (k.q3)^b3`,
//!
//!     MONO a b1 b2 b3
//!     TERM coeff=( .. ) <MASTER args...>      (repeated)
//!     ENDMONO
//!
//! Kinematics come from argv as eight integers -- numerator/denominator pairs for
//! s, t, m^2 and delta, in that order. There are no defaults; the validated point is
//!   cargo run --release --example box_reduce -p one-loop-reduce -- -3 1 -17 10 1 1 1 100000
//! i.e. s = -3, t = -17/10, m^2 = 1, delta = 1e-5.

use oneloopreduce::masters::MasterIntegral;
use oneloopreduce::symbols::S;
use oneloopreduce::{Integral, IntegralFamily, Kinematics, Propagator, reduce};
use symbolica::atom::{Atom, AtomCore};
use symbolica::function;

fn family(s: &Atom, t: &Atom, msq: &Atom, dl: &Atom, numerator: Atom) -> IntegralFamily {
    IntegralFamily {
        propagators: (0..4)
            .map(|_| Propagator {
                momentum: Atom::Zero,
                mass_sq: msq.clone(),
            })
            .collect(),
        isps: vec![],
        kinematics: Kinematics {
            invariants: vec![
                dl.clone(),
                s.clone(),
                dl.clone(),
                dl.clone(),
                t.clone(),
                dl.clone(),
            ],
        },
        targets: vec![Integral {
            propagator_exponents: vec![1, 1, 1, 1],
            isp_exponents: vec![],
        }],
        numerator,
    }
}

fn master_line(m: &MasterIntegral) -> String {
    let s = |x: &Atom| x.expand().to_string();
    match m {
        MasterIntegral::Tadpole { m_sq } => format!("A0 {}", s(m_sq)),
        MasterIntegral::Bubble { p_sq, m1_sq, m2_sq } => {
            format!("B0 {} {} {}", s(p_sq), s(m1_sq), s(m2_sq))
        }
        MasterIntegral::Triangle {
            p1_sq,
            p2_sq,
            p12_sq,
            m1_sq,
            m2_sq,
            m3_sq,
        } => format!(
            "C0 {} {} {} {} {} {}",
            s(p1_sq),
            s(p2_sq),
            s(p12_sq),
            s(m1_sq),
            s(m2_sq),
            s(m3_sq)
        ),
        MasterIntegral::Box {
            p1_sq,
            p2_sq,
            p3_sq,
            p4_sq,
            s: si,
            t,
            m1_sq,
            m2_sq,
            m3_sq,
            m4_sq,
        } => format!(
            "D0 {} {} {} {} {} {} {} {} {} {}",
            s(p1_sq),
            s(p2_sq),
            s(p3_sq),
            s(p4_sq),
            s(si),
            s(t),
            s(m1_sq),
            s(m2_sq),
            s(m3_sq),
            s(m4_sq)
        ),
    }
}

fn main() {
    if let Ok(key) = std::env::var("SYMBOLICA_LICENSE") {
        let _ = symbolica::prelude::LicenseManager::set_license_key(&key);
    }
    let a: Vec<i64> = std::env::args()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let s = Atom::num(a[0]) / Atom::num(a[1]);
    let t = Atom::num(a[2]) / Atom::num(a[3]);
    let msq = Atom::num(a[4]) / Atom::num(a[5]);
    let dl = Atom::num(a[6]) / Atom::num(a[7]);
    println!(
        "S {} {}  T {} {}  MSQ {} {}  DELTA {} {}",
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    );

    let k = Atom::var(S.k);
    let ll = function!(S.dot, k.clone(), k.clone());
    let lp: Vec<Atom> = (1..=3)
        .map(|j| {
            function!(
                S.dot,
                k.clone(),
                Atom::var(symbolica::symbol!(format!("oneloopreduce::q{j}")))
            )
        })
        .collect();

    for aa in 0..=2 {
        for b1 in 0..=(4 - 2 * aa) {
            for b2 in 0..=(4 - 2 * aa - b1) {
                for b3 in 0..=(4 - 2 * aa - b1 - b2) {
                    let mut num = Atom::num(1);
                    for _ in 0..aa {
                        num = &num * &ll;
                    }
                    for _ in 0..b1 {
                        num = &num * &lp[0];
                    }
                    for _ in 0..b2 {
                        num = &num * &lp[1];
                    }
                    for _ in 0..b3 {
                        num = &num * &lp[2];
                    }
                    let r = reduce(&family(&s, &t, &msq, &dl, num)).unwrap();
                    println!("MONO {aa} {b1} {b2} {b3}");
                    for (c, m) in &r.terms {
                        println!("TERM coeff=( {} ) {}", c.expand(), master_line(m));
                    }
                    println!("ENDMONO");
                }
            }
        }
    }
}

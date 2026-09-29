use symbolica::atom::Atom;

use crate::symbols::S;

#[derive(Debug, Clone, PartialEq)]
pub enum MasterIntegral {
    Tadpole {
        m_sq: Atom,
    },
    Bubble {
        p_sq: Atom,
        m1_sq: Atom,
        m2_sq: Atom,
    },
    Triangle {
        p1_sq: Atom,
        p2_sq: Atom,
        p12_sq: Atom,
        m1_sq: Atom,
        m2_sq: Atom,
        m3_sq: Atom,
    },
    Box {
        p1_sq: Atom,
        p2_sq: Atom,
        p3_sq: Atom,
        p4_sq: Atom,
        s: Atom,
        t: Atom,
        m1_sq: Atom,
        m2_sq: Atom,
        m3_sq: Atom,
        m4_sq: Atom,
    },
}

pub trait MasterBasis {
    fn symbol(&self, integral: &MasterIntegral) -> Atom;
}

pub struct OneLoopMasters;

impl MasterIntegral {
    pub(crate) fn map_arguments(&self, mut map: impl FnMut(&Atom) -> Atom) -> Self {
        match self {
            Self::Tadpole { m_sq } => Self::Tadpole { m_sq: map(m_sq) },
            Self::Bubble { p_sq, m1_sq, m2_sq } => Self::Bubble {
                p_sq: map(p_sq),
                m1_sq: map(m1_sq),
                m2_sq: map(m2_sq),
            },
            Self::Triangle {
                p1_sq,
                p2_sq,
                p12_sq,
                m1_sq,
                m2_sq,
                m3_sq,
            } => Self::Triangle {
                p1_sq: map(p1_sq),
                p2_sq: map(p2_sq),
                p12_sq: map(p12_sq),
                m1_sq: map(m1_sq),
                m2_sq: map(m2_sq),
                m3_sq: map(m3_sq),
            },
            Self::Box {
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
            } => Self::Box {
                p1_sq: map(p1_sq),
                p2_sq: map(p2_sq),
                p3_sq: map(p3_sq),
                p4_sq: map(p4_sq),
                s: map(s),
                t: map(t),
                m1_sq: map(m1_sq),
                m2_sq: map(m2_sq),
                m3_sq: map(m3_sq),
                m4_sq: map(m4_sq),
            },
        }
    }

    /// Kinematic arguments in the native OneLOop order, without the scale.
    pub fn arguments(&self) -> Vec<&Atom> {
        match self {
            Self::Tadpole { m_sq } => vec![m_sq],
            Self::Bubble { p_sq, m1_sq, m2_sq } => vec![p_sq, m1_sq, m2_sq],
            Self::Triangle {
                p1_sq,
                p2_sq,
                p12_sq,
                m1_sq,
                m2_sq,
                m3_sq,
            } => vec![p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq],
            Self::Box {
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
        }
    }
}

impl OneLoopMasters {
    /// An untagged primitive `oneloopmaster::A0/B0/C0/D0` inspection call.
    ///
    /// The squared renormalization scale is the last argument. The same native
    /// Symbols accept a leading Laurent tag (`0`, `-1`, or `-2`) for numerical
    /// evaluation through their registered direct Rust hooks.
    pub fn symbol_with_scale(&self, integral: &MasterIntegral, mu_squared: &Atom) -> Atom {
        let head = match integral {
            MasterIntegral::Tadpole { .. } => S.a0,
            MasterIntegral::Bubble { .. } => S.b0,
            MasterIntegral::Triangle { .. } => S.c0,
            MasterIntegral::Box { .. } => S.d0,
        };
        let mut arguments = integral.arguments();
        arguments.push(mu_squared);
        head.call(arguments.as_slice())
    }
}

impl MasterBasis for OneLoopMasters {
    fn symbol(&self, integral: &MasterIntegral) -> Atom {
        self.symbol_with_scale(integral, &Atom::num(1))
    }
}

#[cfg(test)]
mod tests {
    use super::{MasterBasis, MasterIntegral, OneLoopMasters};
    use crate::symbols::S;
    use symbolica::atom::Atom;
    use symbolica::function;

    fn massless_bubble() -> MasterIntegral {
        MasterIntegral::Bubble {
            p_sq: Atom::var(S.psq),
            m1_sq: Atom::Zero,
            m2_sq: Atom::Zero,
        }
    }

    #[test]
    fn bubble_maps_to_symbolic_b0() {
        crate::ensure_symbolica_license();
        let got = OneLoopMasters.symbol(&massless_bubble());
        let want = function!(S.b0, Atom::var(S.psq), Atom::Zero, Atom::Zero, 1);
        assert_eq!(got, want);
    }

    #[test]
    fn tadpole_maps_to_a0() {
        crate::ensure_symbolica_license();
        let m = MasterIntegral::Tadpole { m_sq: Atom::num(1) };
        assert_eq!(OneLoopMasters.symbol(&m), function!(S.a0, Atom::num(1), 1));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn reducer_master_symbols_have_direct_native_numerical_hooks() {
        use symbolica::domains::float::Complex;

        crate::ensure_symbolica_license();
        for master in [S.a0, S.b0, S.c0, S.d0] {
            assert!(master.get_evaluation_info().is_some());
        }

        let info = S.a0.get_evaluation_info().unwrap();
        let arguments = [Complex::new(2.0_f64, 0.0), Complex::new(3.0, 0.0)];
        let expected = [2.0 * (1.0 - (2.0_f64 / 3.0).ln()), 2.0, 0.0];
        for (tag, expected) in [0, -1, -2].into_iter().zip(expected) {
            let callback = info
                .get_evaluator::<Complex<f64>>(&[Atom::num(tag).as_view()])
                .expect("registered direct native master hook");
            let value = callback(&arguments);
            assert!((value.re - expected).abs() < 1e-12);
            assert_eq!(value.im, 0.0);
        }
    }

    #[test]
    fn triangle_maps_to_c0() {
        crate::ensure_symbolica_license();
        let m = MasterIntegral::Triangle {
            p1_sq: Atom::num(1),
            p2_sq: Atom::num(2),
            p12_sq: Atom::num(3),
            m1_sq: Atom::Zero,
            m2_sq: Atom::Zero,
            m3_sq: Atom::Zero,
        };
        let want = function!(
            S.c0,
            Atom::num(1),
            Atom::num(2),
            Atom::num(3),
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            Atom::num(1)
        );
        assert_eq!(OneLoopMasters.symbol(&m), want);
    }

    #[test]
    fn box_maps_to_d0() {
        crate::ensure_symbolica_license();
        let m = MasterIntegral::Box {
            p1_sq: Atom::Zero,
            p2_sq: Atom::Zero,
            p3_sq: Atom::Zero,
            p4_sq: Atom::Zero,
            s: Atom::num(1),
            t: Atom::num(2),
            m1_sq: Atom::Zero,
            m2_sq: Atom::Zero,
            m3_sq: Atom::Zero,
            m4_sq: Atom::Zero,
        };
        let want = function!(
            S.d0,
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            Atom::num(1),
            Atom::num(2),
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            Atom::num(1)
        );
        assert_eq!(OneLoopMasters.symbol(&m), want);
    }
}

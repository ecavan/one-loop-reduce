use std::sync::LazyLock;
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    symbol,
};

use crate::OneLoopError;

/// Reject obsolete reducer-only symbols anywhere in a symbolic input.
pub fn validate_namespace(input: &Atom) -> Result<(), OneLoopError> {
    let mut obsolete = None;
    input.visitor(&mut |view| {
        if let Some(symbol) = view.get_symbol()
            && symbol.get_name().starts_with("oneloopreduce::")
        {
            obsolete = Some(symbol.get_name().to_string());
        }
        obsolete.is_none()
    });
    if let Some(name) = obsolete {
        Err(OneLoopError::ObsoleteSymbol { name })
    } else {
        Ok(())
    }
}

pub struct OneLoopSymbols {
    /// Spacetime dimension `d`
    pub d: Symbol,
    /// The bubble invariant `p^2`.
    pub psq: Symbol,
    /// The loop momentum.
    pub k: Symbol,
    /// External momenta
    pub q1: Symbol,
    pub q2: Symbol,
    pub q3: Symbol,
    /// The master-integral heads A0/B0/C0/D0.
    pub a0: Symbol,
    pub b0: Symbol,
    pub c0: Symbol,
    pub d0: Symbol,
}

pub static S: LazyLock<OneLoopSymbols> = LazyLock::new(|| OneLoopSymbols {
    d: symbol!("oneloopmaster::d"),
    psq: symbol!("oneloopmaster::psq"),
    k: symbol!("oneloopmaster::k"),
    q1: symbol!("oneloopmaster::q1"),
    q2: symbol!("oneloopmaster::q2"),
    q3: symbol!("oneloopmaster::q3"),
    #[cfg(not(target_arch = "wasm32"))]
    a0: oneloop::A0(),
    #[cfg(target_arch = "wasm32")]
    a0: symbol!("oneloopmaster::A0"),
    #[cfg(not(target_arch = "wasm32"))]
    b0: oneloop::B0(),
    #[cfg(target_arch = "wasm32")]
    b0: symbol!("oneloopmaster::B0"),
    #[cfg(not(target_arch = "wasm32"))]
    c0: oneloop::C0(),
    #[cfg(target_arch = "wasm32")]
    c0: symbol!("oneloopmaster::C0"),
    #[cfg(not(target_arch = "wasm32"))]
    d0: oneloop::D0(),
    #[cfg(target_arch = "wasm32")]
    d0: symbol!("oneloopmaster::D0"),
});

/// Scalar products in the internal coordinate chart use the same Spenso
/// representation as the shared HEP family, rather than a second dot function.
pub fn scalar_product(left: &Atom, right: &Atom) -> Atom {
    static KINEMATICS: LazyLock<feynkit_kinematics::Kinematics> = LazyLock::new(|| {
        feynkit_kinematics::Kinematics::in_dimension(&Atom::var(S.d))
            .expect("the backend dimension is symbolic")
            .with_momenta(
                std::iter::once(Atom::var(S.k))
                    .chain((1..=32).map(|i| Atom::var(symbol!(format!("oneloopmaster::q{i}"))))),
            )
            .expect("internal momentum names are symbols")
    });
    KINEMATICS
        .scalar_product(left, right)
        .expect("internal momenta are linear")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_validation_checks_inside_composite_scales() {
        crate::ensure_symbolica_license();
        let scale = Atom::num(1) + Atom::var(symbol!("oneloopreduce::mu_squared"));
        assert!(matches!(
            validate_namespace(&scale),
            Err(OneLoopError::ObsoleteSymbol { .. })
        ));
        let scale = Atom::num(1) + Atom::var(symbol!("kinematics::mu_squared"));
        assert!(validate_namespace(&scale).is_ok());
    }
}

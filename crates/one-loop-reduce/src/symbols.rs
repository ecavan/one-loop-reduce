use std::sync::LazyLock;
use symbolica::{atom::Symbol, symbol};

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
    /// Symmetric, linear dot product for numerators
    pub dot: Symbol,
    /// The master-integral heads A0/B0/C0/D0.
    pub a0: Symbol,
    pub b0: Symbol,
    pub c0: Symbol,
    pub d0: Symbol,
}

pub static S: LazyLock<OneLoopSymbols> = LazyLock::new(|| OneLoopSymbols {
    d: symbol!("oneloopreduce::d"),
    psq: symbol!("oneloopreduce::psq"),
    k: symbol!("oneloopreduce::k"),
    q1: symbol!("oneloopreduce::q1"),
    q2: symbol!("oneloopreduce::q2"),
    q3: symbol!("oneloopreduce::q3"),
    dot: symbol!("oneloopreduce::dot"; Symmetric, Linear),
    a0: symbol!("oneloopreduce::A0"),
    b0: symbol!("oneloopreduce::B0"),
    c0: symbol!("oneloopreduce::C0"),
    d0: symbol!("oneloopreduce::D0"),
});

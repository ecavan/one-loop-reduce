//! One-loop IBP reduction of Feynman integrals to the four scalar master integrals.

pub mod amplitude;
pub mod bridge;
pub mod error;
pub mod family;
pub mod masters;
pub mod reduce;
pub mod symbols;

pub use amplitude::amplitude;
pub use error::OneLoopError;
pub use family::{Integral, IntegralFamily, Isp, Kinematics, Propagator};
pub use masters::{MasterBasis, MasterIntegral, OneLoopMasters};
pub use reduce::{Reduction, reduce};

/// Activate the Symbolica license once per process, from `SYMBOLICA_LICENSE`.
///
/// Symbolica allows a single unlicensed instance per process and aborts when it
/// is touched from a second thread, so every Symbolica-using test calls this
/// first and the suite is run with `--test-threads=1`. With no key in the
/// environment the call is a no-op and Symbolica runs restricted, which is
/// enough for the test suite.
#[cfg(test)]
pub(crate) fn ensure_symbolica_license() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Ok(key) = std::env::var("SYMBOLICA_LICENSE") {
            let _ = symbolica::license::LicenseManager::set_license_key(&key);
        }
    });
}

//! Normalized scalar data consumed by the internal one-loop recurrences.
//!
//! Family construction, propagator analysis and kinematics belong to Feynkit.
//! The shared-family adapter produces this coordinate representation after
//! applying those operations. The archived validation emitters also use it.

use symbolica::atom::Atom;

#[derive(Debug, Clone)]
pub struct RecurrenceInput {
    pub masses_squared: Vec<Atom>,
    /// Pairwise invariants `(r_i-r_j)^2` in lexicographic order.
    pub invariants: Vec<Atom>,
    /// Nonnegative powers of the normalized quadratic lines.
    pub powers: Vec<i32>,
    /// Scalar polynomial expressed in the recurrence coordinates.
    pub numerator: Atom,
}

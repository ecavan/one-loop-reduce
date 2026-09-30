//! The recurrences' input in their own coordinates, built by `shared_family`
//! from a FeynKit family (the benchmarks build it directly).

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

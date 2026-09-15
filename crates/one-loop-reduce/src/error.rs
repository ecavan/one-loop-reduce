use thiserror::Error;

#[derive(Debug, Error)]
pub enum OneLoopError {
    /// The input graph is not one loop.
    #[error("unsupported loop order: graph has {found} loops, only one-loop is supported")]
    UnsupportedLoopOrder { found: usize },

    /// A host-side graph (e.g. a gammaloop `Graph`, handed over through
    /// [`crate::bridge`]) could not be turned into an `IntegralFamily`.
    #[error("failed to extract integral family from graph: {reason}")]
    ExtractionFailed { reason: String },

    /// Propagator indices the IBP recursion cannot bottom out on.
    #[error(
        "unsupported propagator indices {found:?}: every index must be non-negative \
         (a negative index belongs in `numerator`) and their total at most {max}"
    )]
    UnsupportedIndex { found: Vec<i32>, max: i32 },

    /// Wraps an underlying Symbolica error surfaced during extraction.
    #[error("symbolica error: {0}")]
    Symbolica(String),
}

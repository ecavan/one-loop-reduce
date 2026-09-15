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

    /// Wraps an underlying Symbolica error surfaced during extraction.
    #[error("symbolica error: {0}")]
    Symbolica(String),
}

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum QuantCodecError {
    #[error("{type_name} cannot be empty")]
    EmptyIdentifier { type_name: &'static str },

    #[error("{type_name} contains leading or trailing whitespace")]
    IdentifierWhitespace { type_name: &'static str },

    #[error("invalid shape: {reason}")]
    InvalidShape { reason: String },

    #[error("invalid token span: start={start}, end={end}")]
    InvalidTokenSpan { start: u64, end: u64 },

    #[error("shape mismatch: {reason}")]
    ShapeMismatch { reason: String },

    #[error("integer overflow while computing {context}")]
    IntegerOverflow { context: &'static str },
}

use serde::{Deserialize, Serialize};

use crate::error::{ProveKvError, Result};
use crate::hybrid_manifest::HybridStateManifestV1;

/// Content-addressed identity for a hybrid KV state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HybridStateId(pub String);

impl HybridStateId {
    pub const PREFIX: &'static str = "hybrid-state-v1:";

    /// Derive an identity from the manifest's canonical, runtime-owned fields.
    pub fn from_manifest(manifest: &HybridStateManifestV1) -> Result<Self> {
        manifest.validate()?;
        let canonical = manifest.canonical_bytes()?;
        Ok(Self(format!(
            "{}{}",
            Self::PREFIX,
            blake3::hash(&canonical).to_hex()
        )))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn verify_manifest(&self, manifest: &HybridStateManifestV1) -> Result<()> {
        let expected = Self::from_manifest(manifest)?;
        if &expected != self {
            return Err(ProveKvError::InvalidManifest(format!(
                "state identity mismatch: expected {}, got {}",
                expected.0, self.0
            )));
        }
        Ok(())
    }
}

impl std::fmt::Display for HybridStateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl TryFrom<String> for HybridStateId {
    type Error = ProveKvError;
    fn try_from(value: String) -> Result<Self> {
        if value.starts_with(Self::PREFIX) && value.len() == Self::PREFIX.len() + 64 {
            Ok(Self(value))
        } else {
            Err(ProveKvError::InvalidManifest(
                "invalid hybrid state id".into(),
            ))
        }
    }
}

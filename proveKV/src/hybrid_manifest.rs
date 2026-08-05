use serde::{Deserialize, Serialize};

use crate::error::{ProveKvError, Result};
use crate::shape::KvTensorShape;
use crate::state_id::HybridStateId;

pub const HYBRID_MANIFEST_SCHEMA: &str = "hybrid_state_manifest_v1";

/// A deterministic description of one runtime component in a hybrid state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridComponent {
    pub name: String,
    pub version: String,
    pub digest: String,
}

/// Content reference for a page owned by proveKV.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridPageRef {
    pub page_id: String,
    pub digest: String,
}

/// CPU-first manifest containing every field that participates in replay identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridStateManifestV1 {
    pub schema_version: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub shape: KvTensorShape,
    pub component_inventory: Vec<HybridComponent>,
    pub page_refs: Vec<HybridPageRef>,
    pub parent_lineage: Vec<HybridStateId>,
    pub policy_digest: String,
    pub version_digest: String,
}

impl HybridStateManifestV1 {
    pub fn new(
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        shape: KvTensorShape,
        component_inventory: Vec<HybridComponent>,
        page_refs: Vec<HybridPageRef>,
        parent_lineage: Vec<HybridStateId>,
        policy_digest: impl Into<String>,
        version_digest: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: HYBRID_MANIFEST_SCHEMA.into(),
            model_id: model_id.into(),
            tokenizer_id: tokenizer_id.into(),
            shape,
            component_inventory,
            page_refs,
            parent_lineage,
            policy_digest: policy_digest.into(),
            version_digest: version_digest.into(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != HYBRID_MANIFEST_SCHEMA {
            return Err(ProveKvError::InvalidManifest(
                "unsupported hybrid manifest schema".into(),
            ));
        }
        if self.model_id.is_empty() || self.tokenizer_id.is_empty() {
            return Err(ProveKvError::InvalidManifest(
                "model_id and tokenizer_id are required".into(),
            ));
        }
        if self.policy_digest.is_empty() || self.version_digest.is_empty() {
            return Err(ProveKvError::InvalidManifest(
                "policy/version digests are required".into(),
            ));
        }
        self.shape.validate()?;
        if self
            .component_inventory
            .iter()
            .any(|c| c.name.is_empty() || c.version.is_empty() || c.digest.is_empty())
        {
            return Err(ProveKvError::InvalidManifest(
                "component inventory contains an empty identity field".into(),
            ));
        }
        if self
            .page_refs
            .iter()
            .any(|p| p.page_id.is_empty() || p.digest.is_empty())
        {
            return Err(ProveKvError::InvalidManifest(
                "page reference contains an empty identity field".into(),
            ));
        }
        Ok(())
    }

    /// Stable JSON bytes: serde's declaration order is the canonical field order.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }
    pub fn digest(&self) -> Result<String> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }
    pub fn state_id(&self) -> Result<HybridStateId> {
        HybridStateId::from_manifest(self)
    }
}

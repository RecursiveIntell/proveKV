use crate::{
    ArtifactDigest, CodecProfileDigest, KvCacheShapeV2, KvLayout, KvRole, LayerId,
    ModelFingerprint, QuantCodecError, TokenizerFingerprint,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The canonical component families supported by the hybrid cache contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HybridComponentKind {
    AttentionKv,
    Convolution,
    Recurrent,
}

/// A semantic axis in a component's canonical tensor order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HybridAxis {
    Batch,
    Layer,
    Head,
    Token,
    Feature,
    State,
}

/// Sequence behavior is part of the persisted contract, not runtime policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HybridSequenceSemantics {
    TokenIndexed,
    Recurrent,
    Convolutional,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HybridComponentLayoutV1 {
    pub kind: HybridComponentKind,
    pub layer: LayerId,
    pub axes: Vec<HybridAxis>,
    pub shape: KvCacheShapeV2,
    pub sequence: HybridSequenceSemantics,
    pub role: Option<KvRole>,
    pub codec_profile_digest: CodecProfileDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HybridCacheLayoutV1 {
    pub model: ModelFingerprint,
    pub tokenizer: TokenizerFingerprint,
    pub components: Vec<HybridComponentLayoutV1>,
}

impl HybridCacheLayoutV1 {
    pub fn new(
        model: ModelFingerprint,
        tokenizer: TokenizerFingerprint,
        components: Vec<HybridComponentLayoutV1>,
    ) -> Result<Self, QuantCodecError> {
        let layout = Self {
            model,
            tokenizer,
            components,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn validate(&self) -> Result<(), QuantCodecError> {
        if self.components.is_empty() {
            return Err(invalid("at least one hybrid component is required"));
        }
        let mut previous: Option<(LayerId, HybridComponentKind)> = None;
        for component in &self.components {
            component.shape.validate()?;
            if component.axes.is_empty() || component.axes.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid(
                    "component axes must be non-empty and strictly canonical",
                ));
            }
            if component.axes.windows(2).any(|w| w[0] == w[1]) {
                return Err(invalid("component axes cannot contain duplicates"));
            }
            if let Some(prev) = previous {
                if (component.layer, component.kind) <= prev {
                    return Err(invalid(
                        "components must be sorted by unique layer and kind",
                    ));
                }
            }
            previous = Some((component.layer, component.kind));
            match component.kind {
                HybridComponentKind::AttentionKv if component.role.is_none() => {
                    return Err(invalid("attention KV components require a role"))
                }
                HybridComponentKind::Convolution
                    if component.sequence != HybridSequenceSemantics::Convolutional =>
                {
                    return Err(invalid(
                        "convolution components require convolutional sequence semantics",
                    ))
                }
                HybridComponentKind::Recurrent
                    if component.sequence != HybridSequenceSemantics::Recurrent =>
                {
                    return Err(invalid(
                        "recurrent components require recurrent sequence semantics",
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Stable, length-delimited bytes independent of serde format or map ordering.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, QuantCodecError> {
        self.validate()?;
        let mut out = Vec::new();
        put_str(&mut out, self.model.as_str());
        put_str(&mut out, self.tokenizer.as_str());
        out.extend_from_slice(&(self.components.len() as u32).to_le_bytes());
        for c in &self.components {
            out.push(c.kind as u8);
            out.extend_from_slice(&c.layer.0.to_le_bytes());
            out.push(c.sequence as u8);
            out.push(c.role.map(|r| r as u8 + 1).unwrap_or(0));
            out.extend_from_slice(&(c.axes.len() as u32).to_le_bytes());
            for axis in &c.axes {
                out.push(*axis as u8);
            }
            encode_shape(&mut out, &c.shape);
            out.extend_from_slice(c.codec_profile_digest.as_bytes());
        }
        Ok(out)
    }

    pub fn digest(&self) -> Result<ArtifactDigest, QuantCodecError> {
        Ok(ArtifactDigest::from_canonical_bytes(
            &self.canonical_bytes()?,
        ))
    }
}

fn encode_shape(out: &mut Vec<u8>, s: &KvCacheShapeV2) {
    for n in [
        s.batch as u64,
        s.layers as u64,
        s.num_q_heads as u64,
        s.num_kv_heads as u64,
        s.seq_len,
        s.head_dim as u64,
    ] {
        out.extend_from_slice(&n.to_le_bytes());
    }
    out.push(s.dtype as u8);
    match &s.layout {
        KvLayout::LayersHeadsTokensDim => out.push(0),
        KvLayout::LayersTokensHeadsDim => out.push(1),
        KvLayout::RuntimeSpecific(label) => {
            out.push(2);
            put_str(out, label);
        }
    }
}
fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}
fn invalid(reason: impl Into<String>) -> QuantCodecError {
    QuantCodecError::InvalidShape {
        reason: reason.into(),
    }
}

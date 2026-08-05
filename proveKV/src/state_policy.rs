//! Per-component codec policy and fallback.
//!
//! Every hybrid state component receives an explicit profile decision.
//! Unknown or unadmitted profiles fall back to raw. Full-attention K/V may
//! use admitted lossy profiles; convolution/recurrent state defaults to raw.

use serde::{Deserialize, Serialize};

/// Named codec profile for a component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateProfile {
    /// Exact raw bytes, no compression.
    RawExact,
    /// 4-bit radii-preserved compression (radius stored at f32, angles quantized).
    RadiiPreserved4Bit,
    /// 4-bit radii-lossy compression (both radii and angles quantized).
    RadiiLossy4Bit,
    /// Named custom profile with digest.
    Custom { name: String, digest: String },
}

impl StateProfile {
    /// True if this profile does not apply any compression.
    pub fn is_raw(&self) -> bool {
        matches!(self, StateProfile::RawExact)
    }

    /// True if this profile applies lossy compression.
    pub fn is_lossy(&self) -> bool {
        matches!(
            self,
            StateProfile::RadiiPreserved4Bit | StateProfile::RadiiLossy4Bit
        )
    }
}

/// Resolved decision for one component in a hybrid state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentPolicyDecision {
    /// Component kind (e.g. "full_attn_k", "full_attn_v", "conv_state",
    /// "recurrent_state").
    pub component_kind: String,
    /// Selected profile.
    pub profile: StateProfile,
    /// If the requested profile was not admitted, the reason for fallback.
    pub fallback_reason: Option<String>,
    /// Digest of the profile source that admitted this decision (codec
    /// version, calibration run, etc.).
    pub profile_source_digest: Option<String>,
}

/// Admission table for component-state codec profiles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatePolicy {
    /// Profiles admitted for full-attention key components.
    pub admitted_attn_k: Vec<StateProfile>,
    /// Profiles admitted for full-attention value components.
    pub admitted_attn_v: Vec<StateProfile>,
    /// Profiles admitted for convolution state components.
    pub admitted_conv: Vec<StateProfile>,
    /// Profiles admitted for recurrent state components.
    pub admitted_recurrent: Vec<StateProfile>,
    /// If true, unknown component kinds are rejected rather than falling back
    /// to raw.
    pub reject_unknown_components: bool,
}

impl StatePolicy {
    /// A conservative default: raw-only for everything.
    pub fn conservative() -> Self {
        Self {
            admitted_attn_k: vec![StateProfile::RawExact],
            admitted_attn_v: vec![StateProfile::RawExact],
            admitted_conv: vec![StateProfile::RawExact],
            admitted_recurrent: vec![StateProfile::RawExact],
            reject_unknown_components: true,
        }
    }

    /// Resolve a profile for a component kind.
    ///
    /// If the requested profile is in the admitted list for this component
    /// kind, it is selected. Otherwise, raw fallback is used and a reason
    /// recorded. Unknown component kinds are either rejected or fall back
    /// to raw depending on `reject_unknown_components`.
    pub fn resolve(
        &self,
        component_kind: &str,
        requested: Option<&StateProfile>,
    ) -> ComponentPolicyDecision {
        let (admitted, default_profile) = match component_kind {
            "full_attn_k" => (&self.admitted_attn_k, StateProfile::RawExact),
            "full_attn_v" => (&self.admitted_attn_v, StateProfile::RawExact),
            "conv_state" => (&self.admitted_conv, StateProfile::RawExact),
            "recurrent_state" => (&self.admitted_recurrent, StateProfile::RawExact),
            other => {
                if self.reject_unknown_components {
                    return ComponentPolicyDecision {
                        component_kind: other.to_string(),
                        profile: StateProfile::RawExact,
                        fallback_reason: Some(format!("unknown component kind: {}", other)),
                        profile_source_digest: None,
                    };
                }
                return ComponentPolicyDecision {
                    component_kind: other.to_string(),
                    profile: StateProfile::RawExact,
                    fallback_reason: None,
                    profile_source_digest: None,
                };
            }
        };

        match requested {
            Some(req) if admitted.contains(req) => ComponentPolicyDecision {
                component_kind: component_kind.to_string(),
                profile: req.clone(),
                fallback_reason: None,
                profile_source_digest: None,
            },
            Some(req) => ComponentPolicyDecision {
                component_kind: component_kind.to_string(),
                profile: default_profile.clone(),
                fallback_reason: Some(format!(
                    "profile {:?} not admitted for {}; falling back to {:?}",
                    req, component_kind, default_profile
                )),
                profile_source_digest: None,
            },
            None => ComponentPolicyDecision {
                component_kind: component_kind.to_string(),
                profile: default_profile,
                fallback_reason: None,
                profile_source_digest: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_policy_defaults_to_raw() {
        let policy = StatePolicy::conservative();
        let d = policy.resolve("full_attn_k", None);
        assert_eq!(d.profile, StateProfile::RawExact);
        assert!(d.fallback_reason.is_none());
    }

    #[test]
    fn unadmitted_profile_falls_back() {
        let policy = StatePolicy::conservative();
        let d = policy.resolve("full_attn_k", Some(&StateProfile::RadiiLossy4Bit));
        assert_eq!(d.profile, StateProfile::RawExact);
        assert!(d.fallback_reason.is_some());
    }

    #[test]
    fn admitted_lossy_profile_accepted() {
        let mut policy = StatePolicy::conservative();
        policy
            .admitted_attn_k
            .push(StateProfile::RadiiPreserved4Bit);
        let d = policy.resolve("full_attn_k", Some(&StateProfile::RadiiPreserved4Bit));
        assert_eq!(d.profile, StateProfile::RadiiPreserved4Bit);
        assert!(d.fallback_reason.is_none());
    }

    #[test]
    fn lossy_not_admitted_for_recurrent() {
        let policy = StatePolicy::conservative();
        let d = policy.resolve("recurrent_state", Some(&StateProfile::RadiiLossy4Bit));
        assert_eq!(d.profile, StateProfile::RawExact);
        assert!(d.fallback_reason.is_some());
    }

    #[test]
    fn unknown_component_rejected() {
        let policy = StatePolicy::conservative();
        let d = policy.resolve("unknown_kind", None);
        assert!(d.fallback_reason.is_some());
    }

    #[test]
    fn unknown_component_accepted_when_permissive() {
        let mut policy = StatePolicy::conservative();
        policy.reject_unknown_components = false;
        let d = policy.resolve("unknown_kind", None);
        assert_eq!(d.profile, StateProfile::RawExact);
        assert!(d.fallback_reason.is_none());
    }

    #[test]
    fn profile_is_lossy_detection() {
        assert!(!StateProfile::RawExact.is_lossy());
        assert!(StateProfile::RadiiPreserved4Bit.is_lossy());
        assert!(StateProfile::RadiiLossy4Bit.is_lossy());
    }
}

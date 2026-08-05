use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{
    error::{ProveKvError, Result},
    principal::{ExecutionScope, Principal},
    state_id::HybridStateId,
};
use rand::RngCore;

/// Deterministic lease identifier prefix to keep opaque IDs sortable in logs.
pub const LEASE_ID_PREFIX: &str = "lease-v1:";

/// Opaque right/permission set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaseRight {
    Inspect,
    Materialize,
    Fork,
    Append,
    Release,
}

/// Compact bitset representation for lease operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseRights(u8);

impl LeaseRights {
    const INSPECT: u8 = 0b0001;
    const MATERIALIZE: u8 = 0b0010;
    const FORK: u8 = 0b0100;
    const APPEND: u8 = 0b1000;
    const RELEASE: u8 = 0b1_0000;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn all() -> Self {
        Self(Self::INSPECT | Self::MATERIALIZE | Self::FORK | Self::APPEND | Self::RELEASE)
    }

    pub const fn can(self, right: LeaseRight) -> bool {
        match right {
            LeaseRight::Inspect => self.0 & Self::INSPECT != 0,
            LeaseRight::Materialize => self.0 & Self::MATERIALIZE != 0,
            LeaseRight::Fork => self.0 & Self::FORK != 0,
            LeaseRight::Append => self.0 & Self::APPEND != 0,
            LeaseRight::Release => self.0 & Self::RELEASE != 0,
        }
    }

    pub const fn with(self, right: LeaseRight) -> Self {
        let mask = match right {
            LeaseRight::Inspect => Self::INSPECT,
            LeaseRight::Materialize => Self::MATERIALIZE,
            LeaseRight::Fork => Self::FORK,
            LeaseRight::Append => Self::APPEND,
            LeaseRight::Release => Self::RELEASE,
        };
        Self(self.0 | mask)
    }

    pub const fn without(self, right: LeaseRight) -> Self {
        let mask = match right {
            LeaseRight::Inspect => Self::INSPECT,
            LeaseRight::Materialize => Self::MATERIALIZE,
            LeaseRight::Fork => Self::FORK,
            LeaseRight::Append => Self::APPEND,
            LeaseRight::Release => Self::RELEASE,
        };
        Self(self.0 & !mask)
    }
}

/// Lease status snapshot for policy checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseStatus {
    Active,
    Expired,
    Revoked,
    MissingRight,
    StateMismatch,
    PrincipalMismatch,
}

/// Runtime lease over a reused state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateLease {
    /// Opaque, random lease id.
    pub lease_id: String,
    /// Human-facing owner identity.
    pub principal: Principal,
    /// Namespace, duplicated for quick checks and indexing.
    pub namespace: String,
    /// Run/node/attempt tuple where this lease was issued.
    pub scope: ExecutionScope,
    /// Bound state identifier.
    pub state_id: String,
    /// Allowed operations.
    pub rights: LeaseRights,
    /// Issue time in Unix milliseconds.
    pub issued_unix_ms: u64,
    /// Optional expiry time in Unix milliseconds.
    pub expires_unix_ms: Option<u64>,
    /// Lease is valid only while store revocation epoch <= this value.
    pub revocation_epoch: u64,
    /// Per-lease nonce for log correlation and replay fences.
    pub nonce: u64,
}

impl StateLease {
    /// Create a lease with validated fields and typed time semantics.
    pub fn new(
        principal: Principal,
        scope: ExecutionScope,
        state_id: &HybridStateId,
        rights: LeaseRights,
        ttl_millis: Option<u64>,
        revocation_epoch: u64,
        nonce: u64,
    ) -> Result<Self> {
        principal.validate()?;
        scope.validate()?;

        let issued_unix_ms = now_unix_millis()?;
        let namespace = principal.namespace.clone();
        let state_id = state_id.as_str().to_owned();

        let expires_unix_ms = ttl_millis.map(|ttl| issued_unix_ms + ttl);

        let mut lease = Self {
            lease_id: new_lease_id()?,
            namespace,
            principal,
            scope,
            state_id,
            rights,
            issued_unix_ms,
            expires_unix_ms,
            revocation_epoch,
            nonce,
        };

        lease.validate()?;
        Ok(lease)
    }

    /// A stable non-replayable fingerprint for receipts.
    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("lease serialization must be deterministic");
        blake3::hash(&bytes).to_hex().to_string()
    }

    pub fn status(
        &self,
        now_unix_ms: u64,
        revocation_epoch: u64,
        right: Option<LeaseRight>,
        state_id: &str,
        principal: &Principal,
    ) -> LeaseStatus {
        if principal != &self.principal || self.namespace != principal.namespace {
            return LeaseStatus::PrincipalMismatch;
        }

        if self.state_id != state_id {
            return LeaseStatus::StateMismatch;
        }

        if self
            .expires_unix_ms
            .is_some_and(|limit| now_unix_ms >= limit)
        {
            return LeaseStatus::Expired;
        }

        if revocation_epoch > self.revocation_epoch {
            return LeaseStatus::Revoked;
        }

        if let Some(right) = right {
            if !self.rights.can(right) {
                return LeaseStatus::MissingRight;
            }
        }

        LeaseStatus::Active
    }

    /// Enforce all checks for a specific attempted operation.
    pub fn authorize(
        &self,
        right: LeaseRight,
        now_unix_ms: u64,
        revocation_epoch: u64,
        state_id: &str,
        principal: &Principal,
    ) -> Result<()> {
        match self.status(
            now_unix_ms,
            revocation_epoch,
            Some(right),
            state_id,
            principal,
        ) {
            LeaseStatus::Active => Ok(()),
            LeaseStatus::Expired => Err(ProveKvError::InvalidLease("lease expired".into())),
            LeaseStatus::Revoked => Err(ProveKvError::InvalidLease("lease revoked".into())),
            LeaseStatus::MissingRight => Err(ProveKvError::InvalidLease(
                "lease missing required right".into(),
            )),
            LeaseStatus::StateMismatch => {
                Err(ProveKvError::InvalidLease("lease state mismatch".into()))
            }
            LeaseStatus::PrincipalMismatch => Err(ProveKvError::InvalidLease(
                "lease principal mismatch".into(),
            )),
        }
    }

    fn validate(&mut self) -> Result<()> {
        if !self.lease_id.starts_with(LEASE_ID_PREFIX) {
            return Err(ProveKvError::InvalidLease(
                "lease id must begin with lease-v1 prefix".into(),
            ));
        }

        if self.namespace != self.principal.namespace {
            return Err(ProveKvError::InvalidLease(
                "lease namespace must match principal namespace".into(),
            ));
        }

        if self.state_id.is_empty() {
            return Err(ProveKvError::InvalidLease("state id is required".into()));
        }

        if !self.rights.can(LeaseRight::Inspect) && self.rights == LeaseRights::empty() {
            return Err(ProveKvError::InvalidLease(
                "lease rights must include at least one operation".into(),
            ));
        }

        if self
            .expires_unix_ms
            .is_some_and(|expiry| expiry <= self.issued_unix_ms)
        {
            return Err(ProveKvError::InvalidLease(
                "lease expiry must be after issue time".into(),
            ));
        }

        Ok(())
    }
}

fn now_unix_millis() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| ProveKvError::InvalidLease(format!("system time error: {err}")))?
        .as_millis()
        .try_into()
        .expect("timestamp always fits u64"))
}

fn new_lease_id() -> Result<String> {
    let mut bytes = [0u8; 32];
    let mut rng = rand::rngs::OsRng;
    rng.fill_bytes(&mut bytes);
    let random = blake3::hash(&bytes).to_hex();
    Ok(format!("{LEASE_ID_PREFIX}{random}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_state_id() -> HybridStateId {
        HybridStateId(format!("hybrid-state-v1:{}", "f".repeat(64)))
    }

    #[test]
    fn lease_rights_roundtrip() {
        let rights = LeaseRights::empty()
            .with(LeaseRight::Inspect)
            .with(LeaseRight::Append);
        assert!(rights.can(LeaseRight::Inspect));
        assert!(rights.can(LeaseRight::Append));
        assert!(!rights.can(LeaseRight::Fork));
    }

    #[test]
    fn lease_status_tracks_expiry_and_rights() {
        let principal = Principal::new("agent-1", "tenant-a").unwrap();
        let scope = ExecutionScope::new("run-1", "node-1", 0).unwrap();
        let state_id = mk_state_id();
        let lease = StateLease::new(
            principal.clone(),
            scope,
            &state_id,
            LeaseRights::empty().with(LeaseRight::Inspect),
            Some(1000),
            0,
            0,
        )
        .unwrap();

        let now = now_unix_millis().unwrap();
        let expiry_probe = now + 1;
        assert!(matches!(
            lease.status(
                expiry_probe,
                0,
                Some(LeaseRight::Inspect),
                state_id.as_str(),
                &principal,
            ),
            LeaseStatus::Active
        ));
        assert!(matches!(
            lease.status(
                expiry_probe,
                0,
                Some(LeaseRight::Append),
                state_id.as_str(),
                &principal,
            ),
            LeaseStatus::MissingRight
        ));
        assert!(matches!(
            lease.status(
                lease.expires_unix_ms.unwrap() + 1,
                0,
                Some(LeaseRight::Inspect),
                state_id.as_str(),
                &principal
            ),
            LeaseStatus::Expired
        ));
    }

    #[test]
    fn lease_authorize_is_namespace_and_principal_safe() {
        let principal = Principal::new("agent-1", "tenant-a").unwrap();
        let scope = ExecutionScope::new("run-1", "node-1", 0).unwrap();
        let state_id = mk_state_id();
        let lease = StateLease::new(
            principal.clone(),
            scope,
            &state_id,
            LeaseRights::all(),
            None,
            0,
            0,
        )
        .unwrap();

        let wrong_principal = Principal::new("agent-2", "tenant-a").unwrap();
        assert!(lease
            .authorize(
                LeaseRight::Inspect,
                now_unix_millis().unwrap(),
                0,
                state_id.as_str(),
                &wrong_principal,
            )
            .is_err());
    }
}

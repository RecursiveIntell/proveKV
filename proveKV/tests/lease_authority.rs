use provekv::{
    AttentionType, ExecutionScope, HybridPageRef, HybridStateId, HybridStateManifestV1, KvTensorShape,
    LeaseRight, LeaseRights, Principal, StateLease,
};

fn manifest() -> HybridStateManifestV1 {
    HybridStateManifestV1::new(
        "model-a",
        "tokenizer-a",
        KvTensorShape {
            attention_type: AttentionType::MHA,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            hidden_size: 8,
        },
        vec![provekv::HybridComponent {
            name: "cpu-kv".into(),
            version: "1".into(),
            digest: "component-digest".into(),
        }],
        vec![HybridPageRef {
            page_id: "page-0".into(),
            digest: "page-digest".into(),
        }],
        vec![],
        "policy-digest",
        "runtime-version",
    )
}

fn state_id() -> HybridStateId {
    manifest().state_id().expect("manifest should be valid")
}

fn make_lease(rights: LeaseRights, ttl_ms: Option<u64>) -> StateLease {
    let principal = Principal::new("agent-1", "tenant-a").unwrap();
    let scope = ExecutionScope::new("run-1", "node-1", 0).unwrap();
    StateLease::new(principal, scope, &state_id(), rights, ttl_ms, 0, 42).unwrap()
}

#[test]
fn lease_authorize_enforces_rights_namespace_state() {
    let principal = Principal::new("agent-1", "tenant-a").unwrap();
    let state_id = state_id();
    let lease = make_lease(LeaseRights::empty().with(LeaseRight::Inspect), Some(1000));

    assert!(lease
        .authorize(
            LeaseRight::Inspect,
            lease.expires_unix_ms.unwrap() - 1,
            0,
            state_id.as_str(),
            &principal,
        )
        .is_ok());

    assert!(lease
        .authorize(
            LeaseRight::Append,
            lease.expires_unix_ms.unwrap() - 1,
            0,
            state_id.as_str(),
            &principal,
        )
        .is_err());

    let wrong_state = {
        let mut m = manifest();
        m.tokenizer_id = "tokenizer-b".into();
        m.state_id().unwrap()
    };

    assert!(lease
        .authorize(
            LeaseRight::Inspect,
            lease.expires_unix_ms.unwrap() - 1,
            0,
            wrong_state.as_str(),
            &principal,
        )
        .is_err());

    let wrong_principal = Principal::new("agent-2", "tenant-a").unwrap();
    assert!(lease
        .authorize(
            LeaseRight::Inspect,
            lease.expires_unix_ms.unwrap() - 1,
            0,
            state_id.as_str(),
            &wrong_principal,
        )
        .is_err());
}

#[test]
fn lease_expiry_and_revocation_are_authorization_denies() {
    let principal = Principal::new("agent-1", "tenant-a").unwrap();
    let scope = ExecutionScope::new("run-1", "node-1", 0).unwrap();
    let id = state_id();

    let lease = StateLease::new(
        principal.clone(),
        scope.clone(),
        &id,
        LeaseRights::all(),
        Some(1),
        4,
        0,
    )
    .unwrap();

    let near_expiry = lease
        .expires_unix_ms
        .expect("finite expiry with ttl");
    assert!(
        lease
            .authorize(
                LeaseRight::Inspect,
                near_expiry,
                4,
                id.as_str(),
                &principal,
            )
            .is_err()
    );

    let past = near_expiry + 1;
    let fresh = StateLease::new(principal, scope, &id, LeaseRights::all(), None, 10, 0).unwrap();
    assert!(fresh
        .authorize(LeaseRight::Fork, past, 11, id.as_str(), &fresh.principal)
        .is_err());
}

#[test]
fn lease_id_is_prefixed_and_digest_is_stable() {
    let lease = make_lease(LeaseRights::all(), None);
    assert!(lease.lease_id.starts_with("lease-v1:"));
    let digest1 = lease.digest();
    let digest2 = lease.digest();
    assert_eq!(digest1.len(), 64);
    assert_eq!(digest1, digest2);
}

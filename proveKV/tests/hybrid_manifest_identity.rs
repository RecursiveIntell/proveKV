use provekv::{
    AttentionType, HybridComponent, HybridPageRef, HybridStateManifestV1, KvTensorShape,
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
        vec![HybridComponent {
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

#[test]
fn identity_and_serialization_are_deterministic() {
    let a = manifest();
    let b = manifest();
    assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
    assert_eq!(a.digest().unwrap(), b.digest().unwrap());
    assert_eq!(a.state_id().unwrap(), b.state_id().unwrap());
}

#[test]
fn identity_changes_for_model_layout_and_shape() {
    let base = manifest();
    for mutate in [
        |m: &mut HybridStateManifestV1| m.model_id = "model-b".into(),
        |m: &mut HybridStateManifestV1| m.tokenizer_id = "tokenizer-b".into(),
        |m: &mut HybridStateManifestV1| m.shape.head_dim = 8,
        |m: &mut HybridStateManifestV1| m.component_inventory[0].version = "2".into(),
        |m: &mut HybridStateManifestV1| m.page_refs[0].digest = "other-page".into(),
    ] {
        let mut changed = base.clone();
        mutate(&mut changed);
        assert_ne!(base.state_id().unwrap(), changed.state_id().unwrap());
        assert!(base.state_id().unwrap().verify_manifest(&changed).is_err());
    }
}

#[test]
fn invalid_shape_is_rejected() {
    let mut m = manifest();
    m.shape.num_layers = 0;
    assert!(m.state_id().is_err());
}

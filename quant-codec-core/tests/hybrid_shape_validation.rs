use quant_codec_core::*;

fn fixture() -> HybridCacheLayoutV1 {
    let shape = KvCacheShapeV2::gqa(
        1,
        2,
        4,
        2,
        16,
        32,
        KvLayout::LayersHeadsTokensDim,
        DType::F16,
    )
    .unwrap();
    let profile = CodecProfileDigest::from_canonical_bytes(b"hybrid-profile-v1");
    HybridCacheLayoutV1::new(
        ModelFingerprint::new("model-a").unwrap(),
        TokenizerFingerprint::new("tokenizer-a").unwrap(),
        vec![HybridComponentLayoutV1 {
            kind: HybridComponentKind::AttentionKv,
            layer: LayerId(0),
            axes: vec![
                HybridAxis::Layer,
                HybridAxis::Head,
                HybridAxis::Token,
                HybridAxis::Feature,
            ],
            shape,
            sequence: HybridSequenceSemantics::TokenIndexed,
            role: Some(KvRole::Key),
            codec_profile_digest: profile,
        }],
    )
    .unwrap()
}

#[test]
fn serde_roundtrip_preserves_validated_contract() {
    let layout = fixture();
    let encoded = serde_json::to_vec(&layout).unwrap();
    let decoded: HybridCacheLayoutV1 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, layout);
    decoded.validate().unwrap();
}

#[test]
fn canonical_bytes_and_digest_are_stable() {
    let layout = fixture();
    let bytes_a = layout.canonical_bytes().unwrap();
    let bytes_b = layout.canonical_bytes().unwrap();
    assert_eq!(bytes_a, bytes_b);
    assert_eq!(
        layout.digest().unwrap(),
        ArtifactDigest::from_canonical_bytes(&bytes_a)
    );
    assert_eq!(layout.digest().unwrap(), fixture().digest().unwrap());
}

#[test]
fn invalid_shapes_and_contract_order_are_rejected() {
    let mut layout = fixture();
    layout.components[0].shape.seq_len = 0;
    assert!(layout.validate().is_err());

    let mut layout = fixture();
    layout.components[0].axes = vec![HybridAxis::Token, HybridAxis::Layer];
    assert!(layout.validate().is_err());

    let mut layout = fixture();
    layout.components[0].role = None;
    assert!(layout.validate().is_err());
}

#[test]
fn duplicate_or_unsorted_components_are_rejected() {
    let mut layout = fixture();
    layout.components.push(layout.components[0].clone());
    assert!(layout.validate().is_err());
}

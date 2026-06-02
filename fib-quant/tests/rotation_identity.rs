use fib_quant::{StoredRotation, ROTATION_ALGORITHM_VERSION, ROTATION_SCHEMA};

#[test]
fn same_rotation_identity_has_same_digest() {
    let a = StoredRotation::new(8, 91).unwrap();
    let b = StoredRotation::new(8, 91).unwrap();
    assert_eq!(a.rotation_schema(), ROTATION_SCHEMA);
    assert_eq!(a.algorithm_version(), ROTATION_ALGORITHM_VERSION);
    assert_eq!(a.digest().unwrap(), b.digest().unwrap());
}

#[test]
fn rotation_digest_changes_with_seed() {
    let a = StoredRotation::new(8, 91).unwrap();
    let b = StoredRotation::new(8, 92).unwrap();
    assert_ne!(a.digest().unwrap(), b.digest().unwrap());
}

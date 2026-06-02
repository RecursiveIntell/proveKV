use fib_quant::bitpack::{pack_indices, unpack_indices};

#[test]
fn bitpack_roundtrips_non_power_of_two_width() {
    let indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 3, 1];
    let packed = pack_indices(&indices, 3).unwrap();
    assert_eq!(unpack_indices(&packed, indices.len(), 3).unwrap(), indices);
}

#[test]
fn bitpack_rejects_nonzero_padding() {
    let indices = vec![1, 2, 3];
    let mut packed = pack_indices(&indices, 2).unwrap();
    packed[0] |= 0b0100_0000;
    assert!(unpack_indices(&packed, indices.len(), 2).is_err());
}

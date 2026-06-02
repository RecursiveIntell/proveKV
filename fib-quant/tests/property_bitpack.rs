use fib_quant::bitpack::{pack_indices, unpack_indices};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn packed_indices_roundtrip(width in 1u8..=16, count in 0usize..128) {
        let max = (1u32 << width) - 1;
        let indices: Vec<u32> = (0..count)
            .map(|idx| ((idx as u32).wrapping_mul(17).wrapping_add(u32::from(width))) & max)
            .collect();
        let packed = pack_indices(&indices, width)?;
        let unpacked = unpack_indices(&packed, count, width)?;
        prop_assert_eq!(unpacked, indices);
    }

    #[test]
    fn nonzero_padding_rejects(width in 1u8..=15, count in 1usize..128) {
        let indices = vec![0u32; count];
        let total_bits = count * width as usize;
        prop_assume!(total_bits % 8 != 0);
        let mut packed = pack_indices(&indices, width)?;
        let padding_pos = total_bits;
        packed[padding_pos / 8] |= 1 << (padding_pos % 8);
        prop_assert!(unpack_indices(&packed, count, width).is_err());
    }
}

#!/bin/bash
# prove_audit.sh — automated F1-F8 audit checks. Run from the workspace root.
# This is the artifact the F1-F8 audit demanded. If any check fails,
# the repo is not release-proof.

set -euo pipefail

echo "=== F6: workspace-local crates must resolve from path ==="
cargo tree -p provekv -e normal 2>&1 | grep -E "fib-quant|turbo-quant|gpu-backend" | while read line; do
    if ! echo "$line" | grep -q "$(pwd)/"; then
        echo "FAIL: $line is not a workspace path"
        exit 1
    fi
done
echo "OK: all local crates resolve from workspace"

echo ""
echo "=== F5: codebook/rotation digests must be non-empty in receipts ==="
cargo test --release -p provekv --lib pool::tests::test_pool_receipt_has_real_codebook_and_rotation_digests 2>&1 | tail -2

echo ""
echo "=== F4: shell manifest must not lie about its codec ==="
cargo test --release -p provekv --lib manifest::tests 2>&1 | tail -2

echo ""
echo "=== CLAIMS.json schema ==="
python3 -c "
import json, sys
c = json.load(open('CLAIMS.json'))
assert c['schema_version'] == '1.0.0', f\"unexpected schema: {c['schema_version']}\"
for name, claim in c['claims'].items():
    assert 'receipts' in claim, f\"{name}: no receipts\"
    assert 'claim_status' in claim, f\"{name}: no claim_status\"
    # F1/F2 audit: if the claim has raw/compressed bytes, the ratio_vs_f32_raw
    # and ratio_vs_fp16_kv must be derivable from those bytes, not hand-edited.
    if 'raw_total_bytes' in claim and 'compressed_total_bytes' in claim:
        derived_f32 = claim['raw_total_bytes'] / claim['compressed_total_bytes']
        derived_fp16 = derived_f32 / 2
        declared_f32 = claim['ratio_vs_f32_raw']
        declared_fp16 = claim['ratio_vs_fp16_kv']
        if abs(derived_f32 - declared_f32) > 0.001:
            raise AssertionError(f\"{name}: ratio_vs_f32_raw {declared_f32} != derived {derived_f32:.4f}\")
        if abs(derived_fp16 - declared_fp16) > 0.001:
            raise AssertionError(f\"{name}: ratio_vs_fp16_kv {declared_fp16} != derived {derived_fp16:.4f}\")
print(f'OK: {len(c[\"claims\"])} claims validated, all byte-derived ratios are consistent')
"

echo ""
echo "=== Cargo.lock is committed and resolves cleanly ==="
test -f Cargo.lock || { echo "FAIL: Cargo.lock missing"; exit 1; }
cargo build --release --workspace --locked 2>&1 | tail -3
echo ""
echo "=== Full test suite ==="
cargo test --release --workspace --lib 2>&1 | tail -3
echo ""
echo "=== Wall-clock decode bench (sanity) ==="
if [ -x target/release/examples/decode_wallclock ]; then
    # 2 reps, just to confirm the binary works locally. The real receipt
    # is in results/bench/decode_wallclock/ and was generated on both
    # fedora-43 and msi.
    target/release/examples/decode_wallclock 2 2>&1 | tail -1
else
    cargo build --release -p turbo-quant --example decode_wallclock 2>&1 | tail -1
    target/release/examples/decode_wallclock 2 2>&1 | tail -1
fi
echo ""
echo "All audit gates passed."

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
print(f\"OK: {len(c['claims'])} claims validated\")
"

echo ""
echo "=== Cargo.lock is committed and resolves cleanly ==="
test -f Cargo.lock || { echo "FAIL: Cargo.lock missing"; exit 1; }
cargo build --release --workspace --locked 2>&1 | tail -3
echo ""
echo "=== Full test suite ==="
cargo test --release --workspace --lib 2>&1 | tail -3
echo ""
echo "All audit gates passed."

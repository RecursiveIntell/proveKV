#!/usr/bin/env python3
from pathlib import Path
import sys, json, re
root = Path.cwd(); errors=[]; warnings=[]
required_files = [
 'src/kv/mod.rs','src/kv/shape.rs','src/kv/layout.rs','src/kv/profile.rs','src/kv/policy.rs',
 'src/kv/block.rs','src/kv/page.rs','src/kv/codec.rs','src/kv/receipt.rs','src/kv/quality.rs',
 'src/kv/attention_ref.rs','docs/kv/KV_PRODUCTION_READINESS_REPORT.md','docs/kv/KV_FINAL_AUDITOR_HANDOFF.md'
]
for f in required_files:
    if not (root/f).exists(): errors.append(f'missing required file {f}')
# scan for required symbols
symbols = ['KvTensorShapeV1','KvCompressionProfileV1','KvEncodedPageV1','KvCompressionReceiptV1','KvRole','KvRopeState']
all_rs = '\n'.join(p.read_text(errors='ignore') for p in (root/'src').rglob('*.rs')) if (root/'src').exists() else ''
for s in symbols:
    if s not in all_rs: errors.append(f'missing symbol {s}')
# package receipt checks
receipt_dir = root/'target/kv-production-receipts/final'
if not receipt_dir.exists(): warnings.append('missing target/kv-production-receipts/final receipts directory')
print(json.dumps({'errors':errors,'warnings':warnings}, indent=2))
sys.exit(1 if errors else 0)

#!/usr/bin/env python3
from pathlib import Path
import sys

errors = []
pool = Path('crates/proveKV/src/pool.rs')
manifest = Path('crates/proveKV/src/manifest.rs')
memory = Path('crates/proveKV/src/memory.rs')
texts = {p: p.read_text(encoding='utf-8', errors='ignore') for p in [pool, manifest, memory] if p.exists()}

if pool.exists() and 'estimate_manifest_bytes' in texts[pool]:
    errors.append('pool.rs still uses estimate_manifest_bytes; replace with canonical serialized byte accounting')

combined = '\n'.join(texts.values())
for term in ['realized_encoded_bytes', 'metadata_bytes', 'ideal_codec_bits_per_scalar']:
    if term not in combined:
        errors.append(f'missing accounting field or method: {term}')

if 'active_reader_scratch' not in combined and 'active_scratch' not in combined:
    errors.append('active reader scratch bytes are not explicitly tracked')

if errors:
    print('realized accounting findings:')
    for e in errors:
        print(' -', e)
    sys.exit(1)
print('realized accounting check ok')

#!/usr/bin/env python3
from pathlib import Path
import re, sys

errors = []
forbidden_dirs = [
    'crates/quant-governor',
    'crates/scr-runtime-compression',
    'crates/semantic-memory-compression',
]
for d in forbidden_dirs:
    if Path(d).exists():
        errors.append(f'forbidden out-of-scope directory present: {d}')

# Rust core must not depend on PyO3/maturin/python-only packages.
for cargo in [Path('crates/quant-codec-core/Cargo.toml'), Path('crates/proveKV/Cargo.toml')]:
    if cargo.exists():
        text = cargo.read_text(encoding='utf-8', errors='ignore')
        for token in ['pyo3', 'numpy', 'torch', 'maturin']:
            if re.search(rf'(?i)\b{re.escape(token)}\b', text):
                errors.append(f'{cargo} contains Python/binding dependency token {token}; core crates must stay clean')

# proveKV must not contain local Turbo/Fib algorithm implementation names beyond adapter stubs/docs.
for p in Path('crates/proveKV/src').rglob('*.rs') if Path('crates/proveKV/src').exists() else []:
    text = p.read_text(encoding='utf-8', errors='ignore')
    if 'fwht' in text.lower() or 'hadamard' in text.lower() or 'lloyd' in text.lower():
        errors.append(f'{p} appears to contain local value-codec algorithm math; inspect for duplicate Turbo/Fib implementation')

if errors:
    print('boundary drift findings:')
    for e in errors:
        print(' -', e)
    sys.exit(1)
print('boundary drift check ok')

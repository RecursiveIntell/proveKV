#!/usr/bin/env python3
from pathlib import Path
import sys, re

required_terms = {
    'crates/proveKV/src/receipts.rs': [
        'PoolBuildReceipt', 'ReaderInjectionReceipt', 'FallbackReceipt', 'DecodeReceipt', 'CompressionEvalReceipt'
    ],
    'crates/proveKV/src/pool.rs': [
        'CompressionEvalReceipt', 'FallbackReceipt', 'DecodeReceipt'
    ],
}
errors = []
for file, terms in required_terms.items():
    p = Path(file)
    if not p.exists():
        errors.append(f'missing {file}')
        continue
    text = p.read_text(encoding='utf-8', errors='ignore')
    for term in terms:
        if term not in text:
            errors.append(f'{file} missing receipt term {term}')

# Stronger next-pass expectations; warn/fail if absent.
next_terms = ['full_block_decoded', 'decoded_full_values', 'returned_values']
receipt_text = Path('crates/proveKV/src/receipts.rs').read_text(encoding='utf-8', errors='ignore') if Path('crates/proveKV/src/receipts.rs').exists() else ''
for term in next_terms:
    if term not in receipt_text:
        errors.append(f'DecodeReceiptV1 missing next-pass field {term}')

if errors:
    print('receipt integrity findings:')
    for e in errors:
        print(' -', e)
    sys.exit(1)
print('receipt integrity check ok')

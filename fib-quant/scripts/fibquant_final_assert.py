#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--pre', action='store_true', help='pre-run mode: report blockers but do not require receipts')
args = parser.parse_args()
ROOT = Path.cwd()
findings = []

def add(sev, code, msg, path=None):
    findings.append({'severity': sev, 'code': code, 'message': msg, 'path': str(path) if path else None})

def text(path):
    p = ROOT / path
    return p.read_text(encoding='utf-8', errors='replace') if p.exists() else ''

def exists(path): return (ROOT / path).exists()

def receipt_contains(path, patterns):
    data = text(path)
    return all(p in data for p in patterns)

# Required files
for p in ['Cargo.toml','README.md','LICENSE','CITATION.cff','CHANGELOG.md','RELEASE_CHECKLIST.md']:
    if not exists(p): add('blocker','MISSING_FILE',f'{p} missing',p)
for p in ['docs/compression/FIBQUANT_MATH_CONFORMANCE.md','docs/compression/FIBQUANT_PUBLICATION_NONCLAIMS.md','docs/compression/FIBQUANT_BENCHMARK_PLAN.md']:
    if not exists(p): add('blocker','MISSING_DOC',f'{p} missing',p)

# Source checks
cargo = text('Cargo.toml')
if 'CITATION.cff' not in cargo: add('blocker','CITATION_NOT_INCLUDED','Cargo include list must contain CITATION.cff','Cargo.toml')
if 'workspace = true' in cargo: add('blocker','WORKSPACE_INHERITANCE','workspace inheritance remains','Cargo.toml')

codec = text('src/codec.rs')
if re.search(r'fn\s+encode_norm\s*\([^)]*\)\s*->\s*Vec\s*<\s*u8\s*>', codec):
    add('blocker','ENCODE_NORM_VEC','encode_norm must return Result<Vec<u8>>','src/codec.rs')
for token in ['rotation_digest','CODE_SCHEMA','encoded_digest']:
    if token not in codec: add('blocker','CODEC_TOKEN_MISSING',f'{token} missing from codec path','src/codec.rs')

profile = text('src/profile.rs')
for token in ['MAX_AMBIENT_DIM','MAX_BLOCK_DIM','MAX_CODEBOOK_SIZE','checked_mul','ambient_dim == block_dim','rotation_algorithm_version']:
    if token not in profile: add('blocker','PROFILE_TOKEN_MISSING',f'{token} missing from profile','src/profile.rs')

if 'ResourceLimitExceeded' not in text('src/error.rs'):
    add('blocker','NO_RESOURCE_ERROR','ResourceLimitExceeded missing','src/error.rs')

rotation = text('src/rotation.rs')
for token in ['ROTATION_ALGORITHM_VERSION','fn digest','rotation_schema']:
    if token not in rotation: add('blocker','ROTATION_TOKEN_MISSING',f'{token} missing from rotation','src/rotation.rs')

lloyd = text('src/lloyd.rs')
if 'jitter' in lloyd:
    add('blocker','LLOYD_JITTER','jitter-based empty-cell repair remains','src/lloyd.rs')
if 'LloydRepairEvent' not in lloyd and 'repair_events' not in lloyd:
    add('high','NO_REPAIR_EVENTS','repair event details missing','src/lloyd.rs')

# Tests and benches
for p in ['tests/property_codec.rs','tests/property_bitpack.rs','benches/encode_decode.rs','deny.toml']:
    if not exists(p): add('high','MISSING_VERIFY_SURFACE',f'{p} missing',p)

if args.pre:
    out = {'mode':'pre','ok': not any(f['severity']=='blocker' for f in findings), 'findings': findings}
    print(json.dumps(out, indent=2))
    sys.exit(1 if any(f['severity']=='blocker' for f in findings) else 0)

# Receipt checks
required_receipts = ['fmt.txt','test.txt','clippy.txt','examples.txt','doc.txt','package-list.txt','package.txt','publish-dry-run.txt','static-audit.txt']
for r in required_receipts:
    if not exists(f'target/release-receipts/{r}'):
        add('blocker','MISSING_RECEIPT',f'target/release-receipts/{r} missing',f'target/release-receipts/{r}')

pkg_list = text('target/release-receipts/package-list.txt')
if pkg_list:
    if 'CITATION.cff' not in pkg_list: add('blocker','CITATION_NOT_IN_PACKAGE','cargo package list missing CITATION.cff','target/release-receipts/package-list.txt')
    for forbidden in ['z.py','OPERATOR_PASTE_FIRST.md','.codex/','overlays/','.agents/','fib-quant-generic-rust-next-codex-context']:
        if forbidden in pkg_list:
            add('blocker','FORBIDDEN_PACKAGE_FILE',f'cargo package list contains {forbidden}','target/release-receipts/package-list.txt')

publish_dry = text('target/release-receipts/publish-dry-run.txt')
if publish_dry and not ('dry run' in publish_dry.lower() or 'aborting upload' in publish_dry.lower() or 'warning: aborting upload due to dry run' in publish_dry.lower()):
    add('blocker','DRY_RUN_UNCLEAR','publish-dry-run receipt does not clearly show dry-run behavior','target/release-receipts/publish-dry-run.txt')

if exists('docs/compression/FIBQUANT_FINAL_RELEASE_DECISION.md'):
    decision = text('docs/compression/FIBQUANT_FINAL_RELEASE_DECISION.md')
    if 'cargo publish' in decision and 'NO actual cargo publish' not in decision and 'Actual publish performed: NO' not in decision:
        add('blocker','PUBLISH_AMBIGUITY','final decision must state no actual publish was performed','docs/compression/FIBQUANT_FINAL_RELEASE_DECISION.md')
else:
    add('blocker','MISSING_FINAL_DECISION','FIBQUANT_FINAL_RELEASE_DECISION.md missing','docs/compression/FIBQUANT_FINAL_RELEASE_DECISION.md')

out = {'mode':'final','ok': not any(f['severity']=='blocker' for f in findings), 'findings': findings}
Path('target/release-receipts').mkdir(parents=True, exist_ok=True)
Path('target/release-receipts/final-assert.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
if any(f['severity']=='blocker' for f in findings): sys.exit(1)

#!/usr/bin/env python3
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()

findings = []

def add(sev, code, msg, path=None):
    findings.append({'severity': sev, 'code': code, 'message': msg, 'path': str(path) if path else None})

def text(path):
    p = ROOT / path
    return p.read_text(encoding='utf-8') if p.exists() else ''

def exists(path):
    return (ROOT / path).exists()

if 'name = "fib-quant"' not in text('Cargo.toml'):
    add('blocker','NOT_FIB_QUANT','Cargo.toml does not identify fib-quant','Cargo.toml')

if not exists('CITATION.cff'):
    add('blocker','MISSING_CITATION','CITATION.cff missing from repo root','CITATION.cff')

cargo = text('Cargo.toml')
if 'CITATION.cff' not in cargo:
    add('blocker','CITATION_NOT_IN_CARGO_INCLUDE','Cargo include list does not mention CITATION.cff','Cargo.toml')
if 'workspace = true' in cargo:
    add('blocker','WORKSPACE_INHERITANCE','Cargo.toml still has workspace inheritance','Cargo.toml')
if 'z.py' in cargo and 'exclude' not in cargo:
    add('high','ZPY_RISK','Cargo.toml mentions z.py without clear exclusion policy','Cargo.toml')

codec = text('src/codec.rs')
if re.search(r'fn\s+encode_norm\s*\([^)]*\)\s*->\s*Vec\s*<\s*u8\s*>', codec):
    add('blocker','ENCODE_NORM_FAIL_OPEN','encode_norm returns Vec<u8>, should return Result<Vec<u8>>','src/codec.rs')
if 'f16::from_f32' in codec and 'is_finite' not in codec[codec.find('fn encode_norm'):codec.find('fn decode_norm') if 'fn decode_norm' in codec else len(codec)]:
    add('blocker','ENCODE_NORM_NO_CONVERSION_CHECK','encode_norm appears not to validate fp16/f32 conversion','src/codec.rs')
if 'rotation_digest' not in codec:
    add('high','CODE_MISSING_ROTATION_DIGEST','FibCodeV1/codec path appears to lack rotation_digest','src/codec.rs')

profile = text('src/profile.rs')
if 'ResourceLimitExceeded' not in text('src/error.rs'):
    add('blocker','NO_RESOURCE_ERROR','ResourceLimitExceeded error missing','src/error.rs')
if 'MAX_AMBIENT_DIM' not in profile or 'checked_mul' not in profile:
    add('blocker','RESOURCE_BOUNDS_WEAK','Profile resource bounds/checked arithmetic missing or weak','src/profile.rs')
if 'ambient_dim == block_dim' not in profile:
    add('blocker','D_EQ_K_NOT_ENFORCED','d == k alpha law not enforced in profile validation','src/profile.rs')
if 'rotation_algorithm_version' not in profile:
    add('high','PROFILE_MISSING_ROTATION_VERSION','profile lacks rotation_algorithm_version','src/profile.rs')

rotation = text('src/rotation.rs')
if 'ROTATION_ALGORITHM_VERSION' not in rotation or 'fn digest' not in rotation:
    add('high','ROTATION_IDENTITY_WEAK','StoredRotation lacks algorithm version/digest','src/rotation.rs')

lloyd = text('src/lloyd.rs')
if 'jitter' in lloyd:
    add('high','LLOYD_JITTER_REPAIR','Lloyd empty-cell repair still appears to use jitter','src/lloyd.rs')
if 'repair_events' not in lloyd and 'LloydRepairEvent' not in lloyd:
    add('medium','LLOYD_REPAIR_EVENTS_MISSING','Lloyd repair events/details missing','src/lloyd.rs')

if not exists('tests/property_codec.rs') and not exists('tests/property_bitpack.rs'):
    add('high','NO_PROPERTY_TESTS','No proptest property test files found','tests')
if 'proptest' not in cargo:
    add('high','NO_PROPTEST_DEP','Cargo.toml lacks proptest dev-dependency','Cargo.toml')
if not exists('benches/encode_decode.rs'):
    add('medium','NO_CRITERION_BENCH','benches/encode_decode.rs missing','benches/encode_decode.rs')
if 'criterion' not in cargo:
    add('medium','NO_CRITERION_DEP','Cargo.toml lacks criterion dev-dependency','Cargo.toml')
if not exists('deny.toml'):
    add('medium','NO_CARGO_DENY','deny.toml missing','deny.toml')

root_bad = [p.name for p in ROOT.glob('0*_*.md')] + [p.name for p in ROOT.glob('OPERATOR_PASTE_FIRST.md')]
if root_bad:
    add('medium','ROOT_CODEX_FILES','Root Codex/workbench markdown still present: '+', '.join(root_bad[:10]),'.')
if exists('z.py'):
    add('medium','ZPY_PRESENT','z.py present in repo root; acceptable only if not public-release branch and not Cargo package','z.py')

out = {'ok': not any(f['severity']=='blocker' for f in findings), 'findings': findings}
Path('target/release-receipts').mkdir(parents=True, exist_ok=True)
(Path('target/release-receipts/static-audit.json')).write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
if any(f['severity']=='blocker' for f in findings):
    sys.exit(1)

#!/usr/bin/env python3
"""Verify the vendored macro repair against preserved upstream file hashes."""
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / 'vendor/SOURCE_PROVENANCE.json').read_text())
    assert manifest['schema'] == 'VendoredMacroRepairV1'
    for package in manifest['packages']:
        base = ROOT / package['vendored_path']
        for name, expected in package['upstream_files'].items():
            data = (base / name).read_bytes()
            if name == 'Cargo.toml':
                current = tomllib.loads(data.decode())
                assert current['package']['version'] == package['version']
                assert current['dependencies']['paste'] == {'version': '=0.2.3', 'package': 'pastey'}
                old = b'[dependencies.paste]\nversion = "=0.2.3"\npackage = "pastey"'
                new = b'[dependencies.paste]\nversion = "1.0"'
                assert data.count(old) == 1
                data = data.replace(old, new)
            assert hashlib.sha256(data).hexdigest() == expected, f'Unexpected change: {base/name}'
        actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
        assert actual == set(package['upstream_files']), f'Unexpected vendor files: {base}'
    print(json.dumps({'verified': True, 'packages': len(manifest['packages']), 'numerical_source_changes': 0}))


if __name__ == '__main__':
    verify()

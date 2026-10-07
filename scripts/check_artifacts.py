#!/usr/bin/env python3
"""Verify frozen artifact hashes and the supplied baseline ELF/profile association."""
import json
from pathlib import Path
import sys

from provenance import digest

root = Path(__file__).resolve().parents[1]
errors = []
manifest = json.loads((root / 'artifacts/manifest.json').read_text())
for item in manifest['artifacts']:
    path = root / item['path']
    if not path.is_file() or digest(path) != item['sha256']:
        errors.append(item['path'])
    else:
        print('OK:', item['path'])
run = root / 'results/block0_qproj/runs/20261007-imported-b0'
profile = json.loads((run / 'profile.json').read_text())
if digest(run / 'qproj_test.elf') != profile['elf_sha256']:
    errors.append('Imported baseline ELF/profile mismatch')
for key in ['scalar_macs_match_expected', 'requantizations_match_expected_outputs']:
    if profile.get(key) is not True:
        errors.append(key)
if errors:
    print('Verification failed:', ', '.join(errors), file=sys.stderr)
    sys.exit(1)
print('Frozen artifacts and imported profile linkage verified.')

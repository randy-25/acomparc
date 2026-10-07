#!/usr/bin/env python3
"""List saved runs using manifests, rather than inferring results from filenames."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print('experiment\trun\tstatus\tinstructions\tMACs\tcommit')
for path in sorted((root / 'results').glob('*/runs/*/run.json')):
    run = json.loads(path.read_text())
    metrics = run.get('metrics', {})
    commit = (run.get('git', {}).get('commit') or 'unknown')[:12]
    print('\t'.join(str(x) for x in [run['experiment_id'], run['run_id'], run['status'],
          metrics.get('kernel_instructions', '-'), metrics.get('macs', '-'), commit]))

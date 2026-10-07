#!/usr/bin/env python3
"""Create an empty experiment without inheriting old measurements or workload assumptions."""
import argparse
import json
from pathlib import Path
import shutil

from provenance import identifier, json_write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id')
    parser.add_argument('--purpose', required=True)
    args = parser.parse_args()
    exp_id = identifier(args.id)
    destination = ROOT / 'experiments' / exp_id
    destination.mkdir(exist_ok=False)
    config = json.loads((ROOT / 'docs/experiment_template/experiment.json').read_text())
    config['id'] = exp_id
    config['purpose'] = args.purpose
    json_write(destination / 'experiment.json', config)
    template = (ROOT / 'docs/experiment_template/README.md').read_text()
    (destination / 'README.md').write_text(template.replace('{{ID}}', exp_id).replace('{{PURPOSE}}', args.purpose))
    print('Created:', destination)
    print('Fill in source, input hashes, workload, commands, and validation before running.')


if __name__ == '__main__':
    main()

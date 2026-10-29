#!/usr/bin/env python3
"""Export the frozen eight-block profile from the inspected TFLite models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))

import argparse
from pathlib import Path
from tinystories_integer.cli_common import MODELS,PROFILE,PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=MODELS)
    parser.add_argument('--profile', type=Path, default=PROFILE)
    parser.add_argument('--out-dir', type=Path, default=PROJECT/'build/parameter_export')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    from tinystories_integer.parameters import load_parameters
    from tinystories_integer.storage import save_archive
    from tinystories_integer.c_export import write_c_parameters
    # Check both output groups before writing either group.
    expected = ['parameters.json','parameters.npz','c/integer_parameter_types.h']
    expected += [f'c/block{i}_params.{ext}' for i in range(8) for ext in ('c','h')]
    if not args.overwrite and any((args.out_dir/name).exists() for name in expected):
        parser.error('Outputs already exist; choose a new directory or pass --overwrite')
    bundles,profile = load_parameters(args.models_dir,args.profile)
    save_archive(args.out_dir,bundles,profile,overwrite=args.overwrite)
    write_c_parameters(args.out_dir/'c',bundles,profile,overwrite=args.overwrite)
    for b in bundles:
        print(f'Block {b["block_index"]}: INT8 input scale={b["input_scale"]}, output scale={b["output_scale"]}')
    print(f'PASS: exported eight blocks to {args.out_dir}')
    print('C parameters are ready for the implementation under src/sw/.')


if __name__ == '__main__':
    main()

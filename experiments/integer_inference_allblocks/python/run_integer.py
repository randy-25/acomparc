#!/usr/bin/env python3
"""Run the NumPy integer blocks, without Torch or LiteRT inference."""
import argparse
from pathlib import Path
import numpy as np
from tinystories_integer.cli_common import run_chain,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--input-codes', type=Path, required=True)
    parser.add_argument('--input-key', default='inputs', help='Array key for NPZ input')
    parser.add_argument('--out', type=Path, required=True, help='New .npz result file')
    args = parser.parse_args()
    if args.out.suffix != '.npz':
        parser.error('--out must end in .npz')
    if args.out.exists() or args.out.with_suffix('.json').exists():
        parser.error('Result already exists; choose a new output path')
    from tinystories_integer.storage import load_archive
    bundles,profile = load_archive(args.params)
    if args.input_codes.suffix == '.npz':
        with np.load(args.input_codes,allow_pickle=False) as saved:
            inputs = saved[args.input_key].copy()
    else:
        inputs = np.load(args.input_codes,allow_pickle=False)
    outputs,pre_boundary,diagnostics = run_chain(inputs,bundles)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,block_outputs=outputs,pre_boundary_outputs=pre_boundary,output=outputs[:,-1])
    write_json(args.out.with_suffix('.json'),{'profile':profile['profile'],'diagnostics':diagnostics})
    print(f'Wrote {args.out} and {args.out.with_suffix(".json")}')


if __name__ == '__main__':
    main()

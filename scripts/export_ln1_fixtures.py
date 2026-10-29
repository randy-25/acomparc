#!/usr/bin/env python3
"""Export integer Python LN1 checkpoints for all blocks, without host ML tools."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from tinystories_integer.cli_common import PROJECT,write_json
from tinystories_integer.storage import load_archive
from tinystories_integer.c_export import write_c_parameters
from tinystories_integer.integer_reference import integer_ln1

FIELDS = [('input','int8_t',4096),('mean','int32_t',32),
          ('square_fine','int64_t',4096),('square','int32_t',4096),
          ('variance','int32_t',32),('variance_eps','int32_t',32),
          ('inv_std','int32_t',32),('normalized_fine','int64_t',4096),
          ('normalized','int32_t',4096),('affine','int64_t',4096),
          ('output','int8_t',4096)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=PROJECT/'build/parameter_export')
    parser.add_argument('--c-params-dir',type=Path,help='Default: sibling c/ directory beside parameters.json')
    parser.add_argument('--vectors',type=Path,default=PROJECT/'tests/sw/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=PROJECT/'build/fixtures/ln1')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    archive_path=args.params/'parameters.json' if args.params.is_dir() else args.params
    c_dir=args.c_params_dir or archive_path.parent/'c'
    outputs=['ln1_golden.h','ln1_case_names.c','manifest.json']+[f'block{i}_ln1_golden.c' for i in range(8)]
    if not args.overwrite and any((args.out_dir/name).exists() for name in outputs):
        parser.error('Fixtures exist; choose a new directory or pass --overwrite')
    bundles,profile=load_archive(archive_path)
    metadata=json.loads(args.vectors.with_suffix('.json').read_text())
    if (metadata['profile']!=profile['profile'] or metadata['model_sha256']!=profile['model_sha256']
            or metadata['notebook_sha256']!=profile['notebook_sha256']
            or digest(args.vectors)!=metadata['npz_sha256']):
        raise ValueError('Reference vectors do not match this validated profile')
    # Compare actual C data with a fresh rendering before creating test fixtures.
    # This writes only to a temporary directory, never to the user's C parameters.
    with tempfile.TemporaryDirectory(prefix='ln1-parameter-check-') as temporary:
        write_c_parameters(temporary,bundles,profile)
        for expected in Path(temporary).iterdir():
            actual=c_dir/expected.name
            if not actual.is_file() or actual.read_bytes()!=expected.read_bytes():
                raise ValueError(f'C data differs from the parameter archive: {actual}')
    with np.load(args.vectors,allow_pickle=False) as saved:
        inputs=saved['inputs'].copy()
        previous_outputs=saved['block_outputs'].copy()
    if inputs.dtype!=np.int8 or inputs.shape!=(8,1,32,128):
        raise ValueError('Expected eight validated INT8 input cases')
    if previous_outputs.dtype!=np.int8 or previous_outputs.shape!=(8,8,1,32,128):
        raise ValueError('Expected eight-block reference outputs for all eight cases')
    edge_cases=[np.zeros((1,32,128),np.int8),np.full((1,32,128),-128,np.int8),
                np.full((1,32,128),127,np.int8),
                np.broadcast_to(np.array([-128,127]*64,np.int8),(1,32,128)).copy(),
                np.zeros((1,32,128),np.int8)]
    edge_cases[-1][0,:,0]=127
    labels=['benchmark']+[f'calibration_{i}' for i in range(7)]+[
        'edge_zero','edge_all_min','edge_all_max','edge_alternating_extremes','edge_single_outlier']
    args.out_dir.mkdir(parents=True,exist_ok=True)
    header=['#ifndef TINYSTORIES_ALLBLOCKS_LN1_GOLDEN_H\n#define TINYSTORIES_ALLBLOCKS_LN1_GOLDEN_H\n',
            '#include <stdint.h>\n',f'#define LN1_CASE_COUNT {len(labels)}u\n',
            '#define LN1_REFERENCE_CASE_COUNT 8u\n#define LN1_TOKENS 32u\n#define LN1_FEATURES 128u\n#define LN1_ELEMENTS 4096u\n',
            'typedef struct {\n']
    header += [f'    {ctype} {name}[{size}];\n' for name,ctype,size in FIELDS]
    header += ['} Ln1Golden;\n','extern const char *const ln1_case_names[LN1_CASE_COUNT];\n']
    header += [f'extern const Ln1Golden block{i}_ln1_cases[LN1_CASE_COUNT];\n' for i in range(8)]
    header += ['#endif\n']
    (args.out_dir/'ln1_golden.h').write_text(''.join(header))
    (args.out_dir/'ln1_case_names.c').write_text('#include "ln1_golden.h"\nconst char *const ln1_case_names[LN1_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    for i,bundle in enumerate(bundles):
        real=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]
        trials=[]
        for input_codes in real+edge_cases:
            trials.append({'input':input_codes,**integer_ln1(input_codes,bundle['ln1'])})
        path=args.out_dir/f'block{i}_ln1_golden.c'
        with path.open('w') as stream:
            stream.write(f'/* Profile: {profile["profile"]}; block {i}. Integer Python checkpoints. */\n')
            stream.write('#include "ln1_golden.h"\n')
            stream.write(f'const Ln1Golden block{i}_ln1_cases[LN1_CASE_COUNT] = {{\n')
            for c,trial in enumerate(trials):
                stream.write(f'    {{ /* {labels[c]} */\n')
                for name,ctype,size in FIELDS:
                    values=np.asarray(trial[name]).reshape(-1)
                    if values.size!=size:
                        raise ValueError(f'Unexpected {name} checkpoint shape')
                    stream.write(f'        .{name} = {{\n')
                    for pos in range(0,size,16):
                        stream.write('            '+', '.join(str(int(v)) for v in values[pos:pos+16])+',\n')
                    stream.write('        },\n')
                stream.write('    },\n')
            stream.write('};\n')
        print(f'Block {i}: exported {len(trials)} LN1 checkpoint cases',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,
        'input_policy':'Eight actual integer-chain inputs per block, followed by five INT8 edge patterns.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in outputs if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 LN1 cases, 425984 expected INT8 outputs; C parameters verified')


if __name__=='__main__':
    main()

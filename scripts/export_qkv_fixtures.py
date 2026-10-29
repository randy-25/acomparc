#!/usr/bin/env python3
"""Export Q/K/V accumulators and outputs for standalone and LN1-chained tests."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from tinystories_integer.cli_common import PROJECT,write_json
from tinystories_integer.storage import load_archive
from tinystories_integer.c_export import write_c_parameters
from tinystories_integer.integer_reference import integer_ln1,integer_projection
from export_ln1_fixtures import digest


def patterns(zero_point):
    values=[np.full((1,32,128),zero_point,np.int8),
            np.full((1,32,128),-128,np.int8),np.full((1,32,128),127,np.int8),
            np.broadcast_to(np.array([-128,127]*64,np.int8),(1,32,128)).copy(),
            np.full((1,32,128),zero_point,np.int8)]
    values[-1][0,:,0]=127
    return values


def write_values(stream, name, values, indent):
    flat=np.asarray(values).reshape(-1)
    stream.write(indent+f'.{name} = {{\n')
    for pos in range(0,len(flat),16):
        stream.write(indent+'    '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
    stream.write(indent+'},\n')


def write_path(stream, name, trial):
    stream.write(f'        .{name} = {{\n')
    write_values(stream,'input',trial['input'],'            ')
    for field in ('accumulator','output'):
        stream.write(f'            .{field} = {{\n')
        for tag in ('Q','K','V'):
            flat=trial[tag][field].reshape(-1)
            stream.write(f'                {{ /* {tag} */\n')
            for pos in range(0,len(flat),16):
                stream.write('                    '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
            stream.write('                },\n')
        stream.write('            },\n')
    stream.write('        },\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=PROJECT/'build/parameter_export')
    parser.add_argument('--c-params-dir',type=Path)
    parser.add_argument('--vectors',type=Path,default=PROJECT/'tests/sw/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=PROJECT/'build/fixtures/qkv')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    names=['qkv_golden.h','qkv_case_names.c','manifest.json']+[f'block{i}_qkv_golden.c' for i in range(8)]
    if not args.overwrite and any((args.out_dir/name).exists() for name in names):
        parser.error('Fixtures exist; choose a new directory or pass --overwrite')
    archive_path=args.params/'parameters.json' if args.params.is_dir() else args.params
    c_dir=args.c_params_dir or archive_path.parent/'c'
    bundles,profile=load_archive(archive_path)
    metadata=json.loads(args.vectors.with_suffix('.json').read_text())
    if (metadata['profile']!=profile['profile'] or metadata['model_sha256']!=profile['model_sha256']
            or metadata['notebook_sha256']!=profile['notebook_sha256']
            or digest(args.vectors)!=metadata['npz_sha256']):
        raise ValueError('Reference vectors do not match this validated profile')
    with tempfile.TemporaryDirectory(prefix='qkv-parameter-check-') as temporary:
        write_c_parameters(temporary,bundles,profile)
        for expected in Path(temporary).iterdir():
            actual=c_dir/expected.name
            if not actual.is_file() or actual.read_bytes()!=expected.read_bytes():
                raise ValueError(f'C data differs from parameter archive: {actual}')
    with np.load(args.vectors,allow_pickle=False) as saved:
        inputs=saved['inputs'].copy()
        previous_outputs=saved['block_outputs'].copy()
    if inputs.dtype!=np.int8 or inputs.shape!=(8,1,32,128):
        raise ValueError('Expected eight validated INT8 input cases')
    if previous_outputs.dtype!=np.int8 or previous_outputs.shape!=(8,8,1,32,128):
        raise ValueError('Expected eight-block reference outputs')
    labels=['benchmark']+[f'calibration_{i}' for i in range(7)]+[
        'edge_centered_zero','edge_all_min','edge_all_max','edge_alternating_extremes','edge_single_outlier']
    args.out_dir.mkdir(parents=True,exist_ok=True)
    header='''#ifndef TINYSTORIES_ALLBLOCKS_QKV_GOLDEN_H
#define TINYSTORIES_ALLBLOCKS_QKV_GOLDEN_H
#include <stdint.h>
#define QKV_CASE_COUNT 13u
#define QKV_TOKENS 32u
#define QKV_FEATURES 128u
#define QKV_ELEMENTS 4096u
typedef struct {
    int8_t input[4096];
    int32_t accumulator[3][4096];
    int8_t output[3][4096];
} QkvPathGolden;
typedef struct {
    int8_t block_input[4096];
    QkvPathGolden local;
    QkvPathGolden chained;
} QkvGolden;
extern const char *const qkv_case_names[QKV_CASE_COUNT];
'''
    header+=''.join(f'extern const QkvGolden block{i}_qkv_cases[QKV_CASE_COUNT];\n' for i in range(8))+'#endif\n'
    (args.out_dir/'qkv_golden.h').write_text(header)
    (args.out_dir/'qkv_case_names.c').write_text('#include "qkv_golden.h"\nconst char *const qkv_case_names[QKV_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    for i,bundle in enumerate(bundles):
        qkv=bundle['qkv']
        for p in qkv.values():
            if p['weights'].shape!=(128,128) or p.get('bias') is not None or p['shift']!=39:
                raise ValueError('Expected three bias-free 128-to-128 projections with shift 39')
            if p['input_zero_point']!=bundle['ln1']['output_zero_point']:
                raise ValueError('LN1 and QKV quantization disagree')
        block_inputs=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]+patterns(0)
        ln1_outputs=[integer_ln1(x,bundle['ln1'])['output'] for x in block_inputs]
        local_inputs=ln1_outputs[:8]+patterns(qkv['Q']['input_zero_point'])
        with (args.out_dir/f'block{i}_qkv_golden.c').open('w') as stream:
            stream.write(f'/* Profile {profile["profile"]}, block {i}; integer Python reference. */\n')
            stream.write(f'#include "qkv_golden.h"\nconst QkvGolden block{i}_qkv_cases[QKV_CASE_COUNT] = {{\n')
            for c,block_input in enumerate(block_inputs):
                stream.write(f'    {{ /* {labels[c]} */\n')
                write_values(stream,'block_input',block_input,'        ')
                for name,values in [('local',local_inputs[c]),('chained',ln1_outputs[c])]:
                    trial={'input':values,**{tag:integer_projection(values,qkv[tag]) for tag in ('Q','K','V')}}
                    write_path(stream,name,trial)
                stream.write('    },\n')
            stream.write('};\n')
        print(f'Block {i}: exported 13 standalone and 13 LN1-chained QKV cases',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,
        'local_inputs':'Eight Python LN1 outputs from saved chain inputs, followed by five direct INT8 projection edge patterns.',
        'chained_inputs':'Thirteen block inputs pass through LN1 first; C LN1 must reproduce the resulting projection inputs.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in names if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 cases per path, 1277952 Q/K/V outputs per path; C parameters verified')


if __name__=='__main__':
    main()

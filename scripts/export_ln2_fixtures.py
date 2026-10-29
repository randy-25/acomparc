#!/usr/bin/env python3
"""Export LN2 checkpoints for standalone and C Parts-1-through-4 tests."""

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
from tinystories_integer.integer_reference import (
    integer_ln1,integer_ln2,integer_projection,integer_attention_scores,
    integer_causal_softmax,integer_attention_context,integer_out_residual,
)
from export_ln1_fixtures import digest
from export_qkv_fixtures import patterns
from export_attention_fixtures import complete_text_file

FIELDS=[('input','int32_t',4096),('mean','int32_t',32),
        ('square_fine','int64_t',4096),('square','int32_t',4096),
        ('variance','int32_t',32),('variance_eps','int32_t',32),
        ('inv_std','int32_t',32),('normalized_fine','int64_t',4096),
        ('normalized','int32_t',4096),('affine','int64_t',4096),('output','int8_t',4096)]


def direct_ln2_patterns():
    # All codes have the LN2 input scale 2^-16. Constants are 0, -1 and +1.
    values=[np.zeros((1,32,128),np.int32),np.full((1,32,128),-65536,np.int32),
            np.full((1,32,128),65536,np.int32),
            np.broadcast_to(np.array([-65536,65536]*64,np.int32),(1,32,128)).copy(),
            np.zeros((1,32,128),np.int32)]
    values[-1][0,:,0]=65536
    return values


def write_values(stream,name,values,indent):
    flat=np.asarray(values).reshape(-1)
    stream.write(indent+f'.{name} = {{\n')
    for pos in range(0,len(flat),16):
        stream.write(indent+'    '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
    stream.write(indent+'},\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=PROJECT/'build/parameter_export')
    parser.add_argument('--c-params-dir',type=Path)
    parser.add_argument('--vectors',type=Path,default=PROJECT/'tests/sw/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=PROJECT/'build/fixtures/ln2')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    names=['ln2_golden.h','ln2_case_names.c','manifest.json']+[f'block{i}_ln2_golden.c' for i in range(8)]
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
    with tempfile.TemporaryDirectory(prefix='ln2-parameter-check-') as temporary:
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
        'edge_zero','edge_constant_negative','edge_constant_positive','edge_alternating','edge_single_outlier']
    args.out_dir.mkdir(parents=True,exist_ok=True)
    header=['#ifndef TINYSTORIES_ALLBLOCKS_LN2_GOLDEN_H\n#define TINYSTORIES_ALLBLOCKS_LN2_GOLDEN_H\n',
            '#include <stdint.h>\n#define LN2_CASE_COUNT 13u\n#define LN2_TOKENS 32u\n',
            '#define LN2_FEATURES 128u\n#define LN2_ELEMENTS 4096u\n#define LN2_SCORE_ELEMENTS 16384u\n',
            'typedef struct {\n']
    header += [f'    {ctype} {name}[{size}];\n' for name,ctype,size in FIELDS]
    header += ['} Ln2PathGolden;\n','typedef struct {\n    int8_t block_input[4096];\n',
               '    Ln2PathGolden local;\n    Ln2PathGolden chained;\n} Ln2Golden;\n',
               'extern const char *const ln2_case_names[LN2_CASE_COUNT];\n']
    header += [f'extern const Ln2Golden block{i}_ln2_cases[LN2_CASE_COUNT];\n' for i in range(8)]
    header += ['#endif\n']
    (args.out_dir/'ln2_golden.h').write_text(''.join(header))
    (args.out_dir/'ln2_case_names.c').write_text('#include "ln2_golden.h"\nconst char *const ln2_case_names[LN2_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    summaries=[]
    for i,bundle in enumerate(bundles):
        p=bundle['ln2']
        if (p['square_shift']!=23 or p['normalize_shift']!=23 or p['epsilon_q']!=10737
                or p['rsqrt_numerator']!=1<<62):
            raise ValueError('Unexpected fixed LN2 numerical profile')
        block_inputs=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]+patterns(0)
        residuals=[]
        for block_input in block_inputs:
            ln1=integer_ln1(block_input,bundle['ln1'])['output']
            qkv={tag:integer_projection(ln1,bundle['qkv'][tag])['output'] for tag in ('Q','K','V')}
            scores=integer_attention_scores(qkv['Q'],qkv['K'],bundle['attention'])
            softmax=integer_causal_softmax(scores,bundle['attention'])
            attention=integer_attention_context(softmax['probabilities'],qkv['V'],bundle['attention'])['output']
            residuals.append(integer_out_residual(attention,block_input,bundle['out_residual'])['output'])
        local_inputs=residuals[:8]+direct_ln2_patterns()
        saturations={'local':0,'chained':0}
        path=args.out_dir/f'block{i}_ln2_golden.c'
        with complete_text_file(path) as stream:
            stream.write(f'/* Profile {profile["profile"]}, block {i}; checked integer Python LN2. */\n')
            stream.write(f'#include "ln2_golden.h"\nconst Ln2Golden block{i}_ln2_cases[LN2_CASE_COUNT] = {{\n')
            for c,block_input in enumerate(block_inputs):
                stream.write(f'    {{ /* {labels[c]} */\n')
                write_values(stream,'block_input',block_input,'        ')
                for name,values in [('local',local_inputs[c]),('chained',residuals[c])]:
                    result=integer_ln2(values,p)
                    trial={'input':values,**result}
                    saturations[name]+=result['saturation_count']
                    stream.write(f'        .{name} = {{\n')
                    for field,ctype,size in FIELDS:
                        flat=np.asarray(trial[field]).reshape(-1)
                        if len(flat)!=size:
                            raise ValueError(f'Unexpected LN2 checkpoint shape: {field}')
                        write_values(stream,field,flat,'            ')
                    stream.write('        },\n')
                stream.write('    },\n')
            stream.write('};\n')
        completed=path.read_text()
        if not completed.endswith('};\n') or completed.count('    { /* ')!=13:
            raise RuntimeError(f'Incomplete generated fixture: {path}')
        summaries.append({'block':i,'output_saturations':saturations})
        print(f'Block {i}: exported 13 standalone and 13 Parts1-4-chained LN2 checkpoint cases',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,'summary':summaries,
        'local_inputs':'Eight saved-chain Part-4 outputs, then direct INT32 codes for zero, constant -1, constant +1, alternating -1/+1 and one +1 feature at scale 2^-16.',
        'chained_inputs':'Thirteen block inputs pass through Python Parts 1-4. C must reproduce those residuals before LN2.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in names if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 cases per path, 425984 LN2 INT8 outputs per path; C parameters verified')


if __name__=='__main__':
    main()

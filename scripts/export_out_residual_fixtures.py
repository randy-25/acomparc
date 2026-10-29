#!/usr/bin/env python3
"""Export Part-4 projection and residual checkpoints for all eight blocks."""

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
    integer_ln1,integer_projection,integer_attention_scores,integer_causal_softmax,
    integer_attention_context,integer_out_residual,
)
from export_ln1_fixtures import digest
from export_qkv_fixtures import patterns
from export_attention_fixtures import complete_text_file

FIELDS=[('block_input','int8_t'),('ln1','int8_t'),('query','int8_t'),
        ('key','int8_t'),('value','int8_t'),('attention','int8_t'),
        ('accumulator','int32_t'),('projection','int8_t'),
        ('branch','int32_t'),('output','int32_t')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=PROJECT/'build/parameter_export')
    parser.add_argument('--c-params-dir',type=Path)
    parser.add_argument('--vectors',type=Path,default=PROJECT/'tests/sw/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=PROJECT/'build/fixtures/out_residual')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    names=['out_residual_golden.h','out_residual_case_names.c','manifest.json']+[
        f'block{i}_out_residual_golden.c' for i in range(8)]
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
    with tempfile.TemporaryDirectory(prefix='out-residual-parameter-check-') as temporary:
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
        'edge_zero','edge_all_min','edge_all_max','edge_alternating_extremes','edge_single_outlier']
    args.out_dir.mkdir(parents=True,exist_ok=True)
    header=['#ifndef TINYSTORIES_ALLBLOCKS_OUT_RESIDUAL_GOLDEN_H\n#define TINYSTORIES_ALLBLOCKS_OUT_RESIDUAL_GOLDEN_H\n',
            '#include <stdint.h>\n#define OUT_RESIDUAL_CASE_COUNT 13u\n',
            '#define OUT_RESIDUAL_TOKENS 32u\n#define OUT_RESIDUAL_FEATURES 128u\n#define OUT_RESIDUAL_ELEMENTS 4096u\n',
            '#define OUT_RESIDUAL_SCORE_ELEMENTS 16384u\n', 'typedef struct {\n']
    header += [f'    {ctype} {name}[4096];\n' for name,ctype in FIELDS]
    header += ['} OutResidualGolden;\n','extern const char *const out_residual_case_names[OUT_RESIDUAL_CASE_COUNT];\n']
    header += [f'extern const OutResidualGolden block{i}_out_residual_cases[OUT_RESIDUAL_CASE_COUNT];\n' for i in range(8)]
    header += ['#endif\n']
    (args.out_dir/'out_residual_golden.h').write_text(''.join(header))
    (args.out_dir/'out_residual_case_names.c').write_text('#include "out_residual_golden.h"\nconst char *const out_residual_case_names[OUT_RESIDUAL_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    summaries=[]
    for i,bundle in enumerate(bundles):
        p=bundle['out_residual']
        if (p['weights'].shape!=(128,128) or p.get('bias') is None or p['bias'].shape!=(128,)
                or p['shift']!=39 or p['branch_shift']!=22
                or p['input_zero_point']!=bundle['attention']['context_output_zero_point']):
            raise ValueError('Unexpected Part-4 projection/branch parameter layout')
        block_inputs=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]+patterns(0)
        output_min=np.iinfo(np.int32).max
        output_max=np.iinfo(np.int32).min
        projection_clips=0
        path=args.out_dir/f'block{i}_out_residual_golden.c'
        with complete_text_file(path) as stream:
            stream.write(f'/* Profile {profile["profile"]}, block {i}; checked integer Python Part 4. */\n')
            stream.write(f'#include "out_residual_golden.h"\nconst OutResidualGolden block{i}_out_residual_cases[OUT_RESIDUAL_CASE_COUNT] = {{\n')
            for c,block_input in enumerate(block_inputs):
                ln1=integer_ln1(block_input,bundle['ln1'])['output']
                qkv={tag:integer_projection(ln1,bundle['qkv'][tag])['output'] for tag in ('Q','K','V')}
                scores=integer_attention_scores(qkv['Q'],qkv['K'],bundle['attention'])
                softmax=integer_causal_softmax(scores,bundle['attention'])
                attention=integer_attention_context(softmax['probabilities'],qkv['V'],bundle['attention'])['output']
                result=integer_out_residual(attention,block_input,p)
                trial={'block_input':block_input,'ln1':ln1,'query':qkv['Q'],'key':qkv['K'],'value':qkv['V'],
                    'attention':attention,'accumulator':result['projection']['accumulator'],
                    'projection':result['projection']['output'],'branch':result['branch'],'output':result['output']}
                output_min=min(output_min,int(result['output'].min()))
                output_max=max(output_max,int(result['output'].max()))
                projection_clips+=result['projection']['saturation_count']
                stream.write(f'    {{ /* {labels[c]} */\n')
                for name,ctype in FIELDS:
                    flat=np.asarray(trial[name]).reshape(-1)
                    if len(flat)!=4096:
                        raise ValueError(f'Unexpected Part-4 checkpoint shape: {name}')
                    stream.write(f'        .{name} = {{\n')
                    for pos in range(0,4096,16):
                        stream.write('            '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
                    stream.write('        },\n')
                stream.write('    },\n')
            stream.write('};\n')
        completed=path.read_text()
        if not completed.endswith('};\n') or completed.count('    { /* ')!=13:
            raise RuntimeError(f'Incomplete generated fixture: {path}')
        summaries.append({'block':i,'input_alignment':p['input_alignment'],
            'output_int32_range':[int(output_min),int(output_max)],'projection_saturations':projection_clips})
        print(f'Block {i}: exported 13 output-projection/residual checkpoint cases for both paths',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,'summary':summaries,
        'input_policy':'Eight actual saved-chain block inputs plus five block-input edge patterns. Standalone receives Python attention output; chained recomputes C Parts 1-3.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in names if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 cases per path, 425984 residual INT32 outputs per path; C parameters verified')


if __name__=='__main__':
    main()

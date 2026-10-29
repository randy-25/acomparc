#!/usr/bin/env python3
"""Export software attention checkpoints for standalone and LN1/QKV-chain tests."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from tinystories_integer.cli_common import EXPERIMENT,write_json
from tinystories_integer.storage import load_archive
from tinystories_integer.c_export import write_c_parameters
from tinystories_integer.integer_reference import (
    integer_ln1,integer_projection,integer_attention_scores,
    integer_causal_softmax,integer_attention_context,
)
from export_ln1_fixtures import digest
from export_qkv_fixtures import patterns

FIELDS=[('block_input','int8_t',4096),('ln1','int8_t',4096),
        ('query','int8_t',4096),('key','int8_t',4096),('value','int8_t',4096),
        ('scores','int32_t',16384),('shifted','int32_t',16384),
        ('exponentials','int32_t',16384),('probabilities','int32_t',16384),
        ('context','int32_t',4096),('output','int8_t',4096)]


@contextmanager
def complete_text_file(path):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',
                                         suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=EXPERIMENT/'build/parameter_export')
    parser.add_argument('--c-params-dir',type=Path)
    parser.add_argument('--vectors',type=Path,default=EXPERIMENT/'tests/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=EXPERIMENT/'build/fixtures/attention')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    names=['attention_golden.h','attention_case_names.c','manifest.json']+[
        f'block{i}_attention_golden.c' for i in range(8)]
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
    with tempfile.TemporaryDirectory(prefix='attention-parameter-check-') as temporary:
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
    header=['#ifndef TINYSTORIES_ALLBLOCKS_ATTENTION_GOLDEN_H\n#define TINYSTORIES_ALLBLOCKS_ATTENTION_GOLDEN_H\n',
            '#include <stdint.h>\n#define ATTENTION_CASE_COUNT 13u\n',
            '#define ATTENTION_TOKENS 32u\n#define ATTENTION_HEADS 16u\n#define ATTENTION_HEAD_DIM 8u\n',
            '#define ATTENTION_ELEMENTS 4096u\n#define ATTENTION_SCORE_ELEMENTS 16384u\n',
            'typedef struct {\n']
    header += [f'    {ctype} {name}[{size}];\n' for name,ctype,size in FIELDS]
    header += ['} AttentionGolden;\n','extern const char *const attention_case_names[ATTENTION_CASE_COUNT];\n']
    header += [f'extern const AttentionGolden block{i}_attention_cases[ATTENTION_CASE_COUNT];\n' for i in range(8)]
    header += ['#endif\n']
    (args.out_dir/'attention_golden.h').write_text(''.join(header))
    (args.out_dir/'attention_case_names.c').write_text('#include "attention_golden.h"\nconst char *const attention_case_names[ATTENTION_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    summaries=[]
    for i,bundle in enumerate(bundles):
        block_inputs=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]+patterns(0)
        worst_row_sum_error=0
        cutoff_count=0
        path=args.out_dir/f'block{i}_attention_golden.c'
        with complete_text_file(path) as stream:
            stream.write(f'/* Profile {profile["profile"]}, block {i}; checked integer Python checkpoints. */\n')
            stream.write(f'#include "attention_golden.h"\nconst AttentionGolden block{i}_attention_cases[ATTENTION_CASE_COUNT] = {{\n')
            for c,block_input in enumerate(block_inputs):
                ln1=integer_ln1(block_input,bundle['ln1'])['output']
                qkv={tag:integer_projection(ln1,bundle['qkv'][tag])['output'] for tag in ('Q','K','V')}
                scores=integer_attention_scores(qkv['Q'],qkv['K'],bundle['attention'])
                softmax=integer_causal_softmax(scores,bundle['attention'])
                context=integer_attention_context(softmax['probabilities'],qkv['V'],bundle['attention'])
                trial={'block_input':block_input,'ln1':ln1,'query':qkv['Q'],'key':qkv['K'],'value':qkv['V'],
                    'scores':scores,'shifted':softmax['shifted_scores'],'exponentials':softmax['exponentials'],
                    'probabilities':softmax['probabilities'],'context':context['context_merged'],'output':context['output']}
                worst_row_sum_error=max(worst_row_sum_error,softmax['max_row_sum_error'])
                cutoff_count+=softmax['cutoff_count']
                stream.write(f'    {{ /* {labels[c]} */\n')
                for name,ctype,size in FIELDS:
                    flat=np.asarray(trial[name]).reshape(-1)
                    if len(flat)!=size:
                        raise ValueError(f'Unexpected attention checkpoint shape: {name}')
                    stream.write(f'        .{name} = {{\n')
                    for pos in range(0,size,16):
                        stream.write('            '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
                    stream.write('        },\n')
                stream.write('    },\n')
            stream.write('};\n')
        completed=path.read_text()
        if not completed.endswith('};\n') or completed.count('    { /* ')!=13:
            raise RuntimeError(f'Incomplete generated checkpoint file: {path}')
        summaries.append({'block':i,'max_row_sum_error_steps':worst_row_sum_error,'allowed_exponentials_cut_off':cutoff_count})
        print(f'Block {i}: exported 13 attention checkpoint cases for both paths',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,'softmax_summary':summaries,
        'input_policy':'Eight actual saved-chain block inputs plus five block-input edge patterns. Standalone attention receives their Python Q/K/V; the chained path recomputes C LN1 and Q/K/V.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in names if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 cases per path, 425984 attention INT8 outputs per path; C parameters verified')


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Export Part-6 chain fixtures and exhaustive 256-code GELU checkpoints."""

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
    checked_integer,integer_ln1,integer_ln2,integer_projection,integer_gelu,
    integer_attention_scores,integer_causal_softmax,integer_attention_context,integer_out_residual,
)
from export_ln1_fixtures import digest
from export_qkv_fixtures import patterns
from export_attention_fixtures import complete_text_file

MLP_FIELDS=[('input','int8_t',4096),('accumulator','int32_t',16384),
            ('expansion','int8_t',16384),('output','int8_t',16384)]
GELU_FIELDS=[('half_raw','int32_t'),('half','int8_t'),('cube_raw','int32_t'),('cube','int8_t'),
    ('weighted_raw','int32_t'),('weighted_cube','int8_t'),('combined','int64_t'),('sum','int8_t'),
    ('argument_raw','int32_t'),('argument','int8_t'),('tanh','int32_t'),('factor_fine','int32_t'),
    ('factor','int8_t'),('product','int32_t'),('output','int8_t')]


def write_values(stream,name,values,indent):
    flat=np.asarray(values).reshape(-1)
    stream.write(indent+f'.{name} = {{\n')
    for pos in range(0,len(flat),16):
        stream.write(indent+'    '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
    stream.write(indent+'},\n')


def exhaustive_gelu(params):
    values=np.tile(np.arange(-128,128,dtype=np.int16).astype(np.int8),64).reshape(1,32,512)
    result=integer_gelu(values,params)
    def scalar_raw(codes,spec,label):
        return checked_integer((codes.astype(object)-spec['input_zero_point'])*spec['centered_constant'],np.int32,label)
    spec=params['sum']
    result.update(half_raw=scalar_raw(values,params['half'],'GELU half raw'),
        weighted_raw=scalar_raw(result['cube'],params['weighted_cube'],'GELU weighted raw'),
        argument_raw=scalar_raw(result['sum'],params['argument'],'GELU argument raw'),
        combined=checked_integer((values.astype(object)-spec['linear_zero_point'])*spec['linear_multiplier']+
            (result['weighted_cube'].astype(object)-spec['cubic_zero_point'])*spec['cubic_multiplier'],np.int64,'GELU branch sum'))
    return {name:result[name].reshape(-1)[:256].copy() for name,ctype in GELU_FIELDS}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=PROJECT/'build/parameter_export')
    parser.add_argument('--c-params-dir',type=Path)
    parser.add_argument('--vectors',type=Path,default=PROJECT/'tests/sw/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=PROJECT/'build/fixtures/mlp_gelu')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    names=['mlp_gelu_golden.h','mlp_gelu_case_names.c','manifest.json']+[
        f'block{i}_mlp_gelu_golden.c' for i in range(8)]
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
    with tempfile.TemporaryDirectory(prefix='mlp-gelu-parameter-check-') as temporary:
        write_c_parameters(temporary,bundles,profile)
        for expected in Path(temporary).iterdir():
            actual=c_dir/expected.name
            if not actual.is_file() or actual.read_bytes()!=expected.read_bytes():
                raise ValueError(f'C data differs from parameter archive: {actual}')
    with np.load(args.vectors,allow_pickle=False) as saved:
        inputs=saved['inputs'].copy();previous_outputs=saved['block_outputs'].copy()
    if inputs.dtype!=np.int8 or inputs.shape!=(8,1,32,128):
        raise ValueError('Expected eight validated INT8 input cases')
    if previous_outputs.dtype!=np.int8 or previous_outputs.shape!=(8,8,1,32,128):
        raise ValueError('Expected eight-block reference outputs')
    labels=['benchmark']+[f'calibration_{i}' for i in range(7)]+[
        'edge_centered_zero','edge_all_min','edge_all_max','edge_alternating','edge_single_outlier']
    args.out_dir.mkdir(parents=True,exist_ok=True)
    header=['#ifndef TINYSTORIES_ALLBLOCKS_MLP_GELU_GOLDEN_H\n#define TINYSTORIES_ALLBLOCKS_MLP_GELU_GOLDEN_H\n',
        '#include <stdint.h>\n#define MLP_GELU_CASE_COUNT 13u\n#define MLP_INPUT_ELEMENTS 4096u\n',
        '#define MLP_OUTPUT_ELEMENTS 16384u\n#define MLP_SCORE_ELEMENTS 16384u\n#define GELU_CODE_COUNT 256u\n',
        'typedef struct {\n']
    header += [f'    {ctype} {name}[{size}];\n' for name,ctype,size in MLP_FIELDS]
    header += ['} MlpGeluPathGolden;\ntypedef struct {\n    int8_t block_input[4096];\n',
        '    MlpGeluPathGolden local;\n    MlpGeluPathGolden chained;\n} MlpGeluGolden;\n',
        'typedef struct {\n']
    header += [f'    {ctype} {name}[256];\n' for name,ctype in GELU_FIELDS]
    header += ['} GeluGolden;\nextern const char *const mlp_gelu_case_names[MLP_GELU_CASE_COUNT];\n']
    for i in range(8):
        header += [f'extern const MlpGeluGolden block{i}_mlp_gelu_cases[MLP_GELU_CASE_COUNT];\n',
                   f'extern const GeluGolden block{i}_gelu_codes;\n']
    (args.out_dir/'mlp_gelu_golden.h').write_text(''.join(header)+'#endif\n')
    (args.out_dir/'mlp_gelu_case_names.c').write_text('#include "mlp_gelu_golden.h"\nconst char *const mlp_gelu_case_names[MLP_GELU_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    for i,bundle in enumerate(bundles):
        p=bundle['mlp_expand']
        if p['weights'].shape!=(512,128) or p.get('bias') is None or p['bias'].shape!=(512,):
            raise ValueError('Expected biased 128-to-512 expansion')
        block_inputs=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]+patterns(0)
        chained_inputs=[]
        for block_input in block_inputs:
            ln1=integer_ln1(block_input,bundle['ln1'])['output']
            qkv={tag:integer_projection(ln1,bundle['qkv'][tag])['output'] for tag in ('Q','K','V')}
            scores=integer_attention_scores(qkv['Q'],qkv['K'],bundle['attention'])
            softmax=integer_causal_softmax(scores,bundle['attention'])
            attention=integer_attention_context(softmax['probabilities'],qkv['V'],bundle['attention'])['output']
            residual=integer_out_residual(attention,block_input,bundle['out_residual'])['output']
            chained_inputs.append(integer_ln2(residual,bundle['ln2'])['output'])
        local_inputs=chained_inputs[:8]+patterns(p['input_zero_point'])
        codes=exhaustive_gelu(bundle['gelu'])
        path=args.out_dir/f'block{i}_mlp_gelu_golden.c'
        with complete_text_file(path) as stream:
            stream.write(f'/* Profile {profile["profile"]}, block {i}; integer Python Part 6. */\n')
            stream.write(f'#include "mlp_gelu_golden.h"\nconst MlpGeluGolden block{i}_mlp_gelu_cases[MLP_GELU_CASE_COUNT] = {{\n')
            for c,block_input in enumerate(block_inputs):
                stream.write(f'    {{ /* {labels[c]} */\n')
                write_values(stream,'block_input',block_input,'        ')
                for name,values in [('local',local_inputs[c]),('chained',chained_inputs[c])]:
                    expansion=integer_projection(values,p)
                    gelu=integer_gelu(expansion['output'],bundle['gelu'])
                    trial={'input':values,'accumulator':expansion['accumulator'],
                           'expansion':expansion['output'],'output':gelu['output']}
                    stream.write(f'        .{name} = {{\n')
                    for field,ctype,size in MLP_FIELDS:
                        flat=np.asarray(trial[field]).reshape(-1)
                        if len(flat)!=size:raise ValueError(f'Unexpected Part-6 shape: {field}')
                        write_values(stream,field,flat,'            ')
                    stream.write('        },\n')
                stream.write('    },\n')
            stream.write('};\n')
            stream.write(f'const GeluGolden block{i}_gelu_codes = {{\n')
            for field,ctype in GELU_FIELDS:write_values(stream,field,codes[field],'    ')
            stream.write('};\n')
        completed=path.read_text()
        if not completed.endswith('};\n') or completed.count('    { /* ')!=13:
            raise RuntimeError(f'Incomplete generated fixture: {path}')
        print(f'Block {i}: exported 13 cases per path and all 256 GELU input codes',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,
        'local_inputs':'Eight actual LN2 outputs, then five direct INT8 expansion-input patterns.',
        'chained_inputs':'Thirteen block inputs pass through Python Parts 1-5; C must reproduce these expansion inputs.',
        'gelu_code_test':'All 256 INT8 codes repeated across 16384 elements; stored checkpoints have period 256.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in names if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 cases per path, 1703936 MLP/GELU INT8 outputs per path; 256 GELU codes per block; C parameters verified')


if __name__=='__main__':main()

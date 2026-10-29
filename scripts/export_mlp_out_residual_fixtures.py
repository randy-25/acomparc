#!/usr/bin/env python3
"""Export Part-7 checkpoints, final INT8 outputs and boundary-rounding probes."""

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
    integer_ln1,integer_ln2,integer_projection,integer_gelu,integer_attention_scores,
    integer_causal_softmax,integer_attention_context,integer_out_residual,
    integer_mlp_out_residual,requantize_to_i8,
)
from export_ln1_fixtures import digest
from export_qkv_fixtures import patterns
from export_ln2_fixtures import direct_ln2_patterns
from export_attention_fixtures import complete_text_file

FIELDS=[('input','int8_t',16384),('retained','int32_t',4096),
        ('accumulator','int32_t',4096),('projection','int8_t',4096),
        ('branch','int32_t',4096),('output_i32','int32_t',4096),('output','int8_t',4096)]


def direct_gelu_patterns(zero_point):
    values=[np.full((1,32,512),zero_point,np.int8),np.full((1,32,512),-128,np.int8),
            np.full((1,32,512),127,np.int8),
            np.broadcast_to(np.array([-128,127]*256,np.int8),(1,32,512)).copy(),
            np.full((1,32,512),zero_point,np.int8)]
    values[-1][0,:,0]=127
    return values


def boundary_probes(spec):
    step=1<<spec['shift'];half=step//2
    values=np.array([np.iinfo(np.int32).min,np.iinfo(np.int32).min+1,
        -129*step,-128*step-half,-128*step-half+1,-128*step-1,-128*step,
        -128*step+half-1,-128*step+half,-step-half-1,-step-half,-step-half+1,
        -step-1,-step,-step+1,-half-1,-half,-half+1,-1,0,1,half-1,half,half+1,
        step-1,step,step+1,127*step+half-1,127*step+half,128*step,
        np.iinfo(np.int32).max-1,np.iinfo(np.int32).max],np.int32)
    if values.size!=32:raise ValueError('Boundary probe size mismatch')
    return values,requantize_to_i8(values,spec,'block boundary probes')['output']


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
    parser.add_argument('--out-dir',type=Path,default=PROJECT/'build/fixtures/mlp_out_residual')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    names=['mlp_out_residual_golden.h','mlp_out_residual_case_names.c','manifest.json']+[
        f'block{i}_mlp_out_residual_golden.c' for i in range(8)]
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
    with tempfile.TemporaryDirectory(prefix='mlp-out-parameter-check-') as temporary:
        write_c_parameters(temporary,bundles,profile)
        for expected in Path(temporary).iterdir():
            actual=c_dir/expected.name
            if not actual.is_file() or actual.read_bytes()!=expected.read_bytes():
                raise ValueError(f'C data differs from parameter archive: {actual}')
    with np.load(args.vectors,allow_pickle=False) as saved:
        inputs=saved['inputs'].copy();previous_outputs=saved['block_outputs'].copy()
        pre_boundary=saved['pre_boundary_outputs'].copy()
    if inputs.dtype!=np.int8 or inputs.shape!=(8,1,32,128):
        raise ValueError('Expected eight validated INT8 input cases')
    if previous_outputs.dtype!=np.int8 or previous_outputs.shape!=(8,8,1,32,128):
        raise ValueError('Expected eight-block reference outputs')
    if pre_boundary.dtype!=np.int32 or pre_boundary.shape!=(8,8,1,32,128):
        raise ValueError('Expected frozen INT32 pre-boundary outputs')
    labels=['benchmark']+[f'calibration_{i}' for i in range(7)]+[
        'edge_zero','edge_negative','edge_positive','edge_alternating','edge_single_outlier']
    args.out_dir.mkdir(parents=True,exist_ok=True)
    header=['#ifndef TINYSTORIES_ALLBLOCKS_MLP_OUT_RESIDUAL_GOLDEN_H\n#define TINYSTORIES_ALLBLOCKS_MLP_OUT_RESIDUAL_GOLDEN_H\n',
        '#include <stdint.h>\n#define MLP_OUT_CASE_COUNT 13u\n#define MLP_OUT_INPUT_ELEMENTS 16384u\n',
        '#define MLP_OUT_ELEMENTS 4096u\n#define MLP_OUT_SCORE_ELEMENTS 16384u\n#define BOUNDARY_PROBE_COUNT 32u\n',
        'typedef struct {\n']
    header += [f'    {ctype} {name}[{size}];\n' for name,ctype,size in FIELDS]
    header += ['} MlpOutPathGolden;\ntypedef struct {\n    int8_t block_input[4096];\n',
        '    MlpOutPathGolden local;\n    MlpOutPathGolden chained;\n} MlpOutGolden;\n',
        'typedef struct { int32_t input[32]; int8_t output[32]; } BoundaryGolden;\n',
        'extern const char *const mlp_out_residual_case_names[MLP_OUT_CASE_COUNT];\n']
    for i in range(8):
        header += [f'extern const MlpOutGolden block{i}_mlp_out_residual_cases[MLP_OUT_CASE_COUNT];\n',
                   f'extern const BoundaryGolden block{i}_boundary_probes;\n']
    (args.out_dir/'mlp_out_residual_golden.h').write_text(''.join(header)+'#endif\n')
    (args.out_dir/'mlp_out_residual_case_names.c').write_text('#include "mlp_out_residual_golden.h"\nconst char *const mlp_out_residual_case_names[MLP_OUT_CASE_COUNT] = {\n'+
        ''.join(f'    "{label}",\n' for label in labels)+'};\n')
    summaries=[]
    for i,bundle in enumerate(bundles):
        p=bundle['mlp_out']
        if p['weights'].shape!=(128,512) or p.get('bias') is None or p['bias'].shape!=(128,):
            raise ValueError('Expected biased 512-to-128 projection')
        if p['boundary']['multiplier']!=1 or p['boundary']['shift']!=(9 if i in (0,7) else 8) or p['boundary']['output_zero_point']!=0:
            raise ValueError('Unexpected final INT8 boundary profile')
        block_inputs=[inputs[c] if i==0 else previous_outputs[c,i-1] for c in range(8)]+patterns(0)
        chained_inputs=[];retained=[]
        for block_input in block_inputs:
            ln1=integer_ln1(block_input,bundle['ln1'])['output']
            qkv={tag:integer_projection(ln1,bundle['qkv'][tag])['output'] for tag in ('Q','K','V')}
            scores=integer_attention_scores(qkv['Q'],qkv['K'],bundle['attention'])
            softmax=integer_causal_softmax(scores,bundle['attention'])
            attention=integer_attention_context(softmax['probabilities'],qkv['V'],bundle['attention'])['output']
            residual=integer_out_residual(attention,block_input,bundle['out_residual'])['output']
            ln2=integer_ln2(residual,bundle['ln2'])['output']
            expansion=integer_projection(ln2,bundle['mlp_expand'])['output']
            chained_inputs.append(integer_gelu(expansion,bundle['gelu'])['output']);retained.append(residual)
        local_inputs=chained_inputs[:8]+direct_gelu_patterns(p['input_zero_point'])
        local_retained=retained[:8]+direct_ln2_patterns()
        probe_inputs,probe_outputs=boundary_probes(p['boundary'])
        clips={'local':0,'chained':0}
        path=args.out_dir/f'block{i}_mlp_out_residual_golden.c'
        with complete_text_file(path) as stream:
            stream.write(f'/* Profile {profile["profile"]}, block {i}; checked integer Python Part 7. */\n')
            stream.write(f'#include "mlp_out_residual_golden.h"\nconst MlpOutGolden block{i}_mlp_out_residual_cases[MLP_OUT_CASE_COUNT] = {{\n')
            for c,block_input in enumerate(block_inputs):
                stream.write(f'    {{ /* {labels[c]} */\n')
                write_values(stream,'block_input',block_input,'        ')
                for name,values,res in [('local',local_inputs[c],local_retained[c]),('chained',chained_inputs[c],retained[c])]:
                    result=integer_mlp_out_residual(values,res,p)
                    if c<8:
                        np.testing.assert_array_equal(result['output'],previous_outputs[c,i])
                        np.testing.assert_array_equal(result['output_i32'],pre_boundary[c,i])
                    trial={'input':values,'retained':res,'accumulator':result['projection']['accumulator'],
                        'projection':result['projection']['output'],'branch':result['branch'],
                        'output_i32':result['output_i32'],'output':result['output']}
                    clips[name]+=result['boundary_saturations']
                    stream.write(f'        .{name} = {{\n')
                    for field,ctype,size in FIELDS:
                        flat=np.asarray(trial[field]).reshape(-1)
                        if len(flat)!=size:raise ValueError(f'Unexpected Part-7 shape: {field}')
                        write_values(stream,field,flat,'            ')
                    stream.write('        },\n')
                stream.write('    },\n')
            stream.write('};\n')
            stream.write(f'const BoundaryGolden block{i}_boundary_probes = {{\n')
            write_values(stream,'input',probe_inputs,'    ');write_values(stream,'output',probe_outputs,'    ')
            stream.write('};\n')
        completed=path.read_text()
        if not completed.endswith('};\n') or completed.count('    { /* ')!=13:
            raise RuntimeError(f'Incomplete generated fixture: {path}')
        summaries.append({'block':i,'boundary_shift':p['boundary']['shift'],'boundary_clips':clips})
        print(f'Block {i}: exported 13 cases per path and 32 boundary probes; saved block outputs verified',flush=True)
    write_json(args.out_dir/'manifest.json',{'profile':profile,'parameter_json_sha256':digest(archive_path),
        'reference_vectors_sha256':digest(args.vectors),'case_names':labels,'summary':summaries,
        'local_inputs':'Eight actual GELU/residual pairs, then five direct INT8 GELU and INT32 retained-residual edge pairs.',
        'chained_inputs':'Thirteen block inputs pass through Python Parts 1-6, retaining the original Part-4 residual.',
        'boundary_probes':'32 rounding/saturation values, including INT32 endpoints, repeated across a full 4096-element buffer.',
        'saved_output_check':'First eight cases per block match the frozen INT32 and INT8 whole-block vectors.',
        'c_parameters_sha256':{p.name:digest(p) for p in sorted(c_dir.glob('*')) if p.is_file()},
        'fixture_files_sha256':{name:digest(args.out_dir/name) for name in names if name!='manifest.json'}})
    print('PASS: 8 blocks, 104 cases per path, 425984 final INT8 outputs per path; 32 boundary probes per block; C parameters verified')


if __name__=='__main__':main()

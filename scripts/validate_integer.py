#!/usr/bin/env python3
"""Compare extraction vectors, or rerun FP32/TFLite/integer host diagnostics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from tinystories_integer.cli_common import PROJECT,MODELS,FP32,run_chain,write_json,check_validation_file


def regression(args,bundles,profile):
    vector_path = args.vectors
    meta = json.loads(vector_path.with_suffix('.json').read_text())
    if hashlib.sha256(vector_path.read_bytes()).hexdigest() != meta['npz_sha256']:
        raise ValueError('Regression-vector checksum mismatch')
    if meta['model_sha256'] != profile['model_sha256'] or meta['profile'] != profile['profile']:
        raise ValueError('Regression vectors belong to a different profile')
    with np.load(vector_path,allow_pickle=False) as saved:
        inputs = saved['inputs'].copy()
        expected = saved['block_outputs'].copy()
        expected_i32 = saved['pre_boundary_outputs'].copy()
    outputs,pre_boundary,diagnostics = run_chain(inputs,bundles)
    np.testing.assert_array_equal(outputs,expected,err_msg='INT8 chain differs from notebook')
    np.testing.assert_array_equal(pre_boundary,expected_i32,err_msg='INT32 intermediate differs from notebook')
    report = {'mode':'extraction-regression','cases':len(inputs),'block_calls':len(inputs)*8,
              'int8_outputs_checked':outputs.size,'int32_outputs_checked':pre_boundary.size,
              'mismatches':0,'profile':profile['profile'],'diagnostics':diagnostics}
    write_json(args.out_dir/'regression.json',report)
    print(f'PASS: {len(inputs)*8} block calls; {outputs.size} INT8 and {pre_boundary.size} INT32 values match notebook')


def full(args,bundles,profile):
    # Keep these dependencies out of pure integer execution and --help.
    from tinystories_integer.diagnostics import HostReferences,error_metrics,encode_reference_block_output
    check_validation_file(args.validation_inputs,profile)
    with np.load(args.validation_inputs,allow_pickle=False) as saved:
        validation = {k:saved[k].copy() for k in saved.files}
    host = HostReferences(bundles,profile,args.models_dir,args.fp32_dir)
    ids = [('benchmark',validation['benchmark_input_ids'].reshape(1,32))]
    ids += [(f'calibration_{i}',x.reshape(1,32)) for i,x in enumerate(validation['calibration_input_ids'])]
    cases, fixture_arrays = [], {}
    for name,tokens in ids:
        references = host.collect_fp32_boundaries(tokens)
        if name == 'benchmark':
            for i in range(8):
                for edge in ('input','output'):
                    np.testing.assert_allclose(references[i][edge+'_fp32'],validation[f'block{i}_{edge}_fp32'],atol=1e-6,rtol=1e-6)
            import torch
            with torch.no_grad():
                model_logits = host.model(input_ids=torch.as_tensor(tokens.copy(),dtype=torch.long),use_cache=False,return_dict=True).logits
                rebuilt_logits = host.logits_from_final_hidden(references[-1]['output_fp32'])
            np.testing.assert_allclose(rebuilt_logits.cpu().numpy(),model_logits.cpu().numpy(),atol=1e-5,rtol=1e-5)
        initial = references[0]['input_fp32']
        codes,mixed,input_clips = host.encode_boundary(initial,0)
        direct = initial.copy()
        report = {'name':name,'initial_input_clips':input_clips,'blocks':[],'paths':{}}
        hidden_paths = {'fp32':references[-1]['output_fp32']}
        case_codes,case_i32,case_mixed = [],[],[]
        fixture_arrays[name+'_inputs'] = codes.copy()
        for i in range(8):
            from tinystories_integer.integer_reference import integer_block_forward
            direct = host.run_tflite_block(i,direct)
            mixed_raw = host.run_tflite_block(i,mixed)
            encoded = encode_reference_block_output(mixed_raw,bundles[i]['mlp_out']['boundary'])
            mixed = (encoded['output'].astype(np.float64)*bundles[i]['output_scale']).astype(np.float32)
            result = integer_block_forward(codes,bundles[i])
            codes = result['output']
            represented = (codes.astype(np.float64)*bundles[i]['output_scale']).astype(np.float32)
            reference = references[i]['output_fp32']
            report['blocks'].append({'block':i,'direct':error_metrics(direct,reference),
                'mixed_boundary':error_metrics(mixed,reference),'integer':error_metrics(represented,reference),
                'mixed_boundary_clips':encoded['saturation_count'],'integer_diagnostics':{k:v for k,v in result.items() if k not in ('output','output_i32')}})
            case_codes.append(codes.copy());case_i32.append(result['output_i32']);case_mixed.append(encoded['output'])
        hidden_paths.update(direct=direct,mixed_boundary=mixed,integer=represented)
        fp32_predictions = host.predict_from_final_hidden(hidden_paths['fp32'])
        for path,hidden in hidden_paths.items():
            losses = host.next_token_losses(hidden,tokens)
            predictions = host.predict_from_final_hidden(hidden)
            report['paths'][path] = {'losses':losses,'cross_entropy':float(losses.mean()),'perplexity':float(np.exp(losses.mean())),
                'predicted_ids':predictions,'matches_fp32':int(np.sum(predictions==fp32_predictions)),
                'final_position_id':int(predictions[-1])}
        fixture_arrays[name+'_block_outputs'] = np.stack(case_codes)
        fixture_arrays[name+'_pre_boundary_outputs'] = np.stack(case_i32)
        fixture_arrays[name+'_mixed_outputs'] = np.stack(case_mixed)
        cases.append(report)
        print(f'Completed four paths: {name}',flush=True)
    groups = {}
    for group,selected in [('Benchmark',cases[:1]),('Calibration',cases[1:]),('All',cases)]:
        groups[group] = {}
        for path in ('fp32','direct','mixed_boundary','integer'):
            losses = np.concatenate([x['paths'][path]['losses'] for x in selected])
            mean = float(losses.mean())
            groups[group][path] = {'targets':len(losses),'cross_entropy':mean,'perplexity':math.exp(mean)}
        baseline = groups[group]['fp32']['perplexity']
        for row in groups[group].values():
            row['perplexity_ratio_fp32'] = row['perplexity']/baseline
    args.out_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out_dir/'chain_fixtures.npz',**fixture_arrays)
    write_json(args.out_dir/'validation.json',{'profile':profile,'cases':cases,'loss_summary':groups})
    print('PASS: benchmark FP32 reconstruction and checked arithmetic completed')
    print('All cases:')
    for path,row in groups['All'].items():
        print(f'{path}: CE={row["cross_entropy"]:.6f}, PPL={row["perplexity"]:.6f}, ratio={row["perplexity_ratio_fp32"]:.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,required=True)
    parser.add_argument('--mode',choices=('regression','full'),default='regression')
    parser.add_argument('--vectors',type=Path,default=PROJECT/'tests/sw/data/chain_reference.npz')
    parser.add_argument('--models-dir',type=Path,default=MODELS)
    parser.add_argument('--fp32-dir',type=Path,default=FP32)
    parser.add_argument('--validation-inputs',type=Path,default=MODELS/'validation_inputs.npz')
    parser.add_argument('--out-dir',type=Path,required=True)
    args = parser.parse_args()
    target = 'regression.json' if args.mode=='regression' else 'validation.json'
    if (args.out_dir/target).exists() or (args.mode=='full' and (args.out_dir/'chain_fixtures.npz').exists()):
        parser.error('Report already exists; choose a new output directory')
    from tinystories_integer.storage import load_archive
    bundles,profile = load_archive(args.params)
    (regression if args.mode=='regression' else full)(args,bundles,profile)


if __name__ == '__main__':
    main()

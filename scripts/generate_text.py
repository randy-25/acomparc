#!/usr/bin/env python3
"""Greedy text comparison with the notebook's four execution paths."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))

import argparse
from pathlib import Path
from tinystories_integer.cli_common import MODELS,FP32,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,required=True)
    parser.add_argument('--models-dir',type=Path,default=MODELS)
    parser.add_argument('--fp32-dir',type=Path,default=FP32)
    parser.add_argument('--prompt',default='Once upon a time, a small fox')
    parser.add_argument('--max-new-tokens',type=int,default=20)
    parser.add_argument('--context-mode',choices=('prefix','rolling32'),default='prefix')
    parser.add_argument('--truncate-prompt-to32',action='store_true',help='Explicitly keep the first 32 prompt tokens for rolling32, as in the notebook')
    parser.add_argument('--paths',nargs='+',choices=('fp32','direct','mixed_boundary','integer'),default=['fp32','direct','mixed_boundary','integer'])
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.max_new_tokens<=0:
        parser.error('--max-new-tokens must be positive')
    if args.out.exists():
        parser.error('Output already exists; choose a new path')
    if args.truncate_prompt_to32 and args.context_mode!='rolling32':
        parser.error('--truncate-prompt-to32 requires --context-mode rolling32')
    from tinystories_integer.storage import load_archive
    from tinystories_integer.diagnostics import HostReferences
    bundles,profile = load_archive(args.params)
    host = HostReferences(bundles,profile,args.models_dir,args.fp32_dir,load_tokenizer=True)
    ids = host.tokenizer.encode(args.prompt,add_special_tokens=False)
    original_count = len(ids)
    if args.truncate_prompt_to32:
        if len(ids)<32:
            parser.error('Prompt has fewer than 32 tokens')
        ids = ids[:32]
    if args.context_mode=='rolling32' and len(ids)!=32:
        parser.error(f'rolling32 needs exactly 32 prompt tokens; received {len(ids)}')
    if args.context_mode=='prefix' and not 1<=len(ids)<32:
        parser.error(f'prefix needs 1..31 prompt tokens; received {len(ids)}')
    print(f'Effective prompt: {host.tokenizer.decode(ids)!r} ({len(ids)} tokens)')
    method = host.generate_with_path if args.context_mode=='prefix' else host.generate_with_32_token_window
    results = {}
    for path in args.paths:
        results[path] = method(path,ids,max_new_tokens=args.max_new_tokens)
        print(f'\n{path}:\n{results[path]["text"]}\n')
    write_json(args.out,{'profile':profile['profile'],'context_mode':args.context_mode,
        'original_prompt_token_count':original_count,'effective_prompt_ids':ids,
        'effective_prompt':host.tokenizer.decode(ids),'results':results})
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

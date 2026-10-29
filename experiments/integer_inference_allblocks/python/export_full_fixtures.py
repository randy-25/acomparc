#!/usr/bin/env python3
"""Export the frozen eight-case, eight-block regression vectors for C tests."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from tinystories_integer.cli_common import EXPERIMENT,write_json
from tinystories_integer.storage import load_archive
from tinystories_integer.c_export import write_c_parameters
from export_ln1_fixtures import digest
from export_attention_fixtures import complete_text_file


def write_array(stream,name,values,indent):
    stream.write(indent+f'.{name} = {{\n')
    flat=np.asarray(values).reshape(-1)
    for pos in range(0,len(flat),16):
        stream.write(indent+'    '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
    stream.write(indent+'},\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',type=Path,default=EXPERIMENT/'build/parameter_export')
    parser.add_argument('--vectors',type=Path,default=EXPERIMENT/'tests/data/chain_reference.npz')
    parser.add_argument('--out-dir',type=Path,default=EXPERIMENT/'build/fixtures/full')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if not args.overwrite and any((args.out_dir/n).exists() for n in ('full_golden.h','full_golden.c','manifest.json')):
        parser.error('Fixtures exist; choose a new directory or pass --overwrite')
    archive_path=args.params/'parameters.json' if args.params.is_dir() else args.params
    bundles,profile=load_archive(archive_path)
    meta=json.loads(args.vectors.with_suffix('.json').read_text())
    if (digest(args.vectors)!=meta['npz_sha256'] or profile['profile']!=meta['profile']
            or profile['model_sha256']!=meta['model_sha256'] or profile['notebook_sha256']!=meta['notebook_sha256']):
        raise ValueError('Frozen vectors differ from the parameter profile')
    with tempfile.TemporaryDirectory(prefix='full-parameter-check-') as td:
        write_c_parameters(td,bundles,profile)
        for expected in Path(td).iterdir():
            actual=archive_path.parent/'c'/expected.name
            if not actual.is_file() or actual.read_bytes()!=expected.read_bytes():
                raise ValueError(f'C parameters differ from archive: {actual}')
    with np.load(args.vectors,allow_pickle=False) as data:
        inputs=data['inputs'].copy();outputs=data['block_outputs'].copy();wide=data['pre_boundary_outputs'].copy()
    if inputs.dtype!=np.int8 or inputs.shape!=(8,1,32,128):raise ValueError('Invalid input shape/type')
    if outputs.dtype!=np.int8 or outputs.shape!=(8,8,1,32,128):raise ValueError('Invalid output shape/type')
    if wide.dtype!=np.int32 or wide.shape!=(8,8,1,32,128):raise ValueError('Invalid pre-boundary shape/type')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'full_golden.h').write_text('''#ifndef TINYSTORIES_FULL_GOLDEN_H
#define TINYSTORIES_FULL_GOLDEN_H
#include <stdint.h>
#define FULL_CASE_COUNT 8u
#define FULL_ELEMENTS 4096u
typedef struct {
    int8_t input[4096];
    int8_t output[8][4096];
    int32_t pre_boundary[8][4096];
} FullGolden;
extern const FullGolden full_cases[FULL_CASE_COUNT];
extern const char *const full_case_names[FULL_CASE_COUNT];
#endif
''')
    with complete_text_file(args.out_dir/'full_golden.c') as stream:
        stream.write('#include "full_golden.h"\nconst char *const full_case_names[FULL_CASE_COUNT] = {\n')
        stream.write('    "benchmark", '+', '.join(f'"calibration_{i}"' for i in range(7))+'\n};\n')
        stream.write('const FullGolden full_cases[FULL_CASE_COUNT] = {\n')
        for case in range(8):
            stream.write(f'    {{ /* case {case} */\n');write_array(stream,'input',inputs[case],'        ')
            for name,values in [('output',outputs[case]),('pre_boundary',wide[case])]:
                stream.write(f'        .{name} = {{\n')
                for b in range(8):
                    flat=values[b].reshape(-1);stream.write(f'            {{ /* block {b} */\n')
                    for pos in range(0,4096,16):
                        stream.write('                '+', '.join(str(int(v)) for v in flat[pos:pos+16])+',\n')
                    stream.write('            },\n')
                stream.write('        },\n')
            stream.write('    },\n')
        stream.write('};\n')
    write_json(args.out_dir/'manifest.json',{'profile':profile,'vectors_sha256':digest(args.vectors),
        'files_sha256':{name:digest(args.out_dir/name) for name in ('full_golden.h','full_golden.c')},
        'description':'Frozen notebook inputs, every INT8 block boundary and every INT32 pre-boundary sum.'})
    print('PASS: 8 input cases, 64 block boundaries, 262144 INT8 and 262144 INT32 reference values; C parameters verified')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare RV64 disassembly and optional raw-minstret logs for both backends."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))

import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import re

STAGES=('ln1','qkv','attention','out_residual','ln2','mlp_gelu','mlp_out_boundary')
XTILE_NAMES=('cfg','set_ab','set_c','set_zp','run')
XREQUANT_NAMES=('cfg','set_io','set_mul','set_zp','run')
PROFILE_HEADER='PROFILE_HEADER,backend,case,block,region,raw_minstret'


def parse_disassembly(text):
    if 'riscv' not in text[:4096].lower():
        raise ValueError('Expected RISC-V objdump output with its file-format header')
    functions={};current=None
    for line in text.splitlines():
        symbol=re.match(r'^\s*[0-9a-fA-F]+ <([^>]+)>:\s*$',line)
        if symbol:
            current=symbol.group(1)
            functions.setdefault(current,{'instructions':0,'bytes':0,'custom':Counter(),'mnemonics':Counter(),'lines':[line]})
            continue
        if current is None:continue
        record=re.match(r'^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8}|[0-9a-fA-F]{4})\s+(.+)$',line)
        if not record:continue
        encoded,assembly=record.groups();word=int(encoded,16)
        f=functions[current];f['instructions']+=1;f['bytes']+=len(encoded)//2
        f['mnemonics'][assembly.split()[0]]+=1;f['lines'].append(line)
        if len(encoded)==8 and word&0x7f in (0x0b,0x2b):
            opcode=word&0x7f;funct3=(word>>12)&7
            names=XTILE_NAMES if opcode==0x0b else XREQUANT_NAMES
            name=names[funct3] if funct3<len(names) else f'unknown_funct3_{funct3}'
            f['custom'][('xtile_' if opcode==0x0b else 'xrequant_')+name]+=1
    if not functions or not any(f['instructions'] for f in functions.values()):
        raise ValueError('No supported 16/32-bit RISC-V instruction records found')
    totals=Counter()
    for f in functions.values():totals.update(f['custom'])
    return {'functions':functions,'custom_totals':totals}


def validate_disassembly_pair(scalar,custom):
    if sum(scalar['custom_totals'].values()):
        raise ValueError('Scalar disassembly contains custom opcode 0x0b or 0x2b')
    if not custom['custom_totals']['xtile_run'] or not custom['custom_totals']['xrequant_run']:
        raise ValueError('Custom disassembly must contain both XTILE RUN and XREQUANT RUN sites')


def parse_profile(text,backend):
    completions=re.findall(rf'(?m)^PASS: backend={re.escape(backend)} mode=profile cases=(\d+);',text)
    if len(completions)!=1 or 'FAIL' in text:
        raise ValueError(f'{backend}: profile log lacks a successful completion marker')
    declared_cases=int(completions[0])
    if not 1<=declared_cases<=8:
        raise ValueError(f'{backend}: invalid completed profile case count {declared_cases}')
    rows={}
    for line in text.splitlines():
        # GCC can turn the literal header printf into puts. The supplied
        # bare-metal runtime's puts omits its newline, joining header + row.
        # Remove only this exact known header; preserve the complete record
        # and retain all backend, duplicate, completeness and sum checks.
        if line.startswith(PROFILE_HEADER):
            line=line[len(PROFILE_HEADER):].lstrip(' \t\r')
        if not line.startswith('PROFILE,'):continue
        parts=next(csv.reader([line]))
        if len(parts)!=6 or parts[1]!=backend:raise ValueError('Unexpected profile row/backend')
        key=(int(parts[2]),int(parts[3]),parts[4]);value=int(parts[5])
        if key in rows or value<=0:raise ValueError('Duplicate or nonpositive profile interval')
        rows[key]=value
    cases=sorted({key[0] for key in rows})
    if cases!=list(range(declared_cases)):
        raise ValueError(f'{backend}: expected cases 0..{declared_cases-1}, found {cases}')
    expected=set()
    for c in cases:
        expected.update([(c,-1,'chain'),(c,-1,'sum_block_raw')])
        for b in range(8):
            expected.update((c,b,name) for name in ('forward','sum_stage_raw',*STAGES))
    missing=expected-set(rows)
    unexpected=set(rows)-expected
    if missing or unexpected:
        raise ValueError(f'{backend}: missing profile regions {sorted(missing)}; unexpected profile regions {sorted(unexpected)}')
    for c in cases:
        for b in range(8):
            if rows.get((c,b,'sum_stage_raw'))!=sum(rows.get((c,b,s),0) for s in STAGES):
                raise ValueError('Stage aggregate differs from raw intervals')
        if rows.get((c,-1,'sum_block_raw'))!=sum(rows.get((c,b,'forward'),0) for b in range(8)):
            raise ValueError('Block aggregate differs from raw intervals')
    return rows


def comparison_rows(scalar,custom):
    if set(scalar)!=set(custom):raise ValueError('Scalar/custom logs profile different cases or regions')
    return [{'case':c,'block':b,'region':region,'scalar_raw_minstret':scalar[(c,b,region)],
        'custom_raw_minstret':custom[(c,b,region)],
        'reduction_percent':100*(scalar[(c,b,region)]-custom[(c,b,region)])/scalar[(c,b,region)],
        'scalar_custom_instruction_ratio':scalar[(c,b,region)]/custom[(c,b,region)]}
        for c,b,region in sorted(scalar)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar-dis',type=Path,required=True)
    parser.add_argument('--custom-dis',type=Path,required=True)
    parser.add_argument('--scalar-log',type=Path)
    parser.add_argument('--custom-log',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if bool(args.scalar_log)!=bool(args.custom_log):parser.error('Supply both profile logs or neither')
    if not args.overwrite and any((args.out_dir/n).exists() for n in ('report.md','comparison.json','static_functions.csv','dynamic_regions.csv')):
        parser.error('Analysis output exists; choose a new directory or use --overwrite')
    scalar=parse_disassembly(args.scalar_dis.read_text());custom=parse_disassembly(args.custom_dis.read_text())
    validate_disassembly_pair(scalar,custom)
    dynamic=[]
    if args.scalar_log:
        dynamic=comparison_rows(parse_profile(args.scalar_log.read_text(),'scalar'),parse_profile(args.custom_log.read_text(),'custom'))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'static_functions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['backend','function','static_instructions','instruction_bytes','xtile_sites','xrequant_sites'])
        for backend,data in [('scalar',scalar),('custom',custom)]:
            snippets=[]
            for name,f in data['functions'].items():
                writer.writerow([backend,name,f['instructions'],f['bytes'],
                    sum(v for k,v in f['custom'].items() if k.startswith('xtile_')),
                    sum(v for k,v in f['custom'].items() if k.startswith('xrequant_'))])
                if name.startswith(('projection_accumulate_i8','requantize_')):snippets.extend(f['lines']+[''])
            (args.out_dir/f'{backend}_kernels.dis').write_text('\n'.join(snippets)+'\n')
    if dynamic:
        with (args.out_dir/'dynamic_regions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dynamic[0]));writer.writeheader();writer.writerows(dynamic)
    report=['# Scalar and custom instruction comparison\n',
        'Counts below describe instruction retirement and static assembly. They do not estimate accelerator cycle latency or RTL speedup. No counter-overhead correction is applied.\n',
        '## Assembly checks\n','Scalar: no XTILE/XREQUANT opcodes found. Custom: both RUN opcodes found. Detection uses encoded opcode/funct3 fields, including when objdump displays `.4byte` instead of a mnemonic.\n',
        '| Custom operation | Static instruction sites |','|---|---:|']
    report += [f'| {name} | {count} |' for name,count in sorted(custom['custom_totals'].items())]
    report += ['','Static sites are not execution counts. See `static_functions.csv` and the two extracted kernel disassemblies.']
    if dynamic:
        grouped=defaultdict(lambda:[0,0])
        for row in dynamic:
            if row['block']==-1 and row['region']=='chain':grouped['whole_chain'][0]+=row['scalar_raw_minstret'];grouped['whole_chain'][1]+=row['custom_raw_minstret']
            elif row['region'] in STAGES:grouped[row['region']][0]+=row['scalar_raw_minstret'];grouped[row['region']][1]+=row['custom_raw_minstret']
        count_cases=len({row['case'] for row in dynamic});denominator=sum(grouped[s][1] for s in STAGES)
        report += ['','## Raw retired-instruction comparison\n',f'Totals aggregate {count_cases} matched case(s); stage rows also aggregate all eight blocks. Whole-chain and stage intervals are separately measured.\n',
            '| Region | Scalar raw | Custom raw | Reduction | Scalar/custom instruction ratio | Share of custom stage intervals |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('whole_chain',*STAGES):
            a,b=grouped[name];share='—' if name=='whole_chain' else f'{100*b/denominator:.4f}%'
            report.append(f'| {name} | {a} | {b} | {100*(a-b)/a:.4f}% | {a/b:.3f} | {share} |')
        report += ['','The ratio is a retired-instruction ratio, not a cycle speedup. Per-block results and raw aggregate rows are in `dynamic_regions.csv`. The sum of stage intervals can differ from a complete-block or chain interval because their call paths and measurement boundaries differ.']
    else:report += ['','Dynamic comparison is pending: rerun with both validated profile logs.']
    (args.out_dir/'report.md').write_text('\n'.join(report)+'\n')
    serializable={'static':{backend:{'custom_totals':dict(data['custom_totals']),
        'functions':{name:{k:dict(v) if isinstance(v,Counter) else v for k,v in f.items() if k!='lines'}
                     for name,f in data['functions'].items()}} for backend,data in [('scalar',scalar),('custom',custom)]},
        'dynamic':dynamic,'counter_overhead_correction':None,'cycle_speedup_claimed':False}
    (args.out_dir/'comparison.json').write_text(json.dumps(serializable,indent=2)+'\n')
    print(f'PASS: scalar/custom opcode checks; wrote {args.out_dir}/report.md')


if __name__=='__main__':main()

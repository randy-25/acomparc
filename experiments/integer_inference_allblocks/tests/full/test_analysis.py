"""Decoder/log validation tests use synthetic instruction records, not measurements."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('comparison',ROOT/'python/analyze_instruction_comparison.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AnalysisTests(unittest.TestCase):
    def test_unknown_custom_mnemonics_and_compressed_instruction(self):
        scalar=module.parse_disassembly('file format elf64-littleriscv\n80000000 <dot>:\n 80000000: 00a58533 add a0,a1,a0\n 80000004: 0001 nop\n')
        custom=module.parse_disassembly('file format elf64-littleriscv\n80000000 <run>:\n 80000000: 0000400b .4byte 0x400b\n 80000004: 0000402b .insn 0x402b\n')
        module.validate_disassembly_pair(scalar,custom)
        self.assertEqual(scalar['functions']['dot']['instructions'],2)
        self.assertEqual(scalar['functions']['dot']['bytes'],6)
        self.assertEqual(custom['custom_totals']['xtile_run'],1)
        with self.assertRaisesRegex(ValueError,'Scalar disassembly'):
            module.validate_disassembly_pair(custom,custom)

    def profile(self,backend):
        rows=[f'PROFILE,{backend},0,-1,chain,801',f'PROFILE,{backend},0,-1,sum_block_raw,800']
        for b in range(8):
            rows += [f'PROFILE,{backend},0,{b},forward,100',f'PROFILE,{backend},0,{b},sum_stage_raw,70']
            rows += [f'PROFILE,{backend},0,{b},{s},10' for s in module.STAGES]
        return '\n'.join(rows)+f'\nPASS: backend={backend} mode=profile cases=1; outputs match\n'

    def test_profile_completeness_and_aggregate_checks(self):
        a=module.parse_profile(self.profile('scalar'),'scalar')
        b=module.parse_profile(self.profile('custom'),'custom')
        self.assertEqual(len(module.comparison_rows(a,b)),74)
        with self.assertRaises(ValueError):module.parse_profile(self.profile('scalar').replace('sum_stage_raw,70','sum_stage_raw,71'),'scalar')
        with self.assertRaises(ValueError):module.parse_profile(self.profile('scalar').split('PASS:')[0],'scalar')

    def test_joined_header_retains_chain_row(self):
        original=self.profile('scalar')
        joined=module.PROFILE_HEADER+original
        normal=module.PROFILE_HEADER+'\n'+original
        self.assertEqual(module.parse_profile(joined,'scalar'),module.parse_profile(normal,'scalar'))
        self.assertEqual(module.parse_profile(joined,'scalar')[(0,-1,'chain')],801)

    def test_truly_missing_or_extra_rows_are_rejected(self):
        text=self.profile('scalar')
        with self.assertRaisesRegex(ValueError,'missing profile regions'):
            module.parse_profile(text.replace('PROFILE,scalar,0,-1,chain,801\n',''),'scalar')
        with self.assertRaisesRegex(ValueError,'unexpected profile regions'):
            module.parse_profile('PROFILE,scalar,0,0,extra,1\n'+text,'scalar')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            module.parse_profile('PROFILE,scalar,0,-1,chain,801\n'+text,'scalar')

    def test_completion_case_count_is_enforced(self):
        with self.assertRaisesRegex(ValueError,'expected cases'):
            module.parse_profile(self.profile('scalar').replace('cases=1;','cases=2;'),'scalar')

    def test_cli_outputs_with_explicit_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'scalar.dis').write_text('file format elf64-littleriscv\n80000000 <projection_accumulate_i8>:\n 80000000: 00a58533 add a0,a1,a0\n')
            (root/'custom.dis').write_text('file format elf64-littleriscv\n80000000 <projection_accumulate_i8>:\n 80000000: 0000400b .4byte 0x400b\n80000004 <requantize_buffer>:\n 80000004: 0000402b .4byte 0x402b\n')
            for backend in ('scalar','custom'):(root/f'{backend}.log').write_text(self.profile(backend))
            subprocess.run([sys.executable,str(ROOT/'python/analyze_instruction_comparison.py'),
                '--scalar-dis',str(root/'scalar.dis'),'--custom-dis',str(root/'custom.dis'),
                '--scalar-log',str(root/'scalar.log'),'--custom-log',str(root/'custom.log'),
                '--out-dir',str(root/'analysis')],check=True,capture_output=True)
            self.assertTrue((root/'analysis/dynamic_regions.csv').exists())
            self.assertIn('not a cycle speedup',(root/'analysis/report.md').read_text())


if __name__=='__main__':unittest.main()

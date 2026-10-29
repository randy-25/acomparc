"""Check archive integrity and preservation without optional model dependencies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src/python'))
from tinystories_integer.storage import save_archive,load_archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.profile=json.loads((PROJECT/'models/profiles/int8_boundaries_v1.json').read_text())
        exponents=self.profile['boundary_scale_exponents']
        shared=np.array([-128,0,127],np.int8)
        self.blocks=[{'block_index':i,'input_scale':2.**exponents[i],
            'output_scale':2.**exponents[i+1],'weights':shared,
            'multiplier':5266217801} for i in range(8)]

    def test_round_trip_retains_large_multiplier_and_deduplicates(self):
        manifest=save_archive(self.directory,self.blocks,self.profile)
        self.assertEqual(len(manifest['arrays']),1)
        blocks,profile=load_archive(self.directory)
        self.assertEqual(profile,self.profile)
        self.assertEqual(blocks[1]['multiplier'],5266217801)
        np.testing.assert_array_equal(blocks[7]['weights'],self.blocks[7]['weights'])
        self.assertFalse(blocks[7]['weights'].flags.writeable)
        with self.assertRaises(FileExistsError):
            save_archive(self.directory,self.blocks,self.profile)

    def test_corrupt_npz_is_rejected(self):
        save_archive(self.directory,self.blocks,self.profile)
        with (self.directory/'parameters.npz').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError,'checksum'):
            load_archive(self.directory)

    def test_incompatible_scale_is_rejected(self):
        blocks=copy.deepcopy(self.blocks)
        blocks[1]['input_scale']=2.**-8
        save_archive(self.directory,blocks,self.profile)
        with self.assertRaisesRegex(ValueError,'scales'):
            load_archive(self.directory)


if __name__=='__main__':
    unittest.main()

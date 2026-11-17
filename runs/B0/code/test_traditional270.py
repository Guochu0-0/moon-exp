"""Synthetic MAT/manifest regressions for #270; CPU/server, no MATLAB/data/GPU."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

spec = importlib.util.spec_from_file_location('traditional270', Path(__file__).with_name('traditional270.py'))
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


class CollectionBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.io = self.root / 'io'
        self.out = self.io / 'rift/val/out'
        self.out.mkdir(parents=True)
        self.pairs = ['ROI_001/patch_1', 'ROI_001/patch_2']
        self.cfg = {'method': 'rift', 'repo': 'third_party/RIFT', 'upstream_commit': 'fixed'}
        config = self.root / 'runs/B0/configs'
        config.mkdir(parents=True)
        (config / 'rift_270.json').write_text(json.dumps(self.cfg))
        self.manifest(self.pairs)
        for pair in self.pairs:
            self.mat(pair)
        ds = unittest.mock.Mock()
        ds.pairs.return_value = self.pairs
        for p in [patch.object(target, 'REPO', self.root), patch.object(target, 'Data', return_value=ds),
                  patch.object(target, 'require_clean'), patch.object(target, 'git_head', return_value='fixed'),
                  patch.object(target, 'env_info', return_value={})]:
            p.start()
            self.addCleanup(p.stop)

    def manifest(self, pairs):
        (self.out.parent / 'manifest.json').write_text(json.dumps(
            {'pairs': pairs, 'config': self.cfg, 'commit': 'fixed', 'selfcheck': False}))

    def mat(self, pair, error=''):
        p = np.array([[1., 2.], [4., 5.], [7., 8.]])
        savemat(self.out / (target.key(pair) + '.mat'), {'p1': p, 'p2': p+[7, -5], 'sec': 0.1, 'err': error})

    def run_collect(self, *extra):
        argv = ['traditional270', 'collect', '--method', 'rift', '--split', 'val',
                '--io', str(self.io), '--data', 'unused', *extra]
        with patch.object(sys, 'argv', argv):
            target.main()

    def rejected_without_output(self, *extra):
        with self.assertRaises((ValueError, RuntimeError, SystemExit)):
            self.run_collect(*extra)
        self.assertFalse((self.root / 'runs/B0/raw').exists(), 'Rejected batch published raw')

    def test_all_environment_errors_rejected(self):
        for p in self.pairs:
            self.mat(p, "MATLAB:UndefinedFunction: Undefined function 'matchFeatures'")
        self.rejected_without_output()

    def test_one_error_rejected(self):
        self.mat(self.pairs[1], 'MATLAB:license:checkout: License checkout failed')
        self.rejected_without_output()

    def test_missing_result_rejected(self):
        (self.out / (target.key(self.pairs[1]) + '.mat')).unlink()
        self.rejected_without_output()

    def test_partial_manifest_rejected(self):
        self.manifest(self.pairs[:1])
        self.rejected_without_output()

    def test_selfcheck_manifest_cannot_publish_formal_split(self):
        path = self.out.parent / 'manifest.json'
        doc = json.loads(path.read_text())
        doc['selfcheck'] = True
        path.write_text(json.dumps(doc))
        self.rejected_without_output()

    def test_prep_limit_rejected_before_writing_inputs(self):
        with patch.object(sys, 'argv', ['traditional270', 'prep', '--method', 'rift', '--split', 'val',
                                       '--io', str(self.io), '--data', 'unused', '--limit', '1']):
            with self.assertRaises(ValueError):
                target.main()
        self.assertFalse((self.out.parent / 'in').exists())

    def test_limit_cannot_publish_formal_split(self):
        self.manifest(self.pairs[:1])
        self.rejected_without_output('--limit', '1')

    def test_invalid_coordinates_rejected(self):
        savemat(self.out / (target.key(self.pairs[1]) + '.mat'),
                {'p1': np.array([[np.nan, 2.]]), 'p2': np.array([[1., 2.]]), 'sec': 0.1, 'err': ''})
        self.rejected_without_output()

    def test_corrupt_mat_rejected(self):
        (self.out / (target.key(self.pairs[1]) + '.mat')).write_bytes(b'corrupt mat')
        with self.assertRaises(Exception):
            self.run_collect()
        self.assertFalse((self.root / 'runs/B0/raw').exists())

    def test_serialization_failure_preserves_formal_outputs(self):
        dest = self.root / 'runs/B0/raw/rift'
        dest.mkdir(parents=True)
        (dest / 'val.npz').write_bytes(b'old matches')
        with patch.object(target.np, 'savez_compressed', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.run_collect()
        self.assertEqual((dest / 'val.npz').read_bytes(), b'old matches')
        self.assertFalse((dest / 'val.jsonl').exists())
        self.assertFalse((dest / 'val.meta.json').exists())

    def test_rejection_preserves_existing_raw(self):
        dest = self.root / 'runs/B0/raw/rift'
        dest.mkdir(parents=True)
        old = {'val.npz': b'old matches', 'val.jsonl': b'old log', 'val.meta.json': b'old meta'}
        for name, data in old.items():
            (dest / name).write_bytes(data)
        self.mat(self.pairs[1], 'MATLAB:UndefinedFunction: missing function')
        with self.assertRaises((ValueError, RuntimeError, SystemExit)):
            self.run_collect()
        for name, data in old.items():
            self.assertEqual((dest / name).read_bytes(), data)

    def test_valid_complete_batch_publishes_original_coordinates(self):
        self.run_collect()
        dest = self.root / 'runs/B0/raw/rift'
        with np.load(dest / 'val.npz') as z:
            self.assertEqual(len(z.files), 2)
            np.testing.assert_allclose(z[target.key(self.pairs[0])][0], [0, 1, 7, -4, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)

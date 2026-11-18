"""只用微型文件检验：半份拟合不能跳过，已记录的逐对错误仍是完整评测。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from completeness import incomplete_reason


class CompletenessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.raw = self.run / "sweep/main/match/step250"
        self.pred = self.run / "sweep/main/S/preds/step250"
        self.raw.mkdir(parents=True)
        self.pred.mkdir(parents=True)
        self.pairs = {"a/1", "a/2"}
        for directory in (self.raw, self.pred):
            self.write_rows(directory, [{"pair": p} for p in sorted(self.pairs)])
        (self.raw / "val.meta.json").write_text(json.dumps({"finished": "now", "n_pairs": 2}))
        (self.pred / "val.meta.json").write_text(json.dumps({"written": "now"}))
        self.write_matches({"a__1": np.zeros((1, 5)), "a__2": np.zeros((0, 5))})

    def write_rows(self, directory, rows):
        (directory / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def write_matches(self, arrays):
        np.savez(self.raw / "val.npz", **arrays)
        np.savez(self.pred / "val_matches.npz", **arrays)

    def reason(self):
        return incomplete_reason(self.run, 250, "val", self.pairs)

    def test_complete_fit_then_partial_fit_is_not_accepted(self):
        self.assertIsNone(self.reason())
        self.write_rows(self.pred, [{"pair": "a/1"}])
        self.assertIn("coverage", self.reason())

    def test_unflushed_npz_is_not_accepted(self):
        self.write_matches({"a__1": np.zeros((1, 5))})
        self.assertIn("unflushed", self.reason())

    def test_recorded_pair_error_remains_in_complete_evaluation(self):
        self.write_rows(self.raw, [{"pair": "a/1"}, {"pair": "a/2", "error": "timeout"}])
        self.write_matches({"a__1": np.zeros((1, 5))})
        self.assertIsNone(self.reason())


if __name__ == "__main__":
    unittest.main()

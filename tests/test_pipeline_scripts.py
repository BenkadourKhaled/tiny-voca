"""Tests for the CPU pipeline helpers (features and annotation shard merges).

Run with:  python -m unittest tests.test_pipeline_scripts -v
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MergeAnnotationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.features = self.root / "features"
        self.features.mkdir()
        for split in ("train", "validation", "test"):
            write_jsonl(self.features / (split + ".jsonl"), [])

    def tearDown(self):
        self.temp.cleanup()

    def make_shard(self, name, rows, rejects=(), revision="halabi-2016-v1"):
        shard = self.root / name
        shard.mkdir()
        identity = {"annotation_version": 3, "acoustic": "a" * 64, "frontend": "b" * 64,
                    "arabic_frontend": "b" * 64, "aligner": {"model": "m", "revision": "r"},
                    "alignment": {"alignment_version": 2}, "dataset_revision": revision, "limit": None,
                    "sources": {}}
        metadata = {"identity": identity, "fingerprint": "0" * 64, "complete": False, "stage": "ctc",
                    "limits": {"max_weak_phone_fraction": 0.25}, "acoustic_root": str(self.features)}
        write_json(shard / "annotations.json", metadata)
        write_jsonl(shard / "train.jsonl", rows)
        write_jsonl(shard / "train.rejected.jsonl", list(rejects))
        write_jsonl(shard / "phonemes.jsonl", [])
        return shard

    def run_merge(self, shards, output):
        command = [PYTHON, str(ROOT / "scripts" / "merge_annotations.py"),
                   "--shards", *[str(shard) for shard in shards],
                   "--features", str(self.features), "--output", str(output)]
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(command, cwd=str(ROOT), env=environment, capture_output=True,
                              text=True, encoding="utf-8")

    def row(self, identifier):
        return {"id": identifier, "split": "train", "speaker": "F2", "frames": 100,
                "tokens": [1, 2], "durations": [40, 60], "acoustic": "a" * 64,
                "frontend": "b" * 64, "annotation": "0" * 64}

    def test_merge_rebuilds_identity_and_rewrites_rows(self):
        write_jsonl(self.features / "train.jsonl", [self.row("a"), self.row("b"), self.row("c")])
        first = self.make_shard("shard-0", [self.row("a")],
                                [{"id": "b", "reasons": ["weak phones: 0.4", "weak phones: 0.5"]}])
        second = self.make_shard("shard-1", [self.row("c")])
        output = self.root / "merged"
        done = self.run_merge([first, second], output)
        self.assertEqual(done.returncode, 0, done.stderr)
        metadata = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(json.dumps(metadata["identity"], sort_keys=True).encode()).hexdigest()
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["fingerprint"], fingerprint)
        rows = read_jsonl(output / "train.jsonl")
        self.assertEqual([row["id"] for row in rows], ["a", "c"])
        self.assertTrue(all(row["annotation"] == fingerprint for row in rows))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rejection_reasons"], {"weak phones": 2})
        self.assertEqual(summary["rejected"]["train"], 1)
        self.assertEqual(summary["counts"]["train"], 2)
        self.assertEqual(read_jsonl(output / "train.rejected.jsonl")[0]["id"], "b")

    def test_merge_refuses_incomplete_coverage(self):
        write_jsonl(self.features / "train.jsonl", [self.row("a"), self.row("b")])
        shard = self.make_shard("incomplete", [self.row("a")])
        done = self.run_merge([shard], self.root / "merged")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("1 accepted + 0 rejected != 2 feature rows", done.stderr)

    def test_merge_refuses_disagreeing_shards(self):
        write_jsonl(self.features / "train.jsonl", [self.row("a")])
        first = self.make_shard("one", [self.row("a")])
        second = self.make_shard("two", [], revision="another-revision")
        done = self.run_merge([first, second], self.root / "merged")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("disagree on dataset_revision", done.stderr)

    def test_merge_refuses_limited_shards(self):
        write_jsonl(self.features / "train.jsonl", [self.row("a")])
        shard = self.make_shard("limited", [self.row("a")])
        metadata = json.loads((shard / "annotations.json").read_text(encoding="utf-8"))
        metadata["identity"]["limit"] = 3
        write_json(shard / "annotations.json", metadata)
        done = self.run_merge([shard], self.root / "merged")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("made with --limit", done.stderr)


if __name__ == "__main__":
    unittest.main()

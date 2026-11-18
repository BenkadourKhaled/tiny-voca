"""Language-aware defaults of the free-running evaluation script.

Run with:  python -m unittest tests.test_evaluate -v

The front end binds ``TINY_VOCA_LANGUAGE`` when it is imported, so each case
uses a fresh interpreter (the same pattern as ``tests/test_arabic.py``).
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CODE = """
import json
from tiny_voca import evaluate
from tiny_voca.text import language, phonemize
print(json.dumps({
    "language": language(),
    "novel": list(evaluate.novel_sentences()),
    "arabic": list(evaluate.ARABIC_NOVEL_SENTENCES),
    "english": list(evaluate.NOVEL_SENTENCES),
    "normalized": evaluate.normalizer()("أَنا أَشْعُرُ بِالاسْتِرْخاءِ، يَا إِبْراهِيمُ!"),
    "normalized_ascii": evaluate.normalizer()("Hello, WORLD!"),
    "phones": [len(phonemize(text, diacritized=True)) for text in evaluate.ARABIC_NOVEL_SENTENCES] if language() == "ar" else [],
    "unknown": [len(phonemize(text, diacritized=True).unknown) for text in evaluate.ARABIC_NOVEL_SENTENCES] if language() == "ar" else [],
}, ensure_ascii=False))
"""


def run(lang):
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["TINY_VOCA_LANGUAGE"] = lang
    if lang == "ar":
        environment["TINY_VOCA_FRONTEND"] = "rules"
    done = subprocess.run([PYTHON, "-c", CODE], cwd=str(ROOT), env=environment, capture_output=True,
                          text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class EvaluateLanguageTests(unittest.TestCase):
    def test_english_defaults_are_unchanged(self):
        result = run("en")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["novel"], result["english"])
        self.assertNotEqual(result["novel"], result["arabic"])

    def test_arabic_uses_vocalised_novel_sentences(self):
        result = run("ar")
        self.assertEqual(result["language"], "ar")
        self.assertEqual(result["novel"], result["arabic"])
        self.assertTrue(all(any("\u064b" <= c <= "\u065f" for c in text) for text in result["novel"]))
        self.assertTrue(all(size > 0 for size in result["phones"]))
        self.assertTrue(all(size == 0 for size in result["unknown"]))

    def test_arabic_normalizer_folds_orthography(self):
        result = run("ar")
        self.assertEqual(result["normalized"], "انا اشعر بالاسترخاء يا ابراهيم")

    def test_english_normalizer_still_lowercases(self):
        result = run("en")
        self.assertEqual(result["normalized_ascii"], "hello world")

    def test_novel_sentences_are_not_in_the_annotations(self):
        annotations = ROOT / "data" / "halabi" / "annotations"
        manifests = [annotations / (split + ".jsonl") for split in ("train", "validation", "test")]
        if not all(manifest.exists() for manifest in manifests):
            self.skipTest("Halabi annotations are not built in this checkout")
        known = set()
        for manifest in manifests:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    known.add(json.loads(line)["text_hash"])
        for text in run("ar")["novel"]:
            self.assertNotIn(hashlib.sha256(text.strip().encode()).hexdigest(), known)


if __name__ == "__main__":
    unittest.main()

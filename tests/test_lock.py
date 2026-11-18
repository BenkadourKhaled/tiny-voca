import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_voca.journal import atomic_json, exclusive_lock


class ExclusiveLockTests(unittest.TestCase):
    def test_lock_is_released_after_success_and_after_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.lock"
            with exclusive_lock(path):
                pass
            with self.assertRaises(RuntimeError):
                with exclusive_lock(path):
                    raise RuntimeError("caller failed")
            with exclusive_lock(path):
                pass

    def test_second_lock_is_rejected_while_first_is_held(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "other.lock"
            with exclusive_lock(path):
                with self.assertRaises(OSError):
                    with exclusive_lock(path):
                        pass


class AtomicJsonTests(unittest.TestCase):
    def test_temporary_file_is_unique_per_process(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "features.json"
            replacements = []
            real_replace = os.replace

            def record_replace(source, destination):
                replacements.append((Path(source).name, Path(destination).name))
                return real_replace(source, destination)

            with patch("tiny_voca.journal.os.getpid", return_value=12345), patch(
                "tiny_voca.journal.os.replace", side_effect=record_replace
            ):
                atomic_json(target, {"fingerprint": "abc"})

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"fingerprint": "abc"})
            self.assertEqual(replacements, [("features.json.12345.tmp", "features.json")])
            self.assertFalse((Path(folder) / "features.json.12345.tmp").exists())


if __name__ == "__main__":
    unittest.main()

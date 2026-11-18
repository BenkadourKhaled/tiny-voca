"""Tests for the Arabic (MSA) front end, Phase 1.

Run with:  python -m unittest tests.test_arabic -v
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_voca import arabic_text as ar
from tiny_voca.arabic_espeak import canonical, parse_espeak


def phones(word):
    return "".join(phone for phone, _ in ar.word_to_phones(word)[0])


class ArabicInventoryTests(unittest.TestCase):
    def test_inventory_is_unique_and_contiguous(self):
        self.assertEqual(len(ar.SYMBOLS), len(set(ar.SYMBOLS)))
        self.assertEqual(ar.SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual([ar.SYMBOL_TO_ID[s] for s in ar.SYMBOLS], list(range(len(ar.SYMBOLS))))
        self.assertTrue(ar.PHONE_SET.issubset(set(ar.SYMBOLS)))
        self.assertIn(ar.SYMBOL_TO_ID[ar.UNK], ar.PHONE_IDS)

    def test_inventory_is_a_subset_of_the_aligner_labels(self):
        # Saved from facebook/wav2vec2-lv-60-espeak-cv-ft/vocab.json on 2026-11-17.
        vocab = json.loads((ROOT / "data" / "aligner_vocab.json").read_text(encoding="utf-8"))
        missing = sorted(set(ar.PHONES) - set(vocab))
        self.assertEqual(missing, [], "align.py would reject these inventory phones")
        self.assertEqual(len(vocab), 392)
        # The two labels that matter most for Arabic must really be there.
        for phone in ("ʔ", "ħ", "ʕ", "χ", "ɣ", "dˤ", "s̪", "t̪", "dʒ", "aː", "iː", "uː"):
            self.assertIn(phone, vocab)

    def test_fingerprint_tracks_inventory_and_version(self):
        baseline = ar.frontend_fingerprint()
        self.assertEqual(len(baseline), 64)
        with patch.object(ar, "FRONTEND_VERSION", 999):
            self.assertNotEqual(ar.frontend_fingerprint(), baseline)
        with patch.object(ar, "SYMBOLS", ar.SYMBOLS + ("new",)):
            self.assertNotEqual(ar.frontend_fingerprint(), baseline)

    def test_frontend_selection(self):
        self.assertEqual(ar.frontend(), "rules")
        ar.set_frontend("rules")
        with self.assertRaises(ValueError):
            ar.set_frontend("en")
        self.assertEqual(ar.frontend(), "rules")


class ArabicNormalizationTests(unittest.TestCase):
    def test_unifies_alef_ya_tatweel_and_whitespace(self):
        self.assertEqual(ar.normalize("مرحبا  بالعالم"), "مرحبا بالعالم")
        self.assertEqual(ar.normalize("أإآٱـ"), "اااا")
        self.assertEqual(ar.normalize("كتابٌ"), "كتابٌ")  # tashkeel survives

    def test_numbers_become_arabic_words(self):
        self.assertEqual(ar.normalize("١٢٣"), "مائة وثلاثة وعشرون")
        self.assertEqual(ar.normalize("٤٥٦"), "أربعة مائة وستة وخمسون")
        self.assertEqual(ar.normalize("٠"), "صفر")
        self.assertEqual(ar.normalize("١٠٠٠"), "ألف")

    def test_quotes_and_punctuation_are_normalised(self):
        self.assertEqual(ar.normalize("“مرحبا” — ١٠؟"), '"مرحبا" — عشرة؟')
        self.assertTrue(ar.has_diacritics("كَتَبَ"))
        self.assertFalse(ar.has_diacritics("كتب"))


class ArabicG2PTests(unittest.TestCase):
    def test_plain_word_and_long_vowels(self):
        self.assertEqual(phones("كَتَبَ"), "kataba")
        self.assertEqual(phones("كِتَابٌ"), "kitaːbun")
        self.assertEqual(phones("كِتَابٍ"), "kitaːbin")
        self.assertEqual(phones("كِتَابًا"), "kitaːban")
        self.assertEqual(phones("مُدَرِّسٌ"), "mudarisun")

    def test_sun_and_moon_article(self):
        self.assertEqual(phones("الْوَلَدُ"), "ʔalwaladu")
        self.assertEqual(phones("الشَّمْسُ"), "ʔaʃʃamsu")
        self.assertEqual(phones("الطَّالِبُ"), "ʔat̪t̪aːlibu")
        self.assertEqual(phones("اللَّهِ"), "ʔallahi")
        self.assertEqual(phones("بِالْعالَمِ"), "biʔalʕalami")

    def test_hamza_and_final_ya(self):
        self.assertEqual(phones("الْأَبُ"), "ʔalʔabu")
        self.assertEqual(phones("إِلَى"), "ʔilaː")
        self.assertEqual(phones("آدَمُ"), "ʔaːdamu")
        self.assertEqual(phones("ظالِمٌ"), "ðalimun")  # alif here is a mater, not ال

    def test_emphatics_and_velars_use_aligner_labels(self):
        self.assertEqual(phones("صادٌ"), "s̪adun")
        self.assertEqual(phones("ضادٌ"), "dˤadun")
        self.assertEqual(phones("ظالِمٌ"), "ðalimun")
        self.assertEqual(phones("طَالِبٌ"), "t̪aːlibun")
        self.assertEqual(phones("خَبَرٌ"), "χabarun")
        self.assertEqual(phones("غَفَرَ"), "ɣafara")
        self.assertTrue(phones("الْجَامِعَةُ").startswith("ʔaldʒaːmiʕa"))

    def test_stress_is_a_feature_not_a_token(self):
        result, unknown = ar.word_to_phones("كَتَبَ")
        self.assertIsNone(unknown)
        self.assertEqual(result[0][1], 0)
        self.assertEqual(result[1][1], 0)
        self.assertEqual(result[-1][1], 1)
        self.assertTrue(all(p in ar.PHONE_SET for p, _ in result))

    def test_unknown_graphemes_are_reported(self):
        result = ar.word_to_phones("كتبX")
        self.assertEqual(result[1], "X")
        self.assertNotIn("X", "".join(p for p, _ in result[0]))


class ArabicTokenizeTests(unittest.TestCase):
    def test_word_boundaries_ids_and_unknown(self):
        result = ar.tokenize([[("m", 0), ("a", 1)], [("b", 0), ("ʘ", 0)]], unknown=("ʘ",))
        self.assertEqual(result.symbols, (ar.BOS, "m", "a", ar.SP, "b", ar.UNK, ar.EOS))
        self.assertEqual(result.stress, (0, 0, 1, 0, 0, 0, 0))
        self.assertEqual(result.ids, tuple(ar.SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(result.unknown, ("ʘ",))

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            ar.phonemize("   ")
        with self.assertRaises(ar.NothingToPronounce):
            ar.tokenize([[]])

    def test_phonemize_diacritized_sentence(self):
        result = ar.phonemize("مَرْحَبًا بِالْعالَمِ", diacritized=True)
        self.assertEqual(result.symbols,
                         (ar.BOS, "m", "a", "r", "ħ", "a", "b", "a", "n", ar.SP,
                          "b", "i", "ʔ", "a", "l", "ʕ", "a", "l", "a", "m", "i", ar.EOS))
        self.assertEqual(result.unknown, ())

    def test_parse_espeak_output(self):
        words = parse_espeak("m_r_ħ_b_ˈaː b_i_l_ʕ_ˈaː_l_a_m\n")
        self.assertEqual(words[0], [("m", 0), ("r", 0), ("ħ", 0), ("b", 0), ("aː", 1)])
        self.assertEqual(words[1][3], ("ʕ", 0))
        self.assertEqual(words[1][4], ("aː", 1))

    def test_espeak_allophones_are_mapped_into_the_inventory(self):
        self.assertEqual(canonical("ħ"), (("ħ",), None))
        self.assertEqual(canonical("a.ː"), (("aː",), None))       # centred vowel
        self.assertEqual(canonical("i."), (("i",), None))
        self.assertEqual(canonical("t̪ː"), (("t̪", "t̪"), None))  # gemination
        self.assertEqual(canonical("θ"), (("s",), None))
        self.assertEqual(canonical("s̪-"), (("s̪",), None))
        mapped, unmapped = canonical("ʘ")
        self.assertEqual(mapped, (ar.UNK,))
        self.assertEqual(unmapped, "ʘ")


class LanguageIntegrationTests(unittest.TestCase):
    def _run(self, env):
        code = (
            "import sys; sys.path.insert(0, {!r}); "
            "from tiny_voca import text; from tiny_voca.config import Config; "
            "sample=(' '.join(text.phonemize('مَرْحَبًا بِالْعالَمِ', diacritized=True).symbols) "
            "if text.language()=='ar' else 'en-phonemize-skipped'); "
            "tag=text.frontend_identity()['language'] if text.language()=='ar' else 'en'; "
            "print(text.language(), len(text.SYMBOLS), Config().vocab_size, sample, tag)"
        ).format(str(ROOT / "src"))
        environment = dict(os.environ)
        environment.pop("TINY_VOCA_LANGUAGE", None)
        environment.update(env)
        environment["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run([sys.executable, "-c", code], env=environment,
                              capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()

    def test_english_remains_the_default(self):
        out = self._run({})
        self.assertEqual(out.split(), ["en", "76", "76", "en-phonemize-skipped", "en"])

    def test_arabic_switches_inventory_and_g2p(self):
        out = self._run({"TINY_VOCA_LANGUAGE": "ar"})
        parts = out.split()
        self.assertEqual(parts[0], "ar")
        self.assertEqual(parts[1], parts[2])
        self.assertEqual(parts[1], str(len(ar.SYMBOLS)))
        self.assertIn("ʔalʕalami", "".join(parts[3:-1]))
        self.assertEqual(parts[-1], "ar")

    def test_arabic_accepts_explicit_rules_frontend(self):
        out = self._run({"TINY_VOCA_LANGUAGE": "ar", "TINY_VOCA_FRONTEND": "rules"})
        parts = out.split()
        self.assertEqual(parts[0], "ar")
        self.assertEqual(parts[1], str(len(ar.SYMBOLS)))
        self.assertEqual(parts[-1], "ar")

    def test_english_still_rejects_arabic_frontend(self):
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self._run({"TINY_VOCA_FRONTEND": "rules"})
        self.assertIn("TINY_VOCA_FRONTEND must be one of", caught.exception.stderr)


class ArabicRuntimeTests(unittest.TestCase):
    """Inference and export helpers must follow the selected language."""

    def _run(self, code, env):
        environment = dict(os.environ)
        environment.pop("TINY_VOCA_LANGUAGE", None)
        environment.pop("TINY_VOCA_FRONTEND", None)
        environment.update(env)
        environment["PYTHONIOENCODING"] = "utf-8"
        done = subprocess.run([sys.executable, "-c", code], env=environment,
                              capture_output=True, text=True, encoding="utf-8", check=True)
        return json.loads(done.stdout.strip())

    def test_arabic_sentences_finished_and_metadata(self):
        code = (
            "import sys, json; sys.path.insert(0, {!r}); "
            "from types import SimpleNamespace; from unittest.mock import patch; "
            "from tiny_voca import infer, export; from tiny_voca.config import Config; "
            "text='مَرْحَبًا بِالْعالَمِ. كَيْفَ حالُكَ؟ خَيْرٌ، وَالْحَمْدُ لِلّٰهِ!';\n"
            "with patch('torch.load', return_value={{'step': 1}}):\n"
            "  meta=export.metadata(SimpleNamespace(config=Config()), 'unused.pt')\n"
            "print(json.dumps({{'sentences': infer.sentences(text), "
            "'finished': infer.finished('مَرْحَبًا'), "
            "'marked': infer.finished('كَيْفَ حالُكَ؟'), "
            "'frontend': meta['frontend'], 'language': meta['language'], "
            "'source': meta.get('data_source'), "
            "'message': infer._nothing_read_message()}}, ensure_ascii=False))"
        ).format(str(ROOT / "src"))
        out = self._run(code, {"TINY_VOCA_LANGUAGE": "ar", "TINY_VOCA_FRONTEND": "rules"})
        self.assertEqual(out["sentences"], ["مَرْحَبًا بِالْعالَمِ.", "كَيْفَ حالُكَ؟",
                                            "خَيْرٌ، وَالْحَمْدُ لِلّٰهِ!"])
        self.assertEqual(out["finished"], "مَرْحَبًا.")
        self.assertEqual(out["marked"], "كَيْفَ حالُكَ؟")
        self.assertEqual(out["frontend"], "rules")
        self.assertEqual(out["language"], "ar")
        self.assertEqual(out["source"]["license"], "CC BY 4.0")
        self.assertIn("Halabi", out["source"]["citation"])
        self.assertIn("Arabic", out["message"])

    def test_english_metadata_is_unchanged(self):
        code = (
            "import sys, json; sys.path.insert(0, {!r}); "
            "from types import SimpleNamespace; from unittest.mock import patch; "
            "from tiny_voca import export; from tiny_voca.config import Config;\n"
            "with patch('torch.load', return_value={{'step': 1}}):\n"
            "  meta=export.metadata(SimpleNamespace(config=Config()), 'unused.pt')\n"
            "print(json.dumps({{'frontend': meta['frontend'], 'language': meta['language']}}))"
        ).format(str(ROOT / "src"))
        self.assertEqual(self._run(code, {}), {"frontend": "espeak", "language": "en"})


if __name__ == "__main__":
    unittest.main()

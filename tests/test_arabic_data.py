import json
import tempfile
import unittest
from pathlib import Path

from tiny_voca import arabic_data as ad
from tiny_voca import arabic_text


class BuckwalterTests(unittest.TestCase):
    def test_full_harakat_conversion(self):
        self.assertEqual(ad.buckwalter_to_arabic("Alt~aqoriyru"), "التَّقْرِيرُ")
        self.assertEqual(ad.buckwalter_to_arabic("Al~a*iy"), "الَّذِي")
        self.assertEqual(ad.buckwalter_to_arabic("|soyaA"), "آسْيَا")
        self.assertEqual(ad.buckwalter_to_arabic(">aboHaA^i"), "أَبْحَائِ")

    def test_xml_safe_and_corpus_aliases(self):
        self.assertEqual(ad.buckwalter_to_arabic("I<O&W})"), "إإأؤؤئ)")

    def test_leading_mark_restores_counted_implicit_alif(self):
        from collections import Counter
        stats = Counter()
        self.assertEqual(ad.buckwalter_to_arabic("al~atiy", stats), "اَلَّتِي")
        self.assertEqual(stats["implicit_alif_marks"], 1)
        # The mark attaches to the restored carrier, not to the next letter.
        stats.clear()
        self.assertEqual(ad.buckwalter_to_arabic("an~a", stats), "اَنَّ")
        self.assertEqual(stats["implicit_alif_marks"], 1)

    def test_clause_separator_becomes_inventory_punctuation(self):
        from tiny_voca import arabic_text
        text = ad.buckwalter_to_arabic("kitAb- kitAb")
        self.assertIn("—", text)
        self.assertNotIn("-", text)
        report = ad.g2p_report([{"text": text}])
        self.assertEqual(report["unknown_graphemes"], {})

    def test_unmapped_character_is_an_error(self):
        with self.assertRaises(ValueError):
            ad.buckwalter_to_arabic("كتاب@")
        # A restored alif must not make arbitrary marks acceptable.
        with self.assertRaises(ValueError):
            ad.buckwalter_to_arabic("كتاب@u")

    def test_converted_text_is_not_lower_cased_by_g2p(self):
        previous = arabic_text.frontend()
        arabic_text.set_frontend("rules")
        try:
            text = ad.buckwalter_to_arabic(">aEad~ahu")
            report = ad.g2p_report([{"text": text}])
            self.assertEqual(report["unknown_graphemes"], {})
            self.assertTrue(arabic_text.has_diacritics(text))
            self.assertIn("ʔ", " ".join(arabic_text.phonemize(text, diacritized=True).symbols))
        finally:
            arabic_text.set_frontend(previous)


class PhoneticCrossCheckTests(unittest.TestCase):
    def test_corpus_labels_map_to_inventory(self):
        phones, unmapped = ad.corpus_phones("sil w a r a' jj A H a tt sil")
        self.assertEqual(phones, ["w", "a", "r", "a", "dʒ", "aː", "ħ", "a", "t"])
        self.assertEqual(unmapped, {})

    def test_unmapped_labels_are_counted(self):
        phones, unmapped = ad.corpus_phones("sil q weird")
        self.assertEqual(phones, ["q"])
        self.assertEqual(unmapped, {"weird": 1})

    def test_all_labels_observed_in_the_archive_are_mapped(self):
        observed = (
            "$$ ** - << AA AH Ah DD EE I1 II0 J SH TH TT U1 UU0 UU1 ZZ ^^ "
            "bb dist ff gg hh ii1 kk p pp qq u1 uu1 ww xx yy zz"
        ).split()
        phones, unmapped = ad.corpus_phones(" ".join(observed))
        self.assertEqual(unmapped, {})
        self.assertTrue(phones)

    def test_agreement(self):
        agreement = ad.sequence_agreement(["a", "b", "c"], ["a", "x", "c"])
        self.assertEqual(agreement["agreement"], round(2 / 3, 4))
        self.assertEqual(agreement["similarity"], round(2 / 3, 4))
        self.assertEqual(agreement["length_ratio"], 1.0)
        self.assertIsNone(ad.sequence_agreement([], ["a"]))

    def test_similarity_tolerates_a_shifted_sequence(self):
        agreement = ad.sequence_agreement(["a", "b", "c"], ["ʔ", "a", "b", "c"])
        self.assertEqual(agreement["agreement"], 0.0)  # positional artefact
        self.assertEqual(agreement["similarity"], 0.75)
        self.assertEqual(ad.sequence_agreement(["a", "b"], ["a", "b"])["similarity"], 1.0)


class TranscriptTests(unittest.TestCase):
    def test_exact_names_and_double_spaces(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "orthographic-transcript.txt"
            path.write_text('"ARA NORM  0002.wav" "Alt~aqoriyru"\r\n"ARA NORM  0003.wav" "mim~aA"\r\n',
                            encoding="utf-8")
            parsed = ad.parse_transcript(path)
        self.assertEqual(parsed["ARA NORM  0002.wav"], "Alt~aqoriyru")
        self.assertEqual(len(parsed), 2)

    def test_duplicate_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.txt"
            path.write_text('"a.wav" "x"\n"a.wav" "y"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                ad.parse_transcript(path)


class SplitTests(unittest.TestCase):
    def test_no_text_leak_and_repeats_stay_together(self):
        rows = []
        for number, text in enumerate(["الشَّمْسُ", "الْقَمَرُ", "كِتَابٌ", "الشَّمْسُ"]):
            rows.append(ad.manifest_row(f"r{number}", text, f"../wav24/r{number}.wav",
                                        "test" if number == 0 else "train", f"REC {number}"))
        splits = ad.assign_splits(rows, validation_fraction=0.34, seed=3)
        hashes = {split: {ad.text_hash(row["text"]) for row in values} for split, values in splits.items()}
        for left in hashes:
            for right in hashes:
                if left < right:
                    self.assertFalse(hashes[left] & hashes[right])
        sun = ad.text_hash("الشَّمْسُ")
        self.assertEqual({split for split, values in splits.items()
                          if any(ad.text_hash(row["text"]) == sun for row in values)}, {"test"})
        self.assertEqual(sum(len(value) for value in splits.values()), len(rows))

    def test_manifest_row_keeps_provenance(self):
        row = ad.manifest_row("halabi_train_1", "كِتَابٌ", "../wav24/a.wav", "train", "ARA NORM  0001.wav")
        self.assertEqual(row["voice_id"], ad.VOICE_ID)
        self.assertEqual(row["source_license"], ad.SOURCE_LICENSE)
        self.assertEqual(row["source_citation"], ad.SOURCE_CITATION)
        self.assertEqual(row["text_id"], "ARA NORM  0001.wav")

    def test_write_manifest_is_jsonl(self):
        row = ad.manifest_row("halabi_train_1", "كِتَابٌ", "../wav24/a.wav", "train", "REC")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl"
            ad.write_manifest(path, [row])
            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(loaded, [row])


if __name__ == "__main__":
    unittest.main()

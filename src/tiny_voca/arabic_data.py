"""Phase 2 data layer for the Halabi Arabic Speech Corpus.

The corpus stores its transcripts in Buckwalter ASCII with full harakat, not in
Arabic script. This module converts that transliteration to Arabic text, builds
tiny-voca manifests and computes the train/validation/test split. It is pure
Python so it can be tested without the 1.2 GB archive.

Facts verified from the official distribution (November 2026):

* 1813 train + 100 test recordings, one studio voice, Damascene accent;
* ``lab/`` files are Buckwalter (``~`` shadda, ``o`` sukun, ``a/u/i`` vowels);
* filenames contain two spaces (``ARA NORM  0002.wav``) and must be matched
  exactly against the transcript and TextGrid files;
* licence CC BY 4.0; citation below.

No third-party dependency: the Buckwalter table is written out explicitly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path

SOURCE_URL = "http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip"
SOURCE_PAGE = "http://en.arabicspeechcorpus.com/"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_CITATION = (
    "Halabi, Nawar (2016). Modern standard Arabic phonetics for speech "
    "synthesis. PhD thesis, University of Southampton."
)
# The corpus has one studio voice; tiny-voca's existing female slot is reused.
VOICE_ID = "F2"

# Standard Buckwalter table (Tim Buckwalter / ALPNET, as documented on the
# Buckwalter transliteration reference). The corpus uses the XML-unsafe
# characters, plus '^' for hamza on ya instead of the standard '}'.
_LETTERS = {
    "A": "ا", "b": "ب", "t": "ت", "v": "ث", "j": "ج", "H": "ح", "x": "خ",
    "d": "د", "*": "ذ", "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص",
    "D": "ض", "T": "ط", "Z": "ظ", "E": "ع", "g": "غ", "G": "غ", "f": "ف",
    "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و",
    "y": "ي", "Y": "ى", "p": "ة",
    "'": "ء", ">": "أ", "<": "إ", "&": "ؤ", "}": "ئ", "^": "ئ",
    "|": "آ", "{": "ٱ", "`": "ٰ",
    "I": "إ", "O": "أ", "W": "ؤ",  # XML-safe aliases
}
_MARKS = {
    "a": "َ", "u": "ُ", "i": "ِ", "o": "ْ", "~": "ّ",
    "F": "ً", "N": "ٌ", "K": "ٍ", "_": "ـ",
}
# Only punctuation that survives into the Arabic text. The corpus uses an
# ASCII '-' as a clause separator. phonemize(diacritized=True) deliberately
# bypasses normalize(), so emit the em dash already present in the validated
# 54-symbol inventory instead of an ASCII hyphen reported as unknown.
_PUNCTUATION = frozenset(".,;:!?()-")
_PUNCT_MAP = {"-": "—"}


def buckwalter_to_arabic(text, stats=None):
    """Buckwalter string -> Arabic script with harakat.

    Diacritics are attached to the preceding base letter so that a later
    ``arabic_text`` G2P pass sees ordinary NFC Arabic graphemes.

    The corpus occasionally drops the carrier alif of a word-initial vowel
    (``istiEdAdan`` for استعداداً, ``al~atiy`` for الذي, ``an~a`` for أنَّ).
    Such a mark is attached to an explicit, counted implicit alif rather than
    being dropped or silently accepted. ``stats`` receives the counter key
    ``implicit_alif_marks`` so the build report can expose it. Genuinely
    unknown characters still raise: nothing is guessed.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    out = []  # [base, marks] for letters; [punctuation, None] otherwise
    last_is_letter = False
    implicit = 0
    for index, char in enumerate(text):
        if char in _LETTERS:
            out.append([_LETTERS[char], []])
            last_is_letter = True
        elif char in _MARKS:
            if not last_is_letter:
                # Restore the omitted carrier instead of losing the vowel.
                out.append(["ا", []])
                implicit += 1
                last_is_letter = True
            out[-1][1].append(_MARKS[char])
        elif char.isspace() or char in _PUNCTUATION:
            out.append([_PUNCT_MAP.get(char, char), None])
            last_is_letter = False
        else:
            raise ValueError(f"unmapped Buckwalter character {char!r} at offset {index}")
    if stats is not None:
        stats["implicit_alif_marks"] += implicit
    pieces = []
    for base, marks in out:
        if marks is None:
            pieces.append(base)
        else:
            # Standard Arabic writing puts the vowel before the shadda, even
            # when Buckwalter writes ~ before the vowel. Sorting is stable, so
            # the relative order of the other marks is preserved.
            pieces.append(base + "".join(sorted(marks, key=lambda mark: mark == "\u0651")))
    return "".join(pieces)


def text_hash(text):
    """Same identity as tiny_voca.annotate: hash of the stripped text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


_QUOTED_LINE = re.compile(r'^"([^"]+)"\s+"(.*)"\s*$')


def _parse_transcript_lines(lines, source):
    result = {}
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        match = _QUOTED_LINE.match(line)
        if not match:
            raise ValueError(f"{source}:{number}: malformed transcript line")
        name, value = match.groups()
        if name in result:
            raise ValueError(f"{source}:{number}: duplicate recording {name!r}")
        result[name] = value
    return result


def parse_transcript(path):
    """Read a ``"wav name" "transcript"`` file into an exact-name dictionary."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        return _parse_transcript_lines(stream, path)


def parse_transcript_from_zip(archive, name):
    """Read the same transcript format from a member of an open ZipFile."""
    with archive.open(name) as stream:
        text = stream.read().decode("utf-8-sig")
    return _parse_transcript_lines(text.splitlines(), name)


def manifest_row(identifier, text, audio, split, recording):
    """One row accepted by ``tiny_voca.prepare``.

    ``audio`` is a path relative to the manifest file. Provenance keys are in
    prepare.py's whitelist so they survive into the feature manifest.
    """
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("id must be a nonempty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split {split!r}")
    return {
        "id": identifier,
        "text": text,
        "voice_id": VOICE_ID,
        "audio": str(audio).replace("\\", "/"),
        "split": split,
        "text_id": recording,
        "origin": "halabi2016",
        "source_id": recording,
        "source_license": SOURCE_LICENSE,
        "source_url": SOURCE_PAGE,
        "source_citation": SOURCE_CITATION,
    }


def assign_splits(rows, validation_fraction=0.05, seed=0):
    """Deduplicate by text and split without any cross-split text leakage.

    The corpus's official split is preserved for test. Repeated readings of
    the same sentence stay together: if any copy is in the official test set,
    the whole group is test; otherwise it is train/validation. ``annotate``
    rejects any text hash shared between splits, so this is a hard constraint.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = defaultdict(list)
    for row in rows:
        groups[text_hash(row["text"])].append(row)
    keys = sorted(groups)
    official = {key: ("test" if any(r["split"] == "test" for r in groups[key]) else "train") for key in keys}
    candidates = [key for key in keys if official[key] == "train"]
    order = sorted(candidates, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    validation = set(order[:max(1, round(len(order) * validation_fraction))])
    result = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        key = text_hash(row["text"])
        split = official[key]
        if split == "train" and key in validation:
            split = "validation"
        result[split].append({**row, "split": split})
    return result


# Corpus phonetic alphabet -> tiny-voca inventory. A trailing apostrophe in
# the transcript marks stress and is stripped; 'sil' is a pause. This table is
# a diagnostic only: those labels never enter the TTS model.
PHONETIC_TO_PHONE = {
    "b": "b", "t": "t", "tt": "t", "v": "s", "j": "dʒ", "jj": "dʒ",
    "H": "ħ", "HH": "ħ", "x": "χ", "d": "d", "dd": "d", "*": "ð",
    "r": "r", "rr": "r", "z": "z", "s": "s", "ss": "s", "$": "ʃ",
    "S": "s̪", "SS": "s̪", "D": "dˤ", "T": "t̪", "Z": "ð", "E": "ʕ",
    "G": "ɣ", "g": "ɣ", "f": "f", "q": "q", "k": "k", "l": "l",
    "ll": "l", "m": "m", "mm": "m", "n": "n", "nn": "n", "h": "h",
    "w": "w", "y": "j", "<": "ʔ", ">": "ʔ", "^": "ʔ", "}": "ʔ",
    "a": "a", "A": "aː", "aa": "aː", "AA": "aː", "AH": "ʔ", "Ah": "ʔ",
    "i": "i", "i0": "i", "I0": "i", "i1": "i", "ii": "iː", "ii0": "iː",
    "ii1": "i", "I": "iː", "I1": "iː", "II0": "iː",
    "u": "u", "u0": "u", "u1": "u", "uu": "uː", "uu0": "uː", "uu1": "u",
    "U0": "u", "U1": "uː", "UU0": "uː", "UU1": "uː", "o": "oː", "oo": "oː",
    "e": "eː", "ee": "eː",
    # Corpus-specific aliases verified against the recordings (see qc.md).
    "$$": "ʃ", "**": "ð", "<<": "ʔ", "^^": "ʔ", "SH": "ʃ", "TH": "ð",
    "J": "dʒ", "xx": "χ", "hh": "h", "p": "b", "pp": "b",
    # Geminates collapse to the single inventory phone (diagnostic only).
    "bb": "b", "DD": "dˤ", "EE": "ʕ", "ff": "f", "gg": "ɣ", "kk": "k",
    "qq": "q", "TT": "t̪", "ww": "w", "yy": "j", "zz": "z", "ZZ": "ð",
    # Pause / disfluency markers: excluded from the phone sequence.
    "sil": None, "-": None, "dist": None,
}


def corpus_phones(transcript):
    """Corpus phonetic transcript -> (inventory phones, unmapped labels)."""
    phones = []
    unmapped = Counter()
    for token in transcript.split():
        label = token.rstrip("'")
        if label not in PHONETIC_TO_PHONE:
            unmapped[label] += 1
            continue
        phone = PHONETIC_TO_PHONE[label]
        if phone is not None:
            phones.append(phone)
    return phones, unmapped


def _levenshtein(left, right):
    """Edit distance between two phone sequences (O(n*m) time, O(m) memory)."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[column - 1] + 1, previous[column] + 1,
                               previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def sequence_agreement(corpus, g2p):
    """Agreement between our G2P phones and the corpus phonetic transcript.

    ``agreement`` is the raw positional score: it is very sensitive to a single
    inserted/deleted phone (e.g. our G2P emits a leading ʔ for hamzat al-wasl,
    the Damascene corpus does not). ``similarity`` uses edit distance and is
    therefore the meaningful diagnostic of a conversion error. Both are
    reported; neither is an acceptance criterion for training.
    """
    if not corpus or not g2p:
        return None
    common = min(len(corpus), len(g2p))
    matches = sum(a == b for a, b in zip(corpus[:common], g2p[:common]))
    distance = _levenshtein(corpus, g2p)
    return {"corpus_phones": len(corpus), "g2p_phones": len(g2p),
            "positional_matches": matches,
            "agreement": round(matches / common, 4),
            "similarity": round(1 - distance / max(len(corpus), len(g2p)), 4),
            "length_ratio": round(common / max(len(corpus), len(g2p)), 4)}


def g2p_report(rows):
    """Dependency-free sanity checks on the converted text.

    The phonetic cross-check against the corpus transcripts is done by
    ``scripts/build_halabi_manifest.py`` because it needs the archive.
    """
    from . import arabic_text

    unknown = Counter()
    phonemized = 0
    for row in rows:
        result = arabic_text.phonemize(row["text"], diacritized=True)
        phonemized += 1
        unknown.update(result.unknown)
    return {"phonemized": phonemized, "unknown_graphemes": dict(unknown)}


def write_manifest(path, rows):
    """Atomically write one split's JSONL manifest."""
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)

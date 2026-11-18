"""Arabic (MSA) front end: raw text -> normalised -> diacritised -> phonemes.

This module is independent of the English ``tiny_voca.text`` front end. It
uses the SAME token convention (BOS/EOS/word space/punctuation, a stress
feature per token) so the rest of tiny-voca (training, inference, ONNX export)
keeps working unchanged once ``TINY_VOCA_FRONTEND=arabic`` is selected.

The phoneme inventory is a SUBSET of the 392 labels of the annotation aligner
``facebook/wav2vec2-lv-60-espeak-cv-ft``. That was checked against its
``vocab.json`` (see docs/arabic-frontend.md). Using only those labels means
``tiny_voca.align.PhoneRecognizer`` accepts every inventory phone without a
fallback, which is required for the ``annotate`` step.

The default phonemizer is a rule-based G2P that needs no external program and
is Apache-compatible. eSpeak NG 1.52 ``-v ar`` is available as a comparison
and annotation backend, but eSpeak NG is GPL-3.0: do not ship it in a
distributed browser bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Inventory. Keep the order append-only: it defines the model token ids.
# ---------------------------------------------------------------------------
PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
# Arabic punctuation. Keep ASCII marks too: mixed text and numbers occur.
PUNCTUATION = ("،", "؛", "؟", "!", ",", ".", ":", ";", "?", "(", ")", '"', "—", "…")

# Consonants of MSA. These are the exact labels eSpeak NG 1.52 emits for the
# ar voice, and every one is present in the aligner vocabulary.
CONSONANTS = (
    "ʔ",  # ء hamza (also hamzat al-wasl)
    "b",  # ب
    "t",  # ت
    "t̪",  # ط (eSpeak: dental t)
    "d",  # د
    "dˤ",  # ض (eSpeak: pharyngealised d)
    "ð",  # ذ / ظ (eSpeak maps both to ð)
    "f",  # ف
    "dʒ",  # ج (eSpeak NG 1.52 -v ar: dʒ)
    "h",  # ه
    "ħ",  # ح
    "χ",  # خ (eSpeak NG 1.52 -v ar: χ)
    "ɣ",  # غ
    "j",  # ي
    "k",  # ق / ك (eSpeak uses q for ق, k for ك)
    "q",  # ق
    "l",  # ل
    "m",  # م
    "n",  # ن
    "r",  # ر
    "s",  # س
    "s̪",  # ص (eSpeak: dental s)
    "ʃ",  # ش
    "w",  # و
    "z",  # ز
    "ʕ",  # ع
)

# Short and long vowels. eSpeak uses ə for an unwritten short vowel; the
# explicit fatha/kasra/damma give a/i/u and the long ones give aː/iː/uː.
VOWELS = ("ə", "a", "i", "u", "aː", "iː", "uː", "eː", "oː")

PHONES = CONSONANTS + VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOWEL_SET = frozenset(VOWELS)
VOWEL_IDS = frozenset(SYMBOL_TO_ID[v] for v in VOWELS)
# Stressed vowels for the probe: voicing checks in align.py use VOWEL_SET.
VOICELESS = frozenset(("t", "t̪", "k", "q", "f", "s", "s̪", "ʃ", "h", "ħ", "χ", "ʔ"))

STRESS_MARKS = {"ˈ": 1, "ˌ": 2}
FRONTEND_VERSION = 1
FRONTENDS = ("rules", "espeak")
DEFAULT_FRONTEND = "rules"
FRONTEND_VARIABLE = "TINY_VOCA_ARABIC_FRONTEND"
_frontend = DEFAULT_FRONTEND

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# Unify the alef variants, final ya and ta marbuta. Tatweel is deleted.
_UNIFY = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و", "ئ": "ي",
    "ـ": "",  # tatweel
})
# Arabic diacritics that survive normalisation (the caller may diacritise later).
_DIACRITICS = "ًٌٍَُِّْٰ"
# Arabic-Indic and Extended Arabic-Indic digits -> Western digits, then words.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DIGITS_EXT = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "–": "—", "―": "—",
})
_ARABIC_PUNCT = str.maketrans({"，": "،", "؟": "؟", "؛": "؛"})

ONES = ("", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة")
TENS = ("", "عشرة", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون")
# A tiny, explicit abbreviation table. Add only entries we can pronounce.
ABBREVIATIONS = {
    "د.": "دكتور",
    "أ.": "أستاذ",
    "م.": "مهندس",
    "الخ": "إلى آخره",
    "إلخ": "إلى آخره",
    "نحو": "نحو",
}

# ---------------------------------------------------------------------------
# Grapheme -> phoneme rules
# ---------------------------------------------------------------------------
# Base letter values. Sun letters assimilate the l of the definite article;
# moon letters leave it. eSpeak realises both with a leading ʔ (hamzat al-wasl).
LETTERS = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aː", "ؤ": "u", "ئ": "i",
    "ب": "b", "ت": "t", "ث": "s", "ج": "dʒ", "ح": "ħ", "خ": "χ",
    # Labels aligned with eSpeak NG 1.52 -v ar (the aligner's label set):
    # ج -> dʒ (not ɡ), خ -> χ (not x), ظ -> ð, ص -> s̪, ط -> t̪, ض -> dˤ.
    "د": "d", "ذ": "ð", "ر": "r", "ز": "z", "س": "s", "ش": "ʃ",
    "ص": "s̪", "ض": "dˤ", "ط": "t̪", "ظ": "ð", "ع": "ʕ", "غ": "ɣ",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "j", "ة": "a", "ى": "aː",
    "ء": "ʔ", "ٔ": "ʔ", "ٕ": "ʔ",
}
SUN_LETTERS = frozenset("تثدذرزسشصضطظن")
MOON_LETTERS = frozenset("ابجحخعغفقكلمةويه")
# Hamzat al-qat' is a full consonant ʔ plus its vowel (eSpeak: إِلَى -> ʔ_i_l_aː).
HAMZA = {"أ": "a", "إ": "i", "آ": "aː", "ؤ": "u", "ئ": "i"}
# Mater letters lengthen the short vowel that precedes them. ى is a final aː.
MATER = {"ا": "aː", "ي": "iː", "و": "uː", "ى": "aː"}
SHORT_VOWELS = {"َ": "a", "ِ": "i", "ُ": "u"}
LONG_VOWELS = {"ا": "aː", "ي": "iː", "و": "uː"}
TANWIN = {"ً": "a", "ٌ": "u", "ٍ": "i"}
SUKOON = "ْ"
SHADDA = "ّ"

# Punctuation that ends a clause and is realised as a pause token.
_PAUSE_CHARS = frozenset("!.؟!؛:،;—…")
_WORD_RE = re.compile(r"[^\s،؛؟!.,:;?\"()—…]+")


class NothingToPronounce(ValueError):
    """The text contains no pronounceable Arabic word, number or letter."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # graphemes with no inventory mapping

    def __len__(self):
        return len(self.ids)


def frontend():
    return _frontend


def set_frontend(name):
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def _number_to_words(number):
    """Say an integer in Arabic words. Handles thousands up to 999999."""
    if number == 0:
        return "صفر"
    if number < 10:
        return ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        if ones == 0:
            return TENS[tens]
        return f"{ONES[ones]} و{TENS[tens]}"
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        head = "مائة" if hundreds == 1 else f"{ONES[hundreds]} مائة"
        return head if rest == 0 else f"{head} و{_number_to_words(rest)}"
    thousands, rest = divmod(number, 1000)
    head = "ألف" if thousands == 1 else f"{_number_to_words(thousands)} آلاف"
    return head if rest == 0 else f"{head} و{_number_to_words(rest)}"


def _expand_numbers(text):
    def replace(match):
        digits = match.group(0).translate(_DIGITS).translate(_DIGITS_EXT)
        if not digits.isdigit():
            return match.group(0)
        return _number_to_words(int(digits))
    return re.sub(r"[٠-٩۰-۹0-9]+", replace, text)


def normalize(text):
    """Raw text -> normalised MSA text.

    Unifies alef/ya/ta-marbuta, removes tatweel, normalises quotes and Arabic
    punctuation, and writes numbers out as Arabic words.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTES).translate(_DIGITS).translate(_DIGITS_EXT)
    text = text.translate(_UNIFY).translate(_ARABIC_PUNCT)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    for short, full in ABBREVIATIONS.items():
        text = text.replace(short, full)
    text = _expand_numbers(text)
    return re.sub(r"\s+", " ", text).strip()


def has_diacritics(text):
    """True if the text already carries tashkeel marks."""
    return any(c in _DIACRITICS for c in text)


def diacritize(text, backend="camel"):
    """Return fully diacritised MSA text.

    ``backend`` selects an external tool; the external call is injected by
    ``scripts/diacritize.py`` (this module stays dependency-free and testable).
    """
    raise NotImplementedError("use scripts/diacritize.py to bind CAMeL or Shakkala")


def _grapheme_to_phones(word):
    """Diacritised Arabic word -> list of (phone, stress) with assimilation.

    Rules, in order:

    * hamzat al-wasl: ال starts with ʔ; before a sun letter the l is dropped
      and the sun letter is geminated (a following shadda is not doubled twice);
    * hamzat al-qat' (أ إ آ ؤ ئ) is ʔ + vowel;
    * shadda geminates the preceding consonant;
    * short vowel marks -> a/i/u; tanwin -> vowel + n;
    * a mater (ا ي و ى) after its matching short vowel -> long vowel;
    * an alif that only carries tanwin (ـًا) is not pronounced twice;
    * sukoon and other diacritics -> skipped;
    * the last vowel of the word carries stress level 1 (word stress).
    """
    result = []  # list of [phone, stress]
    i = 0
    n = len(word)
    unknown = None
    assimilated_shadda_at = -1

    def emit(phone):
        result.append([phone, 0])

    def last_vowel_index():
        for index in range(len(result) - 1, -1, -1):
            if result[index][0] in VOWEL_SET:
                return index
        return None

    def is_article_at(position):
        """True when ا at position starts ال, not the mater of a long ā.

        The article can start the word or follow a prefixed particle (ب ك ل ف و).
        In طَالِب / ظَالِم the alif follows another consonant and is a mater.
        """
        cursor = position - 1
        ignorable = set(SHORT_VOWELS) | set(TANWIN) | {SUKOON, SHADDA, "ٰ"}
        while cursor >= 0 and word[cursor] in ignorable:
            cursor -= 1
        return cursor < 0 or word[cursor] in "بكلفو"

    while i < n:
        char = word[i]
        following = word[i + 1] if i + 1 < n else ""

        if char in SHORT_VOWELS:
            emit(SHORT_VOWELS[char])
            i += 1
            continue
        if char in TANWIN:
            emit(TANWIN[char])
            emit("n")
            # Unicode writes ـًا as tanwin followed by its carrier alif.
            i += 2 if following == "ا" else 1
            continue
        if char == SHADDA:
            if i != assimilated_shadda_at and result and result[-1][0] not in VOWEL_SET:
                result.append([result[-1][0], 0])
            i += 1
            continue
        if char in (SUKOON, "ٰ"):
            i += 1
            continue

        # Alif as a carrier for tanwin: ـًا is /an/, not /a.an/.
        if char == "ا" and following in TANWIN:
            emit(TANWIN[following])
            emit("n")
            i += 2
            continue

        # Definite article ال, including after a prefix (بِالْعالَمِ).
        if char == "ا" and following == "ل" and is_article_at(i):
            emit("ʔ")
            emit("a")
            after_l = word[i + 2] if i + 2 < n else ""
            if after_l in SUN_LETTERS and after_l in LETTERS:
                sun = LETTERS[after_l]
                emit(sun)
                emit(sun)
                assimilated_shadda_at = i + 2
                i += 3
            else:
                emit("l")
                i += 2
            continue

        if char in HAMZA:
            emit("ʔ")
            # When an explicit vowel mark follows, it supplies the vowel;
            # otherwise the hamza letter carries it (آ already has ā).
            if following not in SHORT_VOWELS and following not in TANWIN:
                emit(HAMZA[char])
            i += 1
            continue

        if char in MATER:
            index = last_vowel_index()
            if (index is not None and index == len(result) - 1
                    and MATER[char] == result[index][0] + "ː"):
                result[index][0] = MATER[char]
            else:
                emit(LETTERS.get(char, "a"))
            i += 1
            continue

        if char in LETTERS:
            emit(LETTERS[char])
            i += 1
            continue

        if unknown is None:
            unknown = char
        i += 1

    index = last_vowel_index()
    if index is not None:
        result[index][1] = 1
    return [tuple(item) for item in result], unknown


def word_to_phones(word):
    """Public, testable G2P for one diacritised word.

    Returns ``(phones, unknown)`` where phones is a list of ``(phone, stress)``.
    Handles sun/moon assimilation, shadda (gemination), long/short vowels,
    tanwin, emphatics, hamzat al-wasl and final pause internally.
    """
    return _grapheme_to_phones(word)


def sentence_to_phones(sentence):
    """Diacritised sentence -> list of word phone lists and unknown graphemes."""
    words = []
    unknown = []
    for token in _WORD_RE.findall(sentence):
        phones, bad = word_to_phones(token)
        words.append(phones)
        if bad:
            unknown.append(bad)
    return words, unknown


def tokenize(words, unknown=()):
    """Build the token sequence from word phone lists (pure, testable)."""
    symbols, stress = [BOS], [0]
    for word in words:
        if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
            symbols.append(SP)
            stress.append(0)
        for phone, level in word:
            if phone not in PHONE_SET:
                phone = UNK
            symbols.append(phone)
            stress.append(level)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols),
                      tuple(stress), tuple(unknown))


def phonemize(text, diacritized=False):
    """Raw or diacritised text -> Phonemized (ids + stress).

    ``diacritized=True`` means the caller already ran the tashkeel step.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    # Already-vocalised text must not go through normalize(): it folds أ إ آ
    # ؤ ئ into un-hamzated letters and would silently lose hamzat al-qat'.
    clean = text if diacritized or has_diacritics(text) else normalize(text)
    if _frontend == "espeak":
        from .arabic_espeak import phonemize_espeak
        return phonemize_espeak(clean, diacritized=diacritized)
    words, unknown = sentence_to_phones(clean)
    return tokenize(words, unknown)


def frontend_identity():
    return {"frontend_version": FRONTEND_VERSION, "frontend": _frontend,
            "language": "ar", "symbols": list(SYMBOLS)}


def frontend_fingerprint():
    return hashlib.sha256(
        json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

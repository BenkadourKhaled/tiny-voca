"""eSpeak NG 1.52 ``-v ar`` backend for Phase 1 comparison and annotation.

eSpeak NG is GPL-3.0: use it OFFLINE to build/evaluate data, never bundle it
in a distributed demo. The default runtime front end is the Apache-compatible
rule engine in ``arabic_text.py``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .arabic_text import (BOS, EOS, PHONE_SET, SP, STRESS_MARKS, Phonemized,
                          SYMBOL_TO_ID, UNK)

ESPEAK_VOICE = "ar"
SEPARATOR = "_"
ESPEAK_VERSION = "1.52.0"

# eSpeak emits allophones that are in the aligner's 392-label vocabulary but
# not in our 35-phone inventory. Map them explicitly: a consonant with a
# length mark becomes geminated (two tokens), a vowel with a centred dot keeps
# its length, θ (ث) is realised as /s/ like the rule front end.
_EXTRA = {"θ": "s"}


def canonical(phone):
    """eSpeak label -> ``(phones, unmapped)`` with phones in our inventory."""
    if phone in PHONE_SET:
        return (phone,), None
    cleaned = phone.rstrip("-")
    if cleaned in PHONE_SET:
        return (cleaned,), None
    if cleaned.endswith("ː") and cleaned[:-1] in PHONE_SET:
        return (cleaned[:-1], cleaned[:-1]), None
    stripped = cleaned.replace(".", "")
    if stripped in PHONE_SET:
        return (stripped,), None
    if cleaned in _EXTRA:
        return (_EXTRA[cleaned],), None
    return (UNK,), cleaned


def espeak_executable():
    executable = os.environ.get("ESPEAK_NG") or _which("espeak-ng")
    if executable is None:
        raise RuntimeError(
            "eSpeak NG not found. Set ESPEAK_NG to its path "
            "(e.g. a portable espeak-ng.exe); see docs/arabic-frontend.md"
        )
    # A portable eSpeak NG needs to be told where its data lives; without this
    # it can exit with a native access violation (0xC0000005) on Windows.
    if not os.environ.get("ESPEAK_DATA_PATH"):
        data = Path(executable).parent / "espeak-ng-data"
        if data.is_dir():
            os.environ["ESPEAK_DATA_PATH"] = str(data)
    return executable


def _which(name):
    from shutil import which
    return which(name)


def parse_espeak(output):
    """eSpeak IPA output -> list of word phone lists with stress."""
    words = []
    for word in output.split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            # eSpeak emits language tags like (en) for foreign words.
            if unit.startswith("(") or unit in ("(en)", "(ar)"):
                continue
            unit = "".join(c for c in unit if c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def espeak_words(text):
    """Phonemise one text piece. A temp file avoids Windows argv encoding issues."""
    executable = espeak_executable()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.txt"
        source.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        result = subprocess.run(
            [executable, "-q", "--ipa", "--sep=" + SEPARATOR,
             "-v", ESPEAK_VOICE, "-f", str(source)],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            check=True,
        )
    return parse_espeak(result.stdout)


def phonemize_espeak(text, diacritized=False):
    """Text -> Phonemized using eSpeak NG with explicit allophone mapping."""
    words = espeak_words(text)
    symbols, stress, unknown = [BOS], [0], []
    for word in words:
        if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
            symbols.append(SP)
            stress.append(0)
        for phone, level in word:
            mapped, unmapped = canonical(phone)
            if unmapped:
                unknown.append(unmapped)
            for symbol in mapped:
                symbols.append(symbol)
                stress.append(level)
    symbols.append(EOS)
    stress.append(0)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols),
                      tuple(stress), tuple(unknown))

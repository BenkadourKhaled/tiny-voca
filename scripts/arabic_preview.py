"""Write a human-readable text -> phonemes preview for Arabic MSA.

The default front end is the rule engine. eSpeak NG is only used when
``--espeak`` is given (offline comparison/annotation; GPL-3.0, never bundle it).

Usage:
  python scripts/arabic_preview.py \
      --input out/diac_camel.txt --output out/arabic_phoneme_preview.txt \
      --espeak "C:\\path\\to\\espeak-ng.exe"     # optional
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_voca import arabic_text as ar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 file, one diacritised sentence per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--espeak", default=None, help="path to espeak-ng.exe for the comparison column")
    args = parser.parse_args()

    if args.espeak:
        os.environ["ESPEAK_NG"] = args.espeak
        ar.set_frontend("espeak")
    else:
        ar.set_frontend("rules")

    sentences = [line.strip() for line in Path(args.input).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    lines = [
        "# Aperçu texte -> phonèmes (arabe MSA)",
        "",
        f"Front-end : {ar.frontend()} (version {ar.FRONTEND_VERSION})",
        f"Source : {args.input}",
        "",
    ]
    for number, sentence in enumerate(sentences, 1):
        lines.append(f"## {number}. {sentence}")
        lines.append("")
        try:
            result = ar.phonemize(sentence, diacritized=True)
            lines.append(f"- PHONÈMES : {' '.join(result.symbols[1:-1])}")
            lines.append(f"- INCONNUS : {', '.join(result.unknown) if result.unknown else 'aucun'}")
        except Exception as error:  # keep the preview usable even if one line fails
            lines.append(f"- ERREUR : {type(error).__name__}: {error}")
        lines.append("")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(sentences)} previews to {destination}")


if __name__ == "__main__":
    main()

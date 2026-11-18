"""Diacritise Arabic sentences with CAMeL Tools 1.6 or Shakkala 1.7.

Both tools were checked on 2026-11-17:
  * CAMeL Tools 1.6.0, MIT, PyPI, Python 3.11-3.14. v1.6 uses the CALIMA
    morphology database + MLE disambiguation (``camel_data -i defaults``).
  * Shakkala 1.7, MIT, PyPI, requires TensorFlow 2.9.3 (Python <= 3.10).

Usage:
  python scripts/diacritize.py --backend camel --input data/sentences.txt \
      --output out/diacritized.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def diacritize_camel(sentences):
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.word import simple_word_tokenize

    disambig = MLEDisambiguator(Analyzer(MorphologyDB.builtin_db("calima-msa-r13")))
    result = []
    for sentence in sentences:
        tokens = simple_word_tokenize(sentence)
        analysed = disambig.disambiguate(tokens)
        result.append(" ".join(
            item.analyses[0].analysis.get("diac", item.word) for item in analysed
        ))
    return result


def diacritize_shakkala(sentences):
    from shakkala import Shakkala

    sh = Shakkala(version=3)
    model, _ = sh.get_model()
    result = []
    for sentence in sentences:
        logits = model.predict(sh.prepare_input(sentence))[0]
        result.append(sh.get_final_text(sentence, sh.logits_to_text(logits)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("camel", "shakkala"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sentences = [line.rstrip("\n") for line in Path(args.input).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if args.backend == "camel":
        output = diacritize_camel(sentences)
    else:
        output = diacritize_shakkala(sentences)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} sentences to {destination}")


if __name__ == "__main__":
    main()

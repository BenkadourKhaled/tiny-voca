"""Compare the two tashkeel outputs and write a readable Markdown report.

Usage:
  python scripts/compare_diacritization.py \
      --camel out/diac_camel.txt --shakkala out/diac_shakkala.txt \
      --raw data/sentences_100.txt --output out/diacritization_report.md
"""

import argparse
from pathlib import Path

DIACRITICS = set("ًٌٍَُِّْٰ")


def load(path):
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def coverage(sentences):
    """Fraction of Arabic letters carrying at least one diacritic."""
    letters = marks = 0
    for sentence in sentences:
        for char in sentence:
            if "\u0600" <= char <= "\u06ff":
                if char in DIACRITICS:
                    marks += 1
                else:
                    letters += 1
    return marks / (letters + marks) if letters + marks else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camel", required=True)
    parser.add_argument("--shakkala", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw, camel, shakkala = load(args.raw), load(args.camel), load(args.shakkala)
    n = min(len(raw), len(camel), len(shakkala))
    disagreements = [i for i in range(n) if camel[i] != shakkala[i]]
    same = [i for i in range(n) if camel[i] == shakkala[i]]

    lines = [
        "# Comparaison des diacritiseurs (Phase 1)",
        "",
        f" Jeu de {n} phrases MSA écrites pour ce projet (source `data/sentences_100.txt`).",
        "",
        "| Outil | Version | Licence | Python | Couverture tashkeel |",
        "|---|---|---|---|---|",
        f"| CAMeL Tools (CALIMA + MLE) | 1.6.0 | MIT | 3.11-3.14 | {coverage(camel):.1%} |",
        f"| Shakkala | 1.7 | MIT (mention du projet) | <=3.10 (TF 2.9.3) | {coverage(shakkala):.1%} |",
        "",
        f"- Sorties identiques : {len(same)}/{n}",
        f"- Sorties différentes : {len(disagreements)}/{n}",
        "",
        "## Exemples de divergence",
        "",
        "| # | Brut | CAMeL | Shakkala |",
        "|---|---|---|---|",
    ]
    for i in disagreements[:15]:
        lines.append(f"| {i + 1} | {raw[i]} | {camel[i]} | {shakkala[i]} |")
    lines += [
        "",
        "## Recommandation",
        "",
        "CAMeL Tools est recommandé pour la Phase 2 : licence MIT standard, projet",
        "maintenu en 2026, Python moderne, et intégration directe dans un pipeline",
        "Python. Shakkala reste un bon deuxième avis, mais TensorFlow 2.9.3 impose",
        "Python 3.10 et un environnement séparé. Quelle que soit la sortie retenue,",
        "la relecture manuelle du CSV de la Phase 2 reste obligatoire : aucun",
        "diacritiseur automatique n'est fiable à 100 % sur du MSA.",
        "",
    ]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"n={n} identical={len(same)} different={len(disagreements)} -> {destination}")


if __name__ == "__main__":
    main()

# Front-end arabe (MSA) — Phase 1

## Ce que c'est

Un front-end `texte arabe → phonèmes` indépendant de l'anglais, dans
`src/tiny_voca/arabic_text.py`. Il produit **exactement le même type de jetons**
que le front-end anglais (BOS/EOS, espace de mot, ponctuation, accentuation en
feature séparée), donc l'alignement, l'entraînement, l'inférence et l'export
n'ont pas été modifiés.

- Langue cible : **arabe standard moderne (MSA)**.
- Inventaire : **35 phonèmes**, sous-ensemble vérifié des **392 labels** de
  `facebook/wav2vec2-lv-60-espeak-cv-ft` (instantané :
  `data/aligner_vocab.json`, 2026-11-17). C'est obligatoire : `align.py`
  refuse un phone absent du vocabulaire de l'aligneur.
- Symboles : 5 spéciaux + 14 ponctuations + 35 phones = **54 tokens**
  (`Config.vocab_size` suit automatiquement la langue active).
- Deux moteurs :
  - `rules` (défaut) : règles G2P maison, sans dépendance externe, licence
    Apache-compatible. C'est celui qu'on utilisera pour l'entraînement.
  - `espeak` : eSpeak NG 1.52 `-v ar`, **hors ligne uniquement**, pour
    comparaison/annotation. eSpeak NG est **GPL-3.0** : ne jamais l'embarquer
    dans la démo navigateur.
- La diacritisation (tashkeel) n'est **pas** faite au moment de `phonemize`.
  Elle est faite avant, par `scripts/diacritize.py`, et le texte diacritisé est
  passé avec `diacritized=True`.

## Démarrage rapide

```powershell
# Depuis C:\Users\User\tiny-voca-ar
$PY = "C:\Users\User\AppData\Local\Temp\opencode\venv312\Scripts\python.exe"

# 1. Installer le projet + extras arabes (une seule fois)
& $PY -m pip install -e ".[prepare,arabic-camel,arabic-espeak]"

# 2. Données CAMeL (une seule fois, ~40 Mo)
& "C:\Users\User\AppData\Local\Temp\opencode\venv312\Scripts\camel_data.exe" -i defaults

# 3. Diacritiser les 100 phrases de test
& $PY scripts\diacritize.py --backend camel `
    --input data\sentences_100.txt --output out\diac_camel.txt

# 4. Comparer CAMeL / Shakkala
& $PY scripts\compare_diacritization.py `
    --camel out\diac_camel.txt --shakkala out\diac_shakkala.txt `
    --raw data\sentences_100.txt --output out\diacritization_report.md

# 5. Aperçu lisible texte → phonèmes
& $PY scripts\arabic_preview.py `
    --input out\diac_camel.txt --output out\arabic_phoneme_preview.txt

# Variante eSpeak (comparaison ; binaire GPL hors ligne)
& $PY scripts\arabic_preview.py `
    --input out\diac_camel.txt --output out\arabic_phoneme_espeak.txt `
    --espeak "C:\Users\User\AppData\Local\Temp\opencode\espeak-portable\eSpeak NG\espeak-ng.exe"

# 6. Tests
& $PY -m unittest discover -s tests -p "test_arabic.py" -v
```

Shakkala nécessite Python ≤ 3.10 (TensorFlow 2.9.3), donc un venv séparé :

```powershell
$PY310 = "C:\Users\User\AppData\Local\Temp\opencode\venv310\Scripts\python.exe"
& $PY310 -m pip install -e ".[arabic-shakkala]"
$env:TF_CPP_MIN_LOG_LEVEL = "3"   # masque l'avertissement CUDA sans GPU
& $PY310 scripts\diacritize.py --backend shakkala `
    --input data\sentences_100.txt --output out\diac_shakkala.txt
```

## Utilisation depuis Python

```python
from tiny_voca import text

text.phonemize("مَرْحَبًا بِالْعالَمِ", diacritized=True).symbols
# ('<bos>', 'm', 'a', 'r', 'ħ', 'a', 'b', 'a', 'n', '<sp>',
#  'b', 'i', 'ʔ', 'a', 'l', 'ʕ', 'a', 'l', 'a', 'm', 'i', '<eos>')
```

Variables d'environnement :

| Variable | Valeurs | Rôle |
|---|---|---|
| `TINY_VOCA_LANGUAGE` | `en` (défaut), `ar` | bascule tout le pipeline (inventaire, config, G2P) |
| `TINY_VOCA_FRONTEND` | `rules`, `espeak` (ar) · `espeak`, `own` (en) | moteur de phonémisation |
| `ESPEAK_NG` | chemin vers `espeak-ng.exe` | backend eSpeak hors ligne |
| `ESPEAK_DATA_PATH` | dossier `espeak-ng-data` | déduit automatiquement du binaire si absent |

L'anglais reste le comportement par défaut : aucun test anglais n'a changé.

## Résultats de la Phase 1

| Outil | Version | Licence | Python | Couverture tashkeel sur 100 phrases |
|---|---|---|---|---|
| CAMeL Tools (CALIMA + MLE) | 1.6.0 | MIT | 3.11–3.14 | 32,3 % |
| Shakkala | 1.7 | MIT (mention du projet requise) | ≤ 3.10 / TF 2.9.3 | 44,3 % |

Sorties identiques : **0/100**. Rapport complet : `out/diacritization_report.md`.

**Recommandation actuelle : CAMeL Tools**, malgré sa couverture plus basse —
API moderne, licence MIT standard, projet maintenu, un seul environnement
Python. Shakkala sert de deuxième avis. Dans tous les cas, **relecture manuelle
obligatoire** avant toute utilisation pour l'entraînement : aucun diacritiseur
automatique n'est fiable à 100 %.

Aperçus générés : `out/arabic_phoneme_preview.txt` (règles) et
`out/arabic_phoneme_espeak.txt` (comparaison), 100 phrases chacun, **0 phone
inconnu** dans les deux cas.

## Limites connues

- Le G2P par règles est une approximation : pas de morphologie complète, pas
  de désambiguisation contextuelle. Il vise des phonèmes **stables et dans
  l'inventaire**, pas une transcription parfaite.
- eSpeak peut émettre des allophones hors inventaire (`a.ː`, `t̪ː`, `θ`…) :
  `arabic_espeak.canonical()` les ramène à l'inventaire (gémination conservée).
- Les chiffres arabes sont écrits en toutes lettres ; les nombres > 999 999 ne
  sont pas gérés (ils restent des chiffres et sont signalés comme inconnus).
- Aucun entraînement arabe n'a eu lieu : le modèle existant reste anglais.
  La Phase 2 (données) est bloquée sur le cadre légal/consentement de la voix.

## Licences

| Composant | Licence | Usage dans ce projet |
|---|---|---|
| tiny-voca (code/poids) | Apache-2.0 | base du projet |
| CAMeL Tools 1.6.0 | MIT | diacritisation hors ligne |
| Shakkala 1.7 | MIT + mention du projet | deuxième avis hors ligne |
| 100 phrases de test | CC0 (écrites pour le projet) | `data/sentences_100.txt` |
| eSpeak NG 1.52 | **GPL-3.0** | hors ligne uniquement, jamais embarqué |

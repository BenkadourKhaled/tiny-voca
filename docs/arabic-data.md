# Phase 2 — données vocales arabes (corpus Halabi)

## Source vérifiée

| Élément | Valeur |
|---|---|
| Corpus | Arabic Speech Corpus — Nawar Halabi, université de Southampton, 2016 |
| Page officielle | http://en.arabicspeechcorpus.com/ |
| Archive | http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip (1,19 Go) |
| Licence | **CC BY 4.0** — attribution obligatoire |
| Citation | Halabi, Nawar (2016). *Modern standard Arabic phonetics for speech synthesis*. Thèse de doctorat, University of Southampton. |
| Contenu | 1 813 énoncés d entraînement + 100 de test, **une seule voix de studio**, arabe standard lu avec un accent damascène |
| Audio | WAV 48 kHz dans l archive, converti en **24 kHz mono** pour tiny-voca |
| Transcripts | Fichiers .lab et transcripts en **Buckwalter** (ASCII), avec voyelles, sukun, shadda et tanwin |
| Annotation | TextGrid Praat avec frontières de phonèmes (contrôle qualité indépendant) |

Le code et les poids du projet restent Apache-2.0. La licence CC BY 4.0 impose de conserver la citation ci-dessus dans tout artefact dérivé (manifestes, checkpoints, documentation de la démo). Aucune voix générée par un service en ligne n est utilisée.

## Commandes exactes

Depuis la racine du dépôt, avec Python 3.12 :

```bash
export PYTHONIOENCODING=utf-8
export TINY_VOCA_LANGUAGE=ar
pip install -e .[prepare,annotate]
python scripts/build_halabi_manifest.py --output data/halabi
```

Sous PowerShell, remplacer les deux `export` par :

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:TINY_VOCA_LANGUAGE = "ar"
```

Options utiles :

```bash
python scripts/build_halabi_manifest.py --archive /path/arabic-speech-corpus.zip --output data/halabi
python scripts/build_halabi_manifest.py --archive /path/arabic-speech-corpus.zip --output data/halabi-smoke --limit 5
```

`--limit` sert uniquement au test de fumée : jamais pour entraîner. Le script est idempotent (les WAV déjà rééchantillonnés ne sont pas réécrits). Le téléchargement est reprenable via HTTP Range : une coupure réseau reprend au dernier octet écrit, sans re-télécharger 1,19 Go.

## Ce que fait le script

1. Lit les transcripts orthographiques officiels (train et test) plutôt que les fichiers .lab, qui contiennent une variante de translittération sur certains mots. Le fichier phonétique du train s appelle `phonetic-transcipt.txt` (faute de frappe dans l archive) : le script utilise ce nom exact.
2. Convertit chaque chaîne Buckwalter en arabe avec harakat via `tiny_voca.arabic_data.buckwalter_to_arabic`. Les combinaisons sont explicites : `~` shadda, `o` sukun, `a/u/i` voyelles, `F/N/K` tanwin, `*` dhal, `^` hamza sur ya (variante du corpus). Les voyelles sont placées avant la shadda, conformément à l écriture arabe.
3. Restaure **explicitement et de façon comptée** l alif omis devant une voyelle en début de mot (`istiEdAdan`, `al~atiy`, `an~a`, etc.) : 6 cas dans le corpus. Tout caractère réellement inconnu reste une erreur bloquante.
4. Convertit le séparateur de clauses ASCII `-` en em dash `—`, déjà présent dans l inventaire validé de 54 symboles, afin qu aucun grapheme ne soit inconnu du G2P.
5. Rééchantillonne chaque WAV de 48 kHz vers 24 kHz mono et l écrit dans `data/halabi/wav24/`.
6. Écrit `data/halabi/manifests/{train,validation,test}.jsonl` avec les champs attendus par `tiny_voca.prepare` : `id`, `text`, `voice_id`, `audio`, `split`, plus la provenance (licence, URL, citation).
7. Écrit `data/halabi/build.json` (comptages, empreintes, rapport G2P) et `data/halabi/qc.md` (contrôle qualité phonétique).

## Résultat réel (build du 17 novembre 2026)

| Split | Énoncés | Graphemes inconnus |
|---|---:|---|
| train | 1 722 | 0 |
| validation | 91 | 0 |
| test | 100 | 0 |

Total : 1 913 enregistrements, 24 kHz, une seule voix (`F2`). Les empreintes SHA-256 des trois manifestes sont dans `build.json`, ce qui rend le découpage reproductible. Vérification effectuée : aucun `text_hash` partagé entre splits, aucun WAV manquant.

Le QC compare notre G2P au transcript phonétique officiel sur les 1 913 énoncés : similarité moyenne **0,836** (médiane 0,840), rapport de longueur moyen 0,949, 6 alifs implicites restaurés, aucun label non mappé. L accord positionnel brut (0,472) est plus bas uniquement parce que notre G2P émet un `ʔ` initial pour le hamzat al-wasl, ce qui décale toute la séquence : la similarité par distance d édition est l indicateur pertinent. La table de mapping phonétique ne fait jamais entrer ces labels dans le modèle.

### Features et annotation (phases 3 et 4)

Le pipeline complet est décrit dans [arabic-pipeline.md](arabic-pipeline.md). Résultats réels
obtenus sur CPU avec la config mono-locuteur `F2` / 54 symboles :

| Split | Features acceptées | Rejets `prepare` | Annotation acceptée | Rejets QC alignement | Heures |
|---|---:|---:|---:|---:|---:|
| train | 1 721 | 1 (32,06 s > 20 s) | 1 214 | 507 | 2,85 |
| validation | 90 | 1 (32,06 s > 20 s) | 61 | 29 | 0,15 |
| test | 100 | 0 | 99 | 1 | 0,29 |

Les rejets `prepare` sont des enregistrements dépassant la limite d'admission de 20 s. Les rejets
d'annotation viennent surtout de `low_confidence` et `weak_alignment` : l'aligneur
`facebook/wav2vec2-lv-60-espeak-cv-ft` a été entraîné sur des étiquettes eSpeak, pas sur
l'inventaire arabe, donc une part des alignements ne passe pas les portes de qualité. Le détail
par raison est dans `data/halabi/annotations/summary.json` et les `*.rejected.jsonl`.

Vérification effectuée après fusion des 11 tranches d'annotation : `complete: true`, empreinte
reconstruite identique, tous les fichiers `.pt` présents, aucun id ni texte partagé entre splits,
et les trois `AnnotatedDataset` se chargent sans erreur.

## Découpage

- Le split officiel `test` (100 énoncés) est conservé tel quel.
- La validation est prélevée dans le train avec `validation_fraction=0.05` et un `seed` fixé (0 par défaut).
- Le dédoublage se fait sur le hash du texte : plusieurs lectures d une même phrase restent dans le même split. `tiny_voca.annotate` rejette toute fuite de texte entre splits, donc cette contrainte est vérifiée en amont.
- La voix unique du corpus est mappée sur le slot `F2` de tiny-voca. Un futur entraînement devra utiliser une config à **un seul locuteur**.

## Limites connues

- Accent damascène : ce n est pas une MSA neutre. La prononciation cible du front-end rules est MSA ; les différences sont documentées, pas corrigées.
- Les TextGrid et transcripts phonétiques du corpus utilisent un alphabet propre (ex. `AA`, `II0`, `SS`, `$$`) : une table de mapping diagnostique les convertit vers l inventaire, mais ils ne servent jamais d entrée au modèle. Les géminées sont réduites à une seule phone pour ce diagnostic.
- Le corpus ne fournit pas de consentement écrit séparé : la publication CC BY 4.0 par le titulaire des droits est la base légale utilisée, avec citation obligatoire.
- Les données brutes et rééchantillonnées restent dans `data/`, ignoré par git.

## Tests

```bash
export TINY_VOCA_LANGUAGE=ar
python -m unittest discover -s tests -p "test_arabic_data.py" -v
python -m unittest discover -s tests -p "test_lock.py" -v
```

Le correctif de verrou dans `tiny_voca.journal.exclusive_lock` remplace l import `fcntl` de `prepare.py` : le pipeline fonctionne désormais sur Windows comme sur Linux, avec le même comportement de verrou non bloquant.

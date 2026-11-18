# Phases 3 à 5 — pipeline arabe Halabi sur CPU

Ce document décrit comment produire un modèle **arabe** à partir du corpus
Halabi (CC BY 4.0). Le checkpoint anglais publié ne peut pas être réutilisé :
son inventaire compte 76 symboles et 2 voix, contre 54 symboles et 1 voix ici.
L'entraînement démarre donc **de zéro**.

## Environnement

Toujours définir, sous PowerShell :

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:TINY_VOCA_LANGUAGE = "ar"
$env:TINY_VOCA_FRONTEND = "rules"
```

`rules` est le front-end arabe sans dépendance externe. Ne pas mettre `espeak`
dans le navigateur (binaire GPL-3.0). Les textes doivent être **vocalisés** :
il n'existe pas de diacritisation automatique fiable dans le navigateur.

Installation :

```powershell
python -m pip install -e ".[prepare,annotate,arabic-data]" onnx onnxruntime
```

Le cache Hugging Face doit contenir `facebook/wav2vec2-lv-60-espeak-cv-ft`
(révision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`) : l'aligneur est chargé
en local uniquement (`local_files_only=True`).

## Configuration mono-locuteur

`data/halabi/arabic-single-speaker.json` :

```json
{"speakers": ["F2"], "vocab_size": 54}
```

## Phase 3 — features acoustiques

Commande de référence, un split à la fois :

```powershell
python -m tiny_voca.prepare --inputs data\halabi\manifests\train.jsonl `
  --output data\halabi\features --split train
python -m tiny_voca.prepare --inputs data\halabi\manifests\validation.jsonl `
  --output data\halabi\features --split validation
python -m tiny_voca.prepare --inputs data\halabi\manifests\test.jsonl `
  --output data\halabi\features --split test
```

`prepare` est le coût dominant sur CPU (`librosa.pyin`, plusieurs secondes par
énoncé). Sur une machine sans GPU, `scripts/parallel_prepare_train.py` répartit
les énoncés restants en tranches, puis fusionne les journaux en restaurant la
`record_key` d'origine (index du manifeste complet) :

```powershell
python scripts\parallel_prepare_train.py split --shards 8
# 8 × python -m tiny_voca.prepare --inputs data\halabi\features-shards\train-N.jsonl `
#     --output data\halabi\features-shards\out-N --split train
python scripts\parallel_prepare_train.py merge --shards 8
```

Deux enregistrements dépassent la limite de 20 s (3 000 frames) et sont rejetés
par le contrôle d'admission ; tous les autres sont acceptés. Le détail est dans
`data/halabi/features/*.rejected.jsonl`.

## Phase 4 — annotation / alignement forcé

Commande de référence (une seule course, toutes les voix de `SPLITS`) :

```powershell
python -m tiny_voca.annotate --features data\halabi\features `
  --output data\halabi\annotations `
  --dataset-revision halabi-2016-v1 `
  --config data\halabi\arabic-single-speaker.json `
  --device cpu --splits train validation test --stage ctc
```

Le réglage `--dataset-revision` est libre mais obligatoire ; la valeur
`halabi-2016-v1` identifie le build Phase 2. L'aligneur `wav2vec2` tourne sur
CPU en float32. Les portes de qualité rejettent les alignements faibles ; les
raisons exactes sont dans `summary.json` et `*.rejected.jsonl`.

Sur CPU, l'alignement peut aussi être réparti en tranches indépendantes, puis
fusionné. `scripts/merge_annotations.py` reconstruit l'identité qu'aurait écrite
une course unique (sources = empreintes des manifestes complets) et réécrit le
fingerprint de chaque ligne. Il vérifie aussi que `acceptés + rejetés` couvre
exactement chaque manifeste de features, et refuse les tranches faites avec
`--limit`.

Deux points importants :

1. Les énoncés déjà préparés **avant** le parallélisme doivent avoir leur propre
   tranche (`out-early` : liens durs vers leurs `.pt` dans le répertoire complet).
   Sinon 391 énoncés manquent silencieusement à l'entraînement.
2. PyTorch ouvre trop de threads par processus sur CPU. Limiter chaque
   annotateur à 2 threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
   `OPENBLAS_NUM_THREADS`) évite la sur-souscription quand 11 processus tournent
   en parallèle. Une interruption est reprise avec `--resume` : le journal
   `raw/*.jsonl` conserve chaque énoncé aligné.

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"

# 8 courses avec --splits train sur out-0 … out-7, une sur out-early,
# plus une course par split validation / test sur les features complètes.
# Toutes avec --dataset-revision halabi-2016-v1 --stage ctc --device cpu.
# En cas d'interruption : ajouter --resume (mêmes options, même config).

python scripts\merge_annotations.py `
  --shards data\halabi\annotations-shards\train-0 `
           data\halabi\annotations-shards\train-1 `
           data\halabi\annotations-shards\train-2 `
           data\halabi\annotations-shards\train-3 `
           data\halabi\annotations-shards\train-4 `
           data\halabi\annotations-shards\train-5 `
           data\halabi\annotations-shards\train-6 `
           data\halabi\annotations-shards\train-7 `
           data\halabi\annotations-shards\train-early `
           data\halabi\annotations-shards\validation `
           data\halabi\annotations-shards\test `
  --features data\halabi\features --output data\halabi\annotations
```

`summary.json` donne les comptages, les heures, les rejets et leurs raisons.
Les sondages de timing et quantiles restent dans les `summary.json` des tranches.

`annotations.json` doit porter `"complete": true` avant l'entraînement.

## Phase 5 — entraînement depuis zéro

```powershell
python -m tiny_voca.train `
  --train data\halabi\annotations\train.jsonl `
  --validation data\halabi\annotations\validation.jsonl `
  --output runs\ar-halabi-f2 `
  --config data\halabi\arabic-single-speaker.json `
  --device cpu `
  --steps 3000 --gan-start 100000 `
  --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 `
  --batch-frames 4000 --max-batch 8 `
  --save-every 500 --log-every 25 `
  --max-seconds 1800 --seed 20261113
```

Le GAN reste désactivé (`--gan-start` au-dessus du nombre d'étapes) et le
conditionnement par prosodie enregistrée est nul : ces deux composants coûtent
trop cher sur CPU. `last.pt` contient l'état complet ; `inference.pt` est le
checkpoint léger. Une interruption est reprise avec `--resume runs/.../last.pt`
en conservant exactement la même recette. Les threads sont limités à 8
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) pour éviter la
sur-souscription. Le premier lancement a été interrompu au step 334 par
`--max-seconds 1800` ; la reprise a utilisé `--max-seconds 99999`.

Résultat réel (121 notes de progression, step 25 → 3000) :

| Étape | Loss | Mel validation (prédit) |
|---:|---:|---:|
| 25 | 48,42 | — |
| 334 | 15,06 | 0,855 |
| 1 000 | ~9,5 | 0,626 |
| 2 000 | 9,25 | 0,588 |
| 3 000 | 8,89 | 0,572 |

À 3 000 steps : erreur de hauteur 0,98 demi-ton, précision de voisement 0,978,
précision des pauses 0,936. Le modèle reste limité par la taille du corpus
(2,85 h) et l'absence de GAN : la prosodie générée est plus plate que la
référence. Les courbes complètes sont dans `runs/ar-halabi-f2/metrics.jsonl`.

Export ONNX et vérification :

```powershell
python scripts\export_checkpoint.py --checkpoint runs\ar-halabi-f2\last.pt `
  --output runs\ar-halabi-f2\inference.pt
python -m tiny_voca.export --checkpoint runs\ar-halabi-f2\inference.pt `
  --output runs\ar-halabi-f2\tiny-voca-ar.onnx --opset 17
python -m tiny_voca.export --checkpoint runs\ar-halabi-f2\inference.pt `
  --output runs\ar-halabi-f2\tiny-voca-ar-exact.onnx --external-noise --opset 17
python scripts\check_onnx.py --checkpoint runs\ar-halabi-f2\inference.pt `
  --onnx runs\ar-halabi-f2\tiny-voca-ar-exact.onnx `
  --release runs\ar-halabi-f2\tiny-voca-ar.onnx --output onnx-check --threads 4
```

Les phrases de `check_onnx.py` sont vocalisées quand
`TINY_VOCA_LANGUAGE=ar`. Le fichier `.json` voisin de l'ONNX contient
`frontend: "rules"`, `language: "ar"`, `voices: ["F2"]` et `max_tokens` au
niveau racine (le worker accepte aussi l'ancien `config.model.max_tokens`).

Synthèse depuis un checkpoint arabe :

```powershell
python -m tiny_voca.infer --checkpoint runs\ar-halabi-f2\inference.pt `
  --voice F2 --text "مَرْحَبًا بِالْعالَمِ." --output out\arabic.wav --device cpu
```

## Évaluation libre (sans ASR)

`evaluate.py` respecte `TINY_VOCA_LANGUAGE` : en arabe, les phrases « novel »
sont vocalisées, vérifiées absentes des trois splits, et le normaliseur de texte
plie les variantes orthographiques. Avec `--no-asr`, aucune reconnaissance
vocale n'est téléchargée ; le rapport contient les métriques acoustiques
(`mel_dtw`), de timing (`duration_ratio`, `pause_ratio`), de hauteur et les
paires WAV pour écoute :

```powershell
python -m tiny_voca.evaluate --checkpoint runs\ar-halabi-f2\inference.pt `
  --annotations data\halabi\annotations --output evaluation\ar-f2 --tag ar-f2-step3000 `
  --splits test validation --novel 8 --per-voice 4 `
  --device cpu --no-asr --seed 0
```

Sans `--no-asr`, le modèle Whisper par défaut est `small.en` en anglais et
`small` en arabe (option `--asr-model` pour le remplacer). Le WER reste une
aide à l'écoute, pas une preuve de qualité.

## Attribution

Corpus : Halabi, Nawar (2016). *Modern standard Arabic phonetics for speech
synthesis*. PhD thesis, University of Southampton. Licence CC BY 4.0. La
citation est conservée dans les manifestes, les checkpoints et la démo.

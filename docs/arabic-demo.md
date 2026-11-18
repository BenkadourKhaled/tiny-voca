# Démo navigateur arabe — Phase 6

## Ce qui a changé

La démo `web/public/` ne lit plus l'anglais via dizione : elle utilise le vrai front-end arabe
`src/tiny_voca/arabic_text.py`, copié **à l'identique** dans `web/public/assets/arabic_text.py`.
Pyodide charge ce fichier, puis `phonemize_json(text)` appelle
`arabic_text.phonemize(text, diacritized=True)` et renvoie exactement
`{"ids": [...], "stress": [...], "text": "<bos> … <eos>"}`. Aucun dictionnaire, aucun numpy et
aucun moteur externe (eSpeak NG est GPL-3.0 et n'est jamais embarqué).

- `worker.js` : plus de dizione ; ponctuation arabe (`،` `؛` `؟` `.` `!`) pour la découpe et la fin
  des phrases ; le chemin anglais de `sentences()`/`finished()` est conservé tel quel (le test
  `tests/test_infer.py` le compare à Python) ; `max_tokens` est lu en plat **ou** sous `config.model`,
  avec repli sur 512.
- `index.html` : interface RTL, une seule voix **F2** (`value="0"`), exemples déjà vocalisés,
  avertissement sur le tashkeel. Le contrat avec `player.js` est inchangé.
- `fetch_assets.sh` : accepte un dossier local **ou** une URL de base ; copie le G2P, le ONNX et son
  config dans `public/assets/`. Aucun dépôt Hugging Face n'est codé en dur.
- `tests/test_web.py` : le harnais Node exécute le `phonemize_json` du worker contre le module réel,
  plus des tests ciblés (découpe arabe, voix unique, repli `max_tokens`).

## Commandes exactes

Depuis `C:\Users\User\tiny-voca-ar`, avec Python 3.12. Les assets de la démo ont été copiés depuis le
modèle arabe final (`runs\ar-halabi-f2`) : `model.onnx` (21,4 Mo), `config.json` et le G2P.

```powershell
$PY = "C:\Users\User\AppData\Local\Temp\opencode\venv312\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:TINY_VOCA_LANGUAGE = "ar"
$env:TINY_VOCA_FRONTEND = "rules"

# 1. Recopier les assets (dossier local ou URL de base)
& "C:\Program Files\Git\bin\bash.exe" web/fetch_assets.sh `
  "C:/Users/User/tiny-voca-ar/runs/ar-halabi-f2" tiny-voca-ar.onnx tiny-voca-ar.json

# 2. Vérifier l'intégrité : le G2P web doit rester identique à la source
#    SHA-256 1408EE5C46A257DA355BB40687B400E0BC2C541CFBD7E64F0E4711553200887D
& $PY -m unittest discover -s tests -p "test_web.py" -v

# 3. Servir la page (Pyodide et ONNX Runtime viennent des CDN)
cd web\public
& $PY -m http.server 8080
# ouvrir http://localhost:8080/
```

`config.json` contient `frontend: "rules"`, `language: "ar"`, `voices: ["F2"]` et la citation CC BY 4.0.
Le premier chargement télécharge Pyodide et ONNX Runtime ; ces fichiers sont ensuite mis en cache par
le navigateur. Le ONNX lui-même est servi localement, sans dépôt distant.

## Limites

- **Tashkeel obligatoire** : le texte doit être saisi déjà entièrement vocalisé. Il n'y a **aucun**
  diacritiseur automatique dans le navigateur. Les exemples de la page sont prêts à l'emploi.
- **Une seule voix** : `F2` (identifiant 0). Les anciennes pages à deux voix F2/M2 sont anglaises.
- **Accent damascène** : le modèle a appris le corpus Halabi, lu en arabe standard avec un accent de
  Damas. Ce n'est pas une MSA « neutre » ; les écarts sont documentés, pas corrigés.
- **Attribution** : les données d'entraînement sont **CC BY 4.0**. Toute redistribution doit conserver
  la citation : Halabi, Nawar (2016). *Modern standard Arabic phonetics for speech synthesis*,
  University of Southampton. Le code et les poids restent Apache-2.0.
- Le G2P par règles vise des phonèmes stables dans l'inventaire, pas une transcription parfaite ;
  les noms étrangers et les chiffres complexes peuvent être mal prononcés.

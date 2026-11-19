# Arabic voice-answer service

A small HTTP service that turns a **text question** into an **Arabic voice
message**:

```text
client question (text)
    -> LLM answer (short, Modern Standard Arabic, no diacritics)
    -> CAMeL Tools tashkeel
    -> Arabic G2P + ONNX model (F2, 24 kHz)
    -> WAV reply + the answer text in a response header
```

The service reuses the exact ONNX model and the exact G2P file of the browser
demo (`web/public/assets/`). Nothing is downloaded at inference time. eSpeak NG
(GPL-3.0) is **not** used and **not** shipped.

## 1. Install

From the repository root, with Python 3.11-3.14:

```powershell
python -m pip install -e ".[voice-service]"
camel_data -i defaults          # CALIMA MSA database, once, at image build time
```

`camel_data -i defaults` writes the morphology database to the user data
directory. Run it once when building the image, not at every start.

Copy the three model assets if they are not already present:

```powershell
New-Item -ItemType Directory -Path server\assets -Force | Out-Null
Copy-Item runs\ar-halabi-f2\tiny-voca-ar.onnx server\assets\model.onnx -Force
Copy-Item runs\ar-halabi-f2\tiny-voca-ar.json server\assets\config.json -Force
Copy-Item src\tiny_voca\arabic_text.py server\assets\arabic_text.py -Force
```

The `.gitignore` keeps these heavy files out of git; the release
`v0.1-ar-demo-kit` contains the same model.

## 2. Configure the LLM

Any OpenAI-compatible `/v1/chat/completions` endpoint works: OpenAI, Gemini's
OpenAI compatibility layer, Ollama, vLLM, llama.cpp...

| Variable | Default | Meaning |
|---|---|---|
| `TINY_VOCA_LLM_URL` | `https://api.openai.com/v1/chat/completions` | Chat completions endpoint |
| `TINY_VOCA_LLM_KEY` | *(required)* | Bearer token. For a local server with no auth, set any non-empty value |
| `TINY_VOCA_LLM_MODEL` | `gpt-4o-mini` | Model name sent in the request |
| `TINY_VOCA_LLM_TIMEOUT` | `20` | HTTP timeout in seconds |
| `TINY_VOCA_ASSETS` | `server/assets` | Directory with `model.onnx`, `config.json`, `arabic_text.py` |

The system prompt asks for one or two short sentences in MSA, **without**
diacritics: tashkeel is added by the service. The API key stays on the server;
it is never returned to the client.

Example with a local Ollama server:

```powershell
$env:TINY_VOCA_LLM_URL     = "http://127.0.0.1:11434/v1/chat/completions"
$env:TINY_VOCA_LLM_KEY     = "local"
$env:TINY_VOCA_LLM_MODEL   = "qwen2.5:7b"
$env:TINY_VOCA_LLM_TIMEOUT = "30"
```

`TINY_VOCA_LANGUAGE=ar` and `TINY_VOCA_FRONTEND=rules` are set by the module
itself before importing the library.

## 3. Start

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python -m uvicorn server.tts_service:app --host 0.0.0.0 --port 8000
```

Interactive documentation: <http://127.0.0.1:8000/docs>.
The ONNX session is created once at the first request and protected by a lock;
it is never reloaded per request.

## 4. Endpoints

| Method | Path | Body | Reply |
|---|---|---|---|
| `GET` | `/health` | — | language, frontend, voices, sample rate, step, attribution |
| `POST` | `/api/tts` | `{"text": "مَرْحَبًا بِالْعالَمِ."}` | `audio/wav` |
| `POST` | `/api/voice/answer` | `{"question": "كيف حالك؟"}` | `audio/wav` + `X-Answer-Text` header (percent-encoded) |

`/api/tts` fields: `text`, `voice` (`F2`), `speed` (0.95), `pitch_range` (1.35),
`already_vocalized` (`true`). The rules G2P can also read unvocalised Arabic,
but vocalised text is more accurate; set `already_vocalized` to `false` to run
CAMeL tashkeel on the text first. Text with no readable Arabic word returns 400.

`/api/voice/answer` fields: `question`, `voice`, `speed`, `pitch_range`,
`max_chars` (300). Repeated identical questions are served from an in-memory
cache keyed by model, question, length, voice, speed and pitch range.

### Try it

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/tts" `
  -H "Content-Type: application/json" `
  -d '{"text":"مَرْحَبًا بِالْعالَمِ."}' --output reponse.wav

curl.exe -X POST "http://127.0.0.1:8000/api/voice/answer" `
  -H "Content-Type: application/json" `
  -d '{"question":"كيف حالك اليوم؟"}' --output message-vocal.wav
```

Browser client:

```html
<audio controls src="http://127.0.0.1:8000/api/voice/answer"></audio>
```

For a real client, `POST` with `fetch`, read the `X-Answer-Text` header with
`decodeURIComponent`, and pass the response blob to an `<audio>` element.

## 5. Tests

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m unittest discover -s tests -p "test_voice_service.py" -v
python -m unittest discover -s tests         # full suite
```

The tests exercise the real ONNX model and CAMeL Tools but replace the LLM
with a stand-in: they need neither network nor API key.

## 6. Production notes

- **Format**: WAV works in browsers. For WhatsApp/Telegram, convert to Opus/OGG
  in an image that has a **licensed** ffmpeg build:
  `ffmpeg -i reponse.wav -c:a libopus -b:a 32k reponse.ogg`. ffmpeg is not
  installed on the development machine.
- **Performance**: the Arabic ONNX model runs at ~85x real time on 4 CPU
  threads; about 0.3 s of CPU for 5 s of speech. LLM latency usually dominates.
- **Concurrency**: the ONNX session is locked; for more throughput run several
  worker processes behind a load balancer. Put a disk cache and a rate limit in
  front of the LLM call.
- **Limits**: `max_tokens=512`, `max_chars<=600`. CAMeL tashkeel can misread
  proper names and the pronunciation follows that error.
- **Quality**: single voice `F2`, Damascene accent, flatter prosody than the
  reference and occasionally short audio. Not streaming, no voice cloning.
- **Attribution**: all model/data derivatives come from the Arabic Speech
  Corpus (Halabi 2016), CC BY 4.0. Keep the citation in the service metadata
  and in any user-facing page: Halabi, Nawar (2016). *Modern standard Arabic
  phonetics for speech synthesis*. PhD thesis, University of Southampton.

"""HTTP service: Arabic question -> LLM answer -> vocalised text -> voice message.

Run with:  uvicorn server.tts_service:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              model metadata and attribution
    POST /api/tts             text -> WAV (browser/mobile simple path)
    POST /api/voice/answer    question -> LLM -> diacritised -> WAV

The ONNX session is created once, protected by a lock, and never reloaded per
request. The model and G2P are served from ``server/assets``; no remote
repository is contacted at inference time.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import quote

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from .answer import cut_at_sentence, generate_answer, sanitize
from .diacritize import vocalize

DEFAULT_ASSETS = Path(__file__).resolve().parent / "assets"
ATTRIBUTION = (
    "Arabic Speech Corpus (Halabi 2016), CC BY 4.0. "
    "Halabi, Nawar (2016). Modern standard Arabic phonetics for speech synthesis. "
    "PhD thesis, University of Southampton."
)

DEMO_HTML = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tiny-voca - رسالة صوتية</title>
<style>
  body { font-family: Tahoma, Arial, sans-serif; background: #f5f5f7; margin: 0; padding: 2rem; }
  main { max-width: 640px; margin: auto; background: white; padding: 2rem; border-radius: 12px;
         box-shadow: 0 2px 12px #0001; }
  h1 { font-size: 1.3rem; margin-top: 0; }
  input, button { font-size: 1rem; padding: .7rem; border-radius: 6px; border: 1px solid #bbb; }
  input { width: 70%; }
  button { background: #2257b3; color: white; border: 0; cursor: pointer; }
  button:disabled { background: #999; }
  audio { width: 100%; margin-top: 1.2rem; }
  #answer { margin-top: 1rem; font-size: 1.1rem; }
  #status { color: #666; }
  small { display: block; margin-top: 2rem; color: #777; line-height: 1.6; }
</style>
</head>
<body>
<main>
  <h1>اسأل بالكتابة، واسمع الجواب صوتًا</h1>
  <input id="question" value="كيف حالك اليوم؟" placeholder="اكتب سؤالك بالعربية">
  <button id="ask" onclick="ask()">اسأل</button>
  <p id="status"></p>
  <audio id="audio" controls></audio>
  <p id="answer" dir="rtl"></p>
  <small>الصوت من نموذج tiny-voca العربي (Halabi 2016، CC BY 4.0). قد يستغرق الجواب الأول ثوانٍ.</small>
</main>
<script>
const question = document.getElementById("question");
const button = document.getElementById("ask");
const status = document.getElementById("status");
const audio = document.getElementById("audio");
const answer = document.getElementById("answer");
question.addEventListener("keydown", (event) => { if (event.key === "Enter") ask(); });
let previous = null;
async function ask() {
  const text = question.value.trim();
  if (!text) return;
  button.disabled = true;
  status.textContent = "جارٍ إعداد الجواب...";
  answer.textContent = "";
  try {
    const response = await fetch("/api/voice/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text })
    });
    if (!response.ok) throw new Error("HTTP " + response.status);
    answer.textContent = decodeURIComponent(response.headers.get("X-Answer-Text") || "");
    if (previous) URL.revokeObjectURL(previous);
    previous = URL.createObjectURL(await response.blob());
    audio.src = previous;
    await audio.play();
    status.textContent = "تمّ.";
  } catch (error) {
    status.textContent = "حدث خطأ: " + error.message;
  } finally {
    button.disabled = false;
  }
}
</script>
</body>
</html>"""


class VoiceEngine:
    """Loads the ONNX model and the Arabic G2P once."""

    def __init__(self, assets: Path | None = None, threads: int = 4):
        self.assets = Path(assets or os.getenv("TINY_VOCA_ASSETS", DEFAULT_ASSETS))
        # Load the exact G2P file used by the browser demo, under its own module
        # name: importing a plain ``arabic_text`` could pick up another copy
        # already present in sys.modules. Dataclasses need a registered module.
        spec = importlib.util.spec_from_file_location(
            "server_arabic_text", self.assets / "arabic_text.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.arabic_text = module
        self.NothingToPronounce = module.NothingToPronounce

        self.config = json.loads((self.assets / "config.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(self.assets / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()
        self.model = os.getenv("TINY_VOCA_LLM_MODEL", "gpt-4o-mini")

    def phonemize(self, text: str):
        result = self.arabic_text.phonemize(text, diacritized=True)
        return (np.array([result.ids], dtype=np.int64),
                np.array([result.stress], dtype=np.int64))

    def synthesize(self, text: str, speed: float = 0.95, pitch_range: float = 1.35):
        """Vocalised Arabic text -> float32 waveform at the model sample rate."""
        tokens, stress = self.phonemize(text)
        feeds = {
            "tokens": tokens,
            "stress": stress,
            "voice": np.array([0], dtype=np.int64),  # F2, the single Arabic voice
            "speed": np.array([speed], dtype=np.float32),
            "pitch_shift": np.array([0.0], dtype=np.float32),
            "pitch_range": np.array([pitch_range], dtype=np.float32),
        }
        with self._lock:
            audio, _durations = self.session.run(None, feeds)
        # ``durations`` holds one frame count per token, not the sample count.
        # The browser demo plays the audio tensor as-is; do the same here.
        wave = np.asarray(audio[0], dtype=np.float32)
        peak = float(np.abs(wave).max()) if wave.size else 0.0
        if peak > 0.99:
            wave = wave * (0.99 / peak)
        return wave

    def wav_bytes(self, wave: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, wave, self.config["sample_rate"], format="WAV", subtype="FLOAT")
        return buffer.getvalue()

    def check_voice(self, voice: str):
        if voice not in self.config["voices"]:
            raise HTTPException(400, f"unknown voice {voice!r}; available: {self.config['voices']}")


_engine = None
_answer_cache: dict[str, tuple[str, bytes]] = {}


def get_engine() -> VoiceEngine:
    global _engine
    if _engine is None:
        _engine = VoiceEngine()
    return _engine


app = FastAPI(title="tiny-voca Arabic voice answers", version="1.0.0")


class TtsIn(BaseModel):
    text: str
    voice: str = "F2"
    speed: float = Field(0.95, ge=0.25, le=4)
    pitch_range: float = Field(1.35, ge=0, le=3)
    already_vocalized: bool = True


class AnswerIn(BaseModel):
    question: str
    voice: str = "F2"
    speed: float = Field(0.95, ge=0.25, le=4)
    pitch_range: float = Field(1.35, ge=0, le=3)
    max_chars: int = Field(300, ge=40, le=600)


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    """A ready-to-use RTL client: type a question, hear the voice answer."""
    return DEMO_HTML


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "language": engine.config["language"],
        "frontend": engine.config["frontend"],
        "voices": engine.config["voices"],
        "sample_rate": engine.config["sample_rate"],
        "step": engine.config["step"],
        "attribution": ATTRIBUTION,
    }


@app.post("/api/tts")
def tts(body: TtsIn):
    engine = get_engine()
    engine.check_voice(body.voice)
    text = body.text if body.already_vocalized else vocalize(body.text)
    text = sanitize(text)
    if not text:
        raise HTTPException(400, "nothing to pronounce after cleanup")
    try:
        wave = engine.synthesize(text, body.speed, body.pitch_range)
    except engine.NothingToPronounce as error:
        raise HTTPException(
            400, "nothing to pronounce: send Arabic text (vocalised text is more accurate)"
        ) from error
    return Response(engine.wav_bytes(wave), media_type="audio/wav")


@app.post("/api/voice/answer")
def voice_answer(body: AnswerIn):
    """Question in text, voice message out. The answer text is returned in a
    percent-encoded header so the client can display it next to the audio."""
    engine = get_engine()
    engine.check_voice(body.voice)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    key = hashlib.sha256(
        f"{engine.model}|{question}|{body.max_chars}|{body.voice}|"
        f"{body.speed}|{body.pitch_range}".encode("utf-8")
    ).hexdigest()
    cached = _answer_cache.get(key)
    if cached is None:
        try:
            answer = sanitize(cut_at_sentence(generate_answer(question), body.max_chars))
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        except Exception as error:
            raise HTTPException(502, f"answer generation failed: {error}") from error
        if not answer:
            raise HTTPException(502, "the answer contained nothing pronounceable")
        try:
            wave = engine.synthesize(vocalize(answer), body.speed, body.pitch_range)
        except engine.NothingToPronounce as error:
            raise HTTPException(502, "the answer contained nothing pronounceable") from error
        cached = (answer, engine.wav_bytes(wave))
        _answer_cache[key] = cached
    answer, audio = cached
    return Response(
        audio,
        media_type="audio/wav",
        headers={"X-Answer-Text": quote(answer, safe="")},
    )

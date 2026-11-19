"""Voice-answer service: answer cleanup, real ONNX synthesis and HTTP endpoints.

The model is loaded once from ``server/assets`` (the exact browser demo model).
The LLM is always replaced by a stand-in: no network and no API key are needed.
"""

import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import tts_service
from server.answer import cut_at_sentence, generate_answer, sanitize

ASSETS = ROOT / "server" / "assets"
VOCALIZED = "مَرْحَبًا بِالْعالَمِ."
PLAIN = "مرحبا بالعالم"
_ENGINE = None


def engine():
    """Load the real 21 MB ONNX model once for the whole suite."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = tts_service.VoiceEngine(ASSETS, threads=2)
    return _ENGINE


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}

    def raise_for_status(self):
        return None


class AnswerTextTests(unittest.TestCase):
    def test_sanitize_keeps_arabic_and_drops_latin_and_emoji(self):
        self.assertEqual(sanitize("مرحبا 😊 hello 42، كيف حالك؟"), "مرحبا 42، كيف حالك؟")
        self.assertEqual(sanitize("مرحبا"), "مرحبا")                     # unvocalised Arabic survives
        self.assertEqual(sanitize(VOCALIZED), VOCALIZED)                 # diacritics survive
        self.assertEqual(sanitize("😊 XD https://x.y"), ": .")   # punctuation survives, letters do not

    def test_cut_at_sentence_prefers_a_mark_in_the_second_half(self):
        text = "أولاً جملة طويلة عن الموضوع. ثم جملة أخرى يجب ألا تظهر في الإجابة."
        self.assertEqual(cut_at_sentence(text, 40), "أولاً جملة طويلة عن الموضوع.")
        self.assertEqual(cut_at_sentence("مرحبا", 10), "مرحبا")          # short text is untouched

    def test_cut_at_sentence_falls_back_to_a_word_boundary(self):
        text = "واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية"
        trimmed = cut_at_sentence(text, 20)
        self.assertLessEqual(len(trimmed), 20)
        self.assertTrue(trimmed)
        self.assertTrue(text.startswith(trimmed))

    def test_generate_answer_uses_the_configured_openai_compatible_endpoint(self):
        env = {
            "TINY_VOCA_LLM_URL": "http://localhost:11434/v1/chat/completions",
            "TINY_VOCA_LLM_KEY": "local-key",
            "TINY_VOCA_LLM_MODEL": "llama3",
            "TINY_VOCA_LLM_TIMEOUT": "5",
        }
        client = Mock(post=Mock(return_value=FakeResponse("  مرحبا بك  ")))
        with patch.dict(os.environ, env):
            answer = generate_answer("كيف حالك؟", client=client)
        self.assertEqual(answer, "مرحبا بك")
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], env["TINY_VOCA_LLM_URL"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer local-key")
        self.assertEqual(kwargs["timeout"], 5.0)
        self.assertEqual(kwargs["json"]["model"], "llama3")
        self.assertEqual(kwargs["json"]["messages"][-1], {"role": "user", "content": "كيف حالك؟"})
        self.assertIn("Modern Standard Arabic", kwargs["json"]["messages"][0]["content"])

    def test_generate_answer_without_key_explains_how_to_set_it(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TINY_VOCA_LLM_KEY"):
                generate_answer("سؤال")


@unittest.skipUnless(importlib.util.find_spec("camel_tools"), "needs camel-tools")
class DiacritizationTests(unittest.TestCase):
    def test_camel_adds_diacritics_and_caches_the_result(self):
        from server.diacritize import vocalize

        plain = "مرحبا بالعالم"
        result = vocalize(plain)
        self.assertNotEqual(result, plain)
        self.assertTrue(any("\u064b" <= char <= "\u065f" for char in result), result)
        self.assertIs(vocalize(plain), result)                           # lru_cache hits


class VoiceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speech = engine()

    def test_assets_are_the_arabic_rules_model(self):
        config = self.speech.config
        self.assertEqual(config["language"], "ar")
        self.assertEqual(config["frontend"], "rules")
        self.assertEqual(config["voices"], ["F2"])
        self.assertEqual(config["sample_rate"], 24000)
        self.assertEqual(config["step"], 3000)
        self.assertEqual(config["data_source"]["license"], "CC BY 4.0")
        names = [item.name for item in self.speech.session.get_inputs()]
        self.assertEqual(names, config["inputs"])                        # same contract as worker.js
        self.assertEqual(self.speech.session.get_inputs()[0].shape[0], 1)

    def test_only_f2_is_available(self):
        self.assertIsNone(self.speech.check_voice("F2"))
        with self.assertRaises(Exception) as error:
            self.speech.check_voice("M2")
        self.assertEqual(getattr(error.exception, "status_code", None), 400)

    def test_real_synthesis_produces_a_wav_that_can_be_read(self):
        wave = self.speech.synthesize(VOCALIZED)
        self.assertIsInstance(wave, np.ndarray)
        self.assertGreater(len(wave), 2400)                              # more than 0.1 s
        self.assertLessEqual(float(np.abs(wave).max()), 0.99 + 1e-6)
        raw = self.speech.wav_bytes(wave)
        self.assertTrue(raw.startswith(b"RIFF"))
        samples, rate = sf.read(io.BytesIO(raw), dtype="float32")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(samples), len(wave))
        np.testing.assert_allclose(samples, wave, atol=1e-6)             # float WAV: no quantisation

    def test_speed_and_pitch_range_change_the_output(self):
        slow = self.speech.synthesize(VOCALIZED, speed=0.7)
        fast = self.speech.synthesize(VOCALIZED, speed=1.3)
        self.assertGreater(len(slow), len(fast))
        self.assertFalse(np.array_equal(slow[:len(fast)], fast))


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speech = engine()
        cls.client = TestClient(tts_service.app)

    def setUp(self):
        tts_service._engine = self.speech
        tts_service._answer_cache.clear()

    def wav(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("audio/wav", response.headers["content-type"])
        samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        self.assertEqual(rate, 24000)
        self.assertGreater(len(samples), 0)
        return samples

    def test_demo_page_calls_the_voice_answer_endpoint(self):
        response = self.client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn('lang="ar" dir="rtl"', response.text)
        self.assertIn('fetch("/api/voice/answer"', response.text)
        self.assertIn("<audio", response.text)

    def test_health_reports_language_frontend_voice_and_attribution(self):
        data = self.client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual((data["language"], data["frontend"]), ("ar", "rules"))
        self.assertEqual(data["voices"], ["F2"])
        self.assertEqual(data["sample_rate"], 24000)
        self.assertIn("Halabi, Nawar (2016)", data["attribution"])

    def test_tts_accepts_vocalized_text(self):
        self.wav(self.client.post("/api/tts", json={"text": VOCALIZED}))

    def test_tts_can_vocalize_plain_text_on_request(self):
        with patch.object(tts_service, "vocalize", return_value=VOCALIZED) as vocalize:
            self.wav(self.client.post("/api/tts", json={"text": PLAIN, "already_vocalized": False}))
        vocalize.assert_called_once_with(PLAIN)

    def test_tts_rejects_unknown_voice_and_empty_text(self):
        self.assertEqual(self.client.post("/api/tts", json={"text": VOCALIZED, "voice": "M2"}).status_code, 400)
        self.assertEqual(self.client.post("/api/tts", json={"text": "😊 XD"}).status_code, 400)

    def test_rules_front_end_also_reads_unvocalized_arabic(self):
        response = self.client.post("/api/tts", json={"text": "مرحبا بالعالم"})
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(self.wav(response)), 0)

    def test_voice_answer_returns_audio_and_the_plain_answer(self):
        answer = "مرحبا بك، كيف يمكنني مساعدتك؟"
        with patch.object(tts_service, "generate_answer", return_value=answer) as generated, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED) as vocalize:
            response = self.client.post("/api/voice/answer", json={"question": "مرحبا"})
        self.wav(response)
        self.assertEqual(unquote(response.headers["X-Answer-Text"]), answer)
        generated.assert_called_once_with("مرحبا")
        vocalize.assert_called_once_with(answer)

    def test_voice_answer_caches_identical_questions(self):
        answer = "نعم،那是 صحيح."                                     # Latin/Chinese removed by sanitize
        with patch.object(tts_service, "generate_answer", return_value=answer) as generated, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED):
            request = {"question": "هل تعمل؟"}
            first = self.client.post("/api/voice/answer", json=request)
            second = self.client.post("/api/voice/answer", json=request)
        self.assertEqual(first.content, second.content)
        self.assertEqual(generated.call_count, 1)
        self.assertEqual(unquote(first.headers["X-Answer-Text"]), "نعم، صحيح.")

    def test_voice_answer_reports_missing_llm_key_as_service_unavailable(self):
        with patch.object(tts_service, "generate_answer", side_effect=RuntimeError("no key")):
            response = self.client.post("/api/voice/answer", json={"question": "مرحبا"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("no key", response.json()["detail"])

    def test_voice_answer_rejects_an_empty_question(self):
        self.assertEqual(self.client.post("/api/voice/answer", json={"question": "  "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()

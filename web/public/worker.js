// Runs the Arabic phonemizer (the rules front end, through Pyodide) and the model (ONNX Runtime Web)
// off the main thread. Messages in: {type: "load"} and {type: "speak", id, text, voice, speed}.
// Messages out: status, ready, sentence (with its audio, as soon as it is made), done (with the whole clip),
// error (plain: true when the message is meant for the reader as it is). The G2P is the pure-Python module
// shipped verbatim as assets/arabic_text.py: no dictionary and no external program.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let pyodide, phonemize, session, config;

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Bump when the files under assets/ change, so browsers fetch them again.
const VERSION = "3";

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  status("Loading the phonemizer");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  // rules is dependency-free: no extra packages, no lexicon, no archive to unpack.
  const source = await download("assets/arabic_text.py", "Loading the phonemizer");
  status("Starting the phonemizer");
  pyodide.FS.mkdirTree("/arabic-frontend");
  pyodide.FS.writeFile("/arabic-frontend/arabic_text.py", source);
  pyodide.runPython(`
import json, sys
sys.path.insert(0, "/arabic-frontend")
import arabic_text

def phonemize_json(text):
    try:
        result = arabic_text.phonemize(text, diacritized=True)
    except ValueError:
        # No word, no number or an empty string: nothing to say, the sentence is skipped.
        return "null"
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress),
                       "text": " ".join(result.symbols)})

phonemize_json("مَرْحَبًا بِالْعالَمِ")
`);
  phonemize = pyodide.globals.get("phonemize_json");
}

async function loadModel() {
  config = await (await fetch("assets/config.json")).json();
  const bytes = await download("assets/model.onnx", "Loading the model");
  status("Starting the model");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

const ABBREVIATIONS = /(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$/i;

// Sentences, without breaking after titles, initials or inside numbers.
// The English path is the one tiny_voca.infer.sentences uses and must stay identical: the two are tested together.
// The Arabic path splits after Arabic clause/sentence marks (، ؛ ؟ . ! …) and needs no following capital letter
// or abbreviation table, since the Arabic front end expands its own abbreviations.
function sentences(text) {
  if (/[\u0600-\u06FF]/u.test(text)) {
    return text.replace(/\s+/g, " ").trim()
      .split(/(?<=[،؛؟.!…]["'”’)\]»]?)\s+(?=\S)/u)
      .filter((part) => /[\p{L}\p{N}]/u.test(part));
  }
  const parts = [];
  let current = "";
  for (const piece of text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…]["')\]]?)\s+(?=["'(\[]?[A-Z0-9])/)) {
    current = current ? current + " " + piece : piece;
    if (!ABBREVIATIONS.test(current)) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// A full stop is added when the text ends without a mark: before a final mark the voice lengthens the last
// syllable and lets the sound fall, without one the audio breaks off. A face or another sign after the last word
// is no mark ("see you :)"), a comma or a colon is one. tiny_voca.infer.finished does the same for English;
// keep that path byte-for-byte. Arabic marks (، ؛ ؟ . ! …) count as marks too, also after a closing quote.
function finished(text) {
  if (/[\u0600-\u06FF]/u.test(text)) {
    const arabicBody = text.replace(/\s+$/, "");
    if (!arabicBody) return text;
    const arabicLast = arabicBody.split(/\s+/).pop().replace(/["'”’)\]»]+$/, "").slice(-1);
    return /[.،؛؟!…,;:]/.test(arabicLast) ? text : arabicBody + ".";
  }
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  const last = body.split(/\s+/).pop();
  const marked = /[\p{L}\p{N}]/u.test(last) ? /[.!?…,;:]["'”’)\]»]*$/.test(last) : /[.!?…]/.test(last) || /^[,;:]$/.test(last);
  return marked ? text : body + ".";
}

// One utterance holds at most max_tokens phonemes; longer sentences are cut at clause marks, then by words.
// Export writes max_tokens at the top level; older configs keep it under "model".
function fit(sentence) {
  const result = JSON.parse(phonemize(sentence));
  if (!result) return [];
  const maxTokens = config.max_tokens ?? config.model?.max_tokens ?? 512;
  if (result.ids.length <= maxTokens - 12) return [result];
  const pieces = sentence.split(/(?<=[،؛؟.!—])\s+/u);
  if (pieces.length > 1) return pieces.flatMap(fit);
  const words = sentence.split(" ");
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [...fit(words.slice(0, half).join(" ")), ...fit(words.slice(half).join(" "))];
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
  });
  return output.audio.data;
}

// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// Shown when no sentence of a text could be read. tiny_voca.infer.speak_text says the same.
const ARABIC_ONLY = "يقرأ tiny-voca النص العربي المشكّل فقط.";

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  let phonemeTime = 0;
  let modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = fit(list[index]);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const audio = await synthesize(phones, voice, speed);
      modelTime += performance.now() - mark;
      chunks.push(audio);
      // The page plays this sentence at once, before the whole clip is known: a copy, kept below full scale by itself.
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: ARABIC_ONLY, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadPhonemizer(), loadModel()]);
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};

"""Build tiny-voca JSONL manifests from the Halabi Arabic Speech Corpus.

Downloads the official 1.19 GB archive once (unless --archive is given),
converts its Buckwalter transcripts to Arabic script, resamples the 48 kHz
recordings to the 24 kHz that tiny_voca.prepare requires, and writes one
manifest per split plus a QC report.

Example:
    PYTHONIOENCODING=utf-8 TINY_VOCA_LANGUAGE=ar \
      python scripts/build_halabi_manifest.py --output data/halabi

Nothing here trains a model. The archive and resampled WAVs stay in data/,
which is git-ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_voca import arabic_data as ad  # noqa: E402

TRAIN_ROOT = "arabic-speech-corpus"
TEST_ROOT = "arabic-speech-corpus/test set"
# The training file name is misspelled in the official archive ("transcipt").
PHONETIC_MEMBER = {
    "train": f"{TRAIN_ROOT}/phonetic-transcipt.txt",
    "test": f"{TEST_ROOT}/phonetic-transcript.txt",
}
NUMBER = re.compile(r"(\d+)")


def download(url, destination):
    """Resumable download. The server supports HTTP Range; a dropped 1.2 GB
    transfer must not force a fresh start.
    """
    with urlopen(Request(url, method="HEAD"), timeout=60) as response:
        total = int(response.headers["Content-Length"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = destination.stat().st_size if destination.exists() else 0
    if current > total:
        destination.unlink()
        current = 0
    if current == total:
        print(f"archive already complete: {destination}", flush=True)
        return
    started = time.time()
    while current < total:
        try:
            request = Request(url, headers={"Range": f"bytes={current}-"})
            with urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"server ignored Range: {response.status}")
                mode = "ab" if current else "wb"
                with destination.open(mode) as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        current += len(block)
                    stream.flush()
            print(f"downloaded {current / 2**20:.0f} / {total / 2**20:.0f} MiB", flush=True)
        except (URLError, TimeoutError, RuntimeError) as error:
            current = destination.stat().st_size if destination.exists() else 0
            print(f"download interrupted ({error}); resuming at {current / 2**20:.0f} MiB", flush=True)
            if current >= total:
                break
            time.sleep(2)
    if destination.stat().st_size != total:
        raise RuntimeError(f"incomplete archive: {destination.stat().st_size} != {total}")
    print(f"download finished in {time.time() - started:.0f}s", flush=True)


def recording_name(path):
    """Exact member name, including the corpus's double spaces."""
    return Path(path).name


def identifier(recording, split):
    digits = NUMBER.findall(recording)
    number = digits[-1] if digits else re.sub(r"\W+", "_", recording).strip("_")
    return f"halabi_{split}_{number}"


def split_members(archive, root, split):
    """Map exact wav name -> (wav member, Buckwalter, phonetic transcript)."""
    transcript = ad.parse_transcript_from_zip(archive, f"{root}/orthographic-transcript.txt")
    phonetic = ad.parse_transcript_from_zip(archive, PHONETIC_MEMBER[split])
    prefix = f"{root}/wav/"
    available = {recording_name(name): name for name in archive.namelist()
                 if name.startswith(prefix) and name.lower().endswith(".wav")}
    missing = sorted(set(transcript) - set(available))
    if missing:
        raise ValueError(f"{split}: {len(missing)} transcripts have no recording, e.g. {missing[:3]}")
    unmatched = sorted(set(transcript) ^ set(phonetic))
    if unmatched:
        print(f"WARNING {split}: {len(unmatched)} names differ between orthographic and phonetic transcripts", flush=True)
    return {name: (available[name], value, phonetic[name]) for name, value in transcript.items() if name in phonetic}


def resample_wave(payload, target_rate):
    wave, rate = sf.read(BytesIO(payload), dtype="float32", always_2d=False)
    if wave.ndim != 1:
        if wave.shape[1] == 1:
            wave = wave[:, 0]
        else:
            wave = wave.mean(axis=1)
    if rate != target_rate:
        divisor = np.gcd(rate, target_rate)
        wave = resample_poly(wave, target_rate // divisor, rate // divisor).astype("float32")
    return wave, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "halabi")
    parser.add_argument("--archive", type=Path, help="Use a local copy of arabic-speech-corpus.zip")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Only the first N recordings per official split (smoke test)")
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    import zipfile

    archive_path = args.archive or args.output / "arabic-speech-corpus.zip"
    if args.archive is None:
        download(ad.SOURCE_URL, archive_path)
    if not archive_path.is_file():
        raise SystemExit(f"archive not found: {archive_path}")

    output = args.output
    manifests = output / "manifests"
    wav_dir = output / "wav24"
    manifests.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    cross_checks = []
    conversion_stats = Counter()
    unmapped_labels = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for official, root in (("train", TRAIN_ROOT), ("test", TEST_ROOT)):
            members = split_members(archive, root, official)
            if args.limit:
                members = dict(list(members.items())[:args.limit])
            print(f"{official}: {len(members)} recordings", flush=True)
            for recording, (member, buckwalter, phonetic) in members.items():
                text = ad.buckwalter_to_arabic(buckwalter, conversion_stats)
                identity = identifier(recording, official)
                target = wav_dir / (identity + ".wav")
                if not target.exists():
                    with archive.open(member) as source:
                        wave, original_rate = resample_wave(source.read(), args.sample_rate)
                    if len(wave) < 256:
                        raise ValueError(f"recording too short: {recording}")
                    sf.write(target, wave, args.sample_rate, subtype="PCM_16")
                raw_rows.append(ad.manifest_row(identity, text, Path("../wav24") / target.name, official, recording))
                from tiny_voca import arabic_text
                ours = [symbol for symbol in arabic_text.phonemize(text, diacritized=True).symbols
                        if symbol in arabic_text.PHONE_SET]
                theirs, unmapped = ad.corpus_phones(phonetic)
                agreement = ad.sequence_agreement(theirs, ours)
                if agreement:
                    cross_checks.append({"recording": recording, "text": text, **agreement})
                unmapped_labels.update(unmapped)

    if unmapped_labels:
        print(f"WARNING: {sum(unmapped_labels.values())} unmapped corpus labels: {dict(unmapped_labels)}", flush=True)
    splits = ad.assign_splits(raw_rows, args.validation_fraction, args.seed)
    summary = {"source": ad.SOURCE_PAGE, "license": ad.SOURCE_LICENSE, "citation": ad.SOURCE_CITATION,
               "sample_rate": args.sample_rate, "seed": args.seed,
               "validation_fraction": args.validation_fraction,
               "implicit_alif_marks": conversion_stats["implicit_alif_marks"],
               "unmapped_corpus_labels": dict(unmapped_labels), "splits": {}}
    for split, rows in splits.items():
        ad.write_manifest(manifests / f"{split}.jsonl", rows)
        summary["splits"][split] = {"rows": len(rows), **ad.g2p_report(rows)}
    summary["hashes"] = {split: hashlib.sha256((manifests / f"{split}.jsonl").read_bytes()).hexdigest()
                         for split in splits}
    (output / "build.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    agreements = [item["agreement"] for item in cross_checks]
    similarities = [item["similarity"] for item in cross_checks]
    ratios = [item["length_ratio"] for item in cross_checks]
    lines = [
        "# Contrôle qualité Halabi — Phase 2",
        "",
        f"Source : {ad.SOURCE_PAGE} — licence {ad.SOURCE_LICENSE}.",
        f"Citation : {ad.SOURCE_CITATION}",
        "",
        "Comparaison **diagnostique** entre les phonèmes produits par notre G2P",
        "sur l arabe converti et le transcript phonétique fourni par le corpus.",
        "L accord positionnel est décalé par le ʔ initial que notre G2P ajoute",
        "pour le hamzat al-wasl (absent de la transcription damascène) : la",
        "**similarité** (distance d édition) est le bon indicateur. L accent et",
        "les différences d alphabet expliquent un score inférieur à 1 : ce n est",
        "pas un critère d acceptation, mais un détecteur d erreur de conversion.",
        "",
        f"- énoncés comparés : {len(cross_checks)}",
        f"- similarité moyenne : {statistics.mean(similarities):.3f}" if similarities else "- similarité : n/a",
        f"- similarité médiane : {statistics.median(similarities):.3f}" if similarities else "",
        f"- accord positionnel moyen : {statistics.mean(agreements):.3f}" if agreements else "",
        f"- rapport de longueur moyen : {statistics.mean(ratios):.3f}" if ratios else "",
        f"- alifs implicites restaurés : {conversion_stats['implicit_alif_marks']}",
        f"- labels phonétiques non mappés : {dict(unmapped_labels) or 'aucun'}",
        "",
        "## Énoncés les plus divergents",
        "",
        "| Enregistrement | similarité | longueur G2P / corpus | texte converti |",
        "|---|---:|---:|---|",
    ]
    for item in sorted(cross_checks, key=lambda value: value["similarity"])[:10]:
        lines.append(f"| {item['recording']} | {item['similarity']:.3f} | "
                     f"{item['g2p_phones']} / {item['corpus_phones']} | {item['text'][:80]} |")
    lines.extend(["", "## Marqueurs inconnus par split", ""])
    for split, values in summary["splits"].items():
        lines.append(f"- {split} : {values['unknown_graphemes'] or 'aucun'}")
    lines.append("")
    (output / "qc.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

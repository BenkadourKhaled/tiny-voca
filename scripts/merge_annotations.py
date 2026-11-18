"""Merge CPU-parallel annotation shards into one training-ready directory.

Each shard is a normal ``tiny_voca.annotate`` output made with identical
settings, but with a subset of the rows. The merged identity is exactly the one
a single full run would have written (same fields, ``sources`` rebuilt from the
complete feature manifests), and every row's ``annotation`` fingerprint is
rewritten to that identity. Only rows, rejects and the phoneme cache are copied;
the per-shard journals and timing probes remain in the shard directories.

Usage:
    python scripts/merge_annotations.py --shards data/halabi/annotations-shards/train-* \
        data/halabi/annotations-shards/train-early \
        data/halabi/annotations-shards/validation data/halabi/annotations-shards/test \
        --features data/halabi/features --output data/halabi/annotations
"""

import argparse
import hashlib
import json
from pathlib import Path

from tiny_voca.config import Config
from tiny_voca.journal import file_hash

SPLITS = ("train", "validation", "test")


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                            for record in records), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config()

    metadatas = [json.loads((shard / "annotations.json").read_text(encoding="utf-8")) for shard in args.shards]
    first = metadatas[0]
    for metadata in metadatas[1:]:
        for key in ("annotation_version", "acoustic", "frontend", "arabic_frontend", "aligner",
                    "alignment", "dataset_revision", "limit"):
            if metadata["identity"].get(key) != first["identity"].get(key):
                raise ValueError(f"annotation shards disagree on {key}")
        if metadata["stage"] != first["stage"] or metadata["limits"] != first["limits"]:
            raise ValueError("annotation shards disagree on stage or QC limits")
    identity = dict(first["identity"])
    identity["sources"] = {split: file_hash(args.features / (split + ".jsonl")) for split in SPLITS}
    if identity["limit"]:
        raise ValueError("shards were made with --limit; a full annotation is required for training")
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    args.output.mkdir(parents=True, exist_ok=True)
    counts, rejected_counts, reasons, hours = {}, {}, {}, {}
    seconds_per_frame = config.hop / config.sample_rate
    for split in SPLITS:
        merged, seen = [], set()
        rejects, seen_rejects = [], set()
        for shard in args.shards:
            for row in rows(shard / (split + ".jsonl")):
                if row["id"] in seen:
                    continue
                row["annotation"] = fingerprint
                merged.append(row)
                seen.add(row["id"])
            for row in rows(shard / (split + ".rejected.jsonl")):
                if row["id"] in seen_rejects or row["id"] in seen:
                    continue
                rejects.append(row)
                seen_rejects.add(row["id"])
                for reason in row.get("reasons", []):
                    key = reason.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
        merged.sort(key=lambda row: row["id"])
        rejects.sort(key=lambda row: row["id"])
        expected = len(rows(args.features / (split + ".jsonl")))
        if len(merged) + len(rejects) != expected:
            raise ValueError(f"{split}: {len(merged)} accepted + {len(rejects)} rejected != {expected} feature rows")
        if set(seen) & set(seen_rejects):
            raise ValueError(f"{split}: an id is both accepted and rejected")
        write_jsonl(args.output / (split + ".jsonl"), merged)
        write_jsonl(args.output / (split + ".rejected.jsonl"), rejects)
        counts[split] = len(merged)
        rejected_counts[split] = len(rejects)
        hours[split] = round(sum(row["frames"] for row in merged) * seconds_per_frame / 3600, 3)

    phonemes, seen = [], set()
    for shard in args.shards:
        for row in rows(shard / "phonemes.jsonl"):
            if row["text_hash"] not in seen:
                phonemes.append(row)
                seen.add(row["text_hash"])
    write_jsonl(args.output / "phonemes.jsonl", phonemes)

    metadata = {key: first[key] for key in ("stage", "limits")}
    metadata.update({"identity": identity, "fingerprint": fingerprint, "complete": True,
                     "acoustic_root": str(args.features.resolve()), "counts": counts})
    (args.output / "annotations.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
    summary = {"counts": counts, "rejected": rejected_counts, "hours": hours,
               "rejection_reasons": reasons, "shards": [str(shard) for shard in args.shards]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"fingerprint": fingerprint, "counts": counts, "rejected": rejected_counts,
                      "phonemes": len(phonemes)}))


if __name__ == "__main__":
    main()

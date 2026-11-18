"""Split remaining Halabi train rows into shards for parallel CPU preparation.

The shipped ``tiny_voca.prepare`` processes one manifest in one process. On a
CPU-only machine the pitch tracker dominates and a single split takes hours.
This helper keeps the durable journal contract: each shard writes its own
output directory, then ``merge`` copies accepted rows back into the main
features directory with their original ``record_key`` (source index and row
index of the full manifest), so the main preparation identity stays valid.

Usage:
    python scripts/parallel_prepare_train.py split --shards 8
    python scripts/parallel_prepare_train.py merge --shards 8
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "halabi" / "features"
MANIFEST = ROOT / "data" / "halabi" / "manifests" / "train.jsonl"
WORK = ROOT / "data" / "halabi" / "features-shards"


def complete_lines(path):
    """Read only complete, JSON-valid records; ignore a torn final append."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.endswith("\n") and line.strip():
            rows.append(json.loads(line))
    return rows


def source_rows():
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def split(shards):
    WORK.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in complete_lines(FEATURES / "train.jsonl")}
    rows = source_rows()
    mapping = {row["id"]: index for index, row in enumerate(rows)}
    remaining = [(index, row) for index, row in enumerate(rows) if row["id"] not in done]
    for shard in range(shards):
        path = WORK / f"train-{shard}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for _, row in remaining[shard::shards]), encoding="utf-8")
    (WORK / "index.json").write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"shards": shards, "already_done": len(done), "remaining": len(remaining),
                      "per_shard": [len(remaining[shard::shards]) for shard in range(shards)]}))


def merge(shards):
    mapping = json.loads((WORK / "index.json").read_text(encoding="utf-8"))
    accepted = complete_lines(FEATURES / "train.jsonl")
    known = {row["id"] for row in accepted}
    rejects = complete_lines(FEATURES / "train.rejected.jsonl")
    added, files = 0, 0
    with (FEATURES / "train.jsonl").open("a", encoding="utf-8") as target:
        for shard in range(shards):
            out = WORK / f"out-{shard}"
            for row in complete_lines(out / "train.jsonl"):
                if row["id"] in known:
                    continue
                row["record_key"] = f"0:{mapping[row['id']]}"
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                known.add(row["id"])
                added += 1
            for row in complete_lines(out / "train.rejected.jsonl"):
                identifier = row.get("id")
                if identifier in mapping and identifier not in known:
                    row["record_key"] = f"0:{mapping[identifier]}"
                    rejects.append(row)
                    known.add(identifier)
            source = out / "train"
            if source.is_dir():
                for feature in source.glob("*.pt"):
                    destination = FEATURES / "train" / feature.name
                    if not destination.exists():
                        destination.write_bytes(feature.read_bytes())
                        files += 1
    metadata_path = FEATURES / "train.preparation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"complete": True, "accepted": len(accepted) + added, "rejected": len(rejects)})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (FEATURES / "train.rejected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rejects), encoding="utf-8")
    print(json.dumps({"accepted_before": len(accepted), "accepted_added": added,
                      "accepted_total": len(accepted) + added, "rejected": len(rejects), "feature_files_copied": files}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("split", "merge"))
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    (split if args.action == "split" else merge)(args.shards)

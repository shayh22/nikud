"""שלב 2 — בניית הקורפוס.

ניקוי ל-jsonl, נרמול NFC, סינון ניקוד חלקי, ותיוג שכבה לכל משפט.
סט ההערכה מסונן החוצה כאן — זה המקום היחיד שמבטיח שאין דליפה.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import is_aramaic  # noqa: E402
from clean import clean_line, is_fully_vocalized, split_sentences  # noqa: E402
from fetch_sefaria import commercial_ok  # noqa: E402
from hebrew import normalize, strip_nikud, words  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_DIR = ROOT / "data" / "eval"


def _layer_for(sec: dict, sentence: str) -> str:
    """תיוג שכבה ברמת המשפט. התלמוד מערבב עברית וארמית באותו דף."""
    if sec["source"] == "talmud":
        return "aramaic" if is_aramaic(sentence) else "mishnaic"
    return sec["layer"]


def build(raw_path: Path, out_dir: Path, *, eval_dir: Path,
          commercial_safe: bool = False) -> dict:
    heldout_path = eval_dir / "heldout_refs.json"
    heldout: set[str] = set()
    if heldout_path.exists():
        heldout = set(json.loads(heldout_path.read_text(encoding="utf-8")))
    else:
        print(
            "אזהרה: אין heldout_refs.json. מריצים build_eval.py קודם, "
            "אחרת אין ערובה שסט ההערכה לא דלף לאימון.",
            file=sys.stderr,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}
    stats = Counter()
    seen_hashes: set[str] = set()
    licenses: Counter = Counter()

    def handle(layer: str):
        if layer not in handles:
            handles[layer] = (out_dir / f"{layer}.jsonl").open("w", encoding="utf-8")
        return handles[layer]

    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sec = json.loads(line)
            stats["sections_read"] += 1

            # אותה פסיקה שמייצרת את LICENSES.md, כדי שלא יהיו שני כללים
            # שונים לאותה שאלה. רישיון לא ידוע נחשב אסור.
            if commercial_safe and not commercial_ok(sec.get("license") or ""):
                stats["sections_skipped_license"] += 1
                continue
            if sec["ref"] in heldout:
                stats["sections_skipped_heldout"] += 1
                continue

            for raw_line in sec["lines"]:
                cleaned = clean_line(raw_line)
                for sent in split_sentences(cleaned):
                    stats["sentences_seen"] += 1
                    if not is_fully_vocalized(sent):
                        stats["sentences_dropped_partial_nikud"] += 1
                        continue
                    sent = normalize(sent)
                    key = hashlib.sha1(strip_nikud(sent).encode("utf-8")).hexdigest()
                    if key in seen_hashes:
                        stats["sentences_dropped_duplicate"] += 1
                        continue
                    seen_hashes.add(key)

                    layer = _layer_for(sec, sent)
                    n_words = len(words(sent))
                    handle(layer).write(
                        json.dumps(
                            {
                                "ref": sec["ref"],
                                "source": sec["source"],
                                "layer": layer,
                                "license": sec["license"],
                                "text": sent,
                                "n_words": n_words,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["sentences_kept"] += 1
                    stats[f"words_{layer}"] += n_words
                    stats[f"sentences_{layer}"] += 1
                    licenses[sec["license"]] += n_words

    for fh_out in handles.values():
        fh_out.close()

    summary = {
        "stats": dict(stats),
        "total_words": sum(v for k, v in stats.items() if k.startswith("words_")),
        "words_by_license": dict(licenses),
        "layers": sorted(handles),
        "commercial_safe": commercial_safe,
        "heldout_refs_excluded": len(heldout),
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_corpus(corpus_dir: Path = CORPUS_DIR, layers: list[str] | None = None):
    """מייצר שורות מהקורפוס. עצל, כדי שקורפוס גדול ייכנס לזיכרון של 16GB."""
    for path in sorted(corpus_dir.glob("*.jsonl")):
        layer = path.stem
        if layers and layer not in layers:
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="בניית הקורפוס המנוקד")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "sections.jsonl")
    p.add_argument("--out", type=Path, default=CORPUS_DIR)
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    p.add_argument("--commercial-safe", action="store_true",
                   help="השמטת מקורות CC-BY-NC — המסלול למוצר מסחרי")
    args = p.parse_args(argv)
    summary = build(args.raw, args.out, eval_dir=args.eval_dir,
                    commercial_safe=args.commercial_safe)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

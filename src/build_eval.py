"""שלב 1 — סט ההערכה המוקפא.

2,000 משפטים רבניים מנוקדים — משנה, גמרא, סידור — שלא ייכנסו לעולם לאימון.
הבחירה היא ברמת המקטע (ref) ולא ברמת המשפט, כדי שלא תהיה דליפה בתוך פרק.

התוצר:
  data/eval/eval_set.jsonl      — המשפטים
  data/eval/heldout_refs.json   — ה-refs שנחסמים ל-build_corpus
  data/eval/MANIFEST.json       — hash, ספירות, פרמטרי הבחירה
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import is_aramaic  # noqa: E402
from clean import clean_line, is_fully_vocalized, split_sentences  # noqa: E402
from hebrew import normalize, words  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"

# כמה משפטים מכל שכבה. סך הכול 2,000.
QUOTA = {"mishnah": 800, "talmud": 800, "siddur": 400}
SEED = 20260916


def _iter_sections(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(raw_path: Path, out_dir: Path, *, quota: dict[str, int], seed: int) -> dict:
    by_source: dict[str, list[dict]] = {k: [] for k in quota}
    for sec in _iter_sections(raw_path):
        if sec["source"] in by_source:
            by_source[sec["source"]].append(sec)

    rng = random.Random(seed)
    rows: list[dict] = []
    heldout: list[str] = []
    shortfall: dict[str, int] = {}

    for source, want in quota.items():
        sections = sorted(by_source[source], key=lambda s: s["ref"])
        rng.shuffle(sections)
        taken = 0
        for sec in sections:
            if taken >= want:
                break
            sentences = []
            for line in sec["lines"]:
                cleaned = clean_line(line)
                for sent in split_sentences(cleaned):
                    if is_fully_vocalized(sent):
                        sentences.append(sent)
            if not sentences:
                continue
            heldout.append(sec["ref"])
            for sent in sentences:
                if taken >= want:
                    break
                layer = sec["layer"]
                if source == "talmud":
                    layer = "aramaic" if is_aramaic(sent) else "mishnaic"
                rows.append(
                    {
                        "id": f"{sec['ref']}#{taken}",
                        "ref": sec["ref"],
                        "source": source,
                        "layer": layer,
                        "license": sec["license"],
                        "text": normalize(sent),
                        "n_words": len(words(sent)),
                    }
                )
                taken += 1
        if taken < want:
            shortfall[source] = want - taken

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval_set.jsonl"
    with eval_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out_dir / "heldout_refs.json").write_text(
        json.dumps(sorted(set(heldout)), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    digest = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    manifest = {
        "sha256": digest,
        "n_sentences": len(rows),
        "n_words": sum(r["n_words"] for r in rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_layer": dict(Counter(r["layer"] for r in rows)),
        "by_license": dict(Counter(r["license"] for r in rows)),
        "heldout_refs": len(set(heldout)),
        "seed": seed,
        "quota": quota,
        "shortfall": shortfall,
        "frozen": True,
        "note": "סט מוקפא. אין להכניס אותו לאימון בשום מצב. "
                "build_corpus.py מסנן את heldout_refs.json.",
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def verify(out_dir: Path = EVAL_DIR) -> bool:
    """מאמת שה-hash תואם. נקרא מ-evaluate.py לפני כל הרצה."""
    manifest_path = out_dir / "MANIFEST.json"
    eval_path = out_dir / "eval_set.jsonl"
    if not (manifest_path.exists() and eval_path.exists()):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return hashlib.sha256(eval_path.read_bytes()).hexdigest() == manifest["sha256"]


def load(out_dir: Path = EVAL_DIR) -> list[dict]:
    path = out_dir / "eval_set.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="בניית סט ההערכה המוקפא")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "sections.jsonl")
    p.add_argument("--out", type=Path, default=EVAL_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--quota", type=json.loads, default=None,
                   help='למשל: \'{"mishnah":800,"talmud":800,"siddur":400}\'')
    p.add_argument("--force", action="store_true",
                   help="דריסת סט הערכה קיים. לא עושים את זה בלי סיבה טובה.")
    args = p.parse_args(argv)

    if (args.out / "eval_set.jsonl").exists() and not args.force:
        print(
            "סט הערכה כבר קיים ומוקפא. לבנייה מחדש צריך --force, "
            "ואז כל המספרים הקודמים בטלים.",
            file=sys.stderr,
        )
        return 1
    manifest = build(args.raw, args.out, quota=args.quota or QUOTA, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

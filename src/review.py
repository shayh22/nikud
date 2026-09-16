"""שלב 6 — לולאת התיקונים.

כל תיקון אנושי הוא דוגמת אימון. הכלי מציג מילים שהמנוע לא היה בטוח בהן
בהקשרן, מקבל הכרעה, וכותב אותה לשני מקומות:

  lexicon/overrides.json    — משפיע על הריצה הבאה מיד
  data/corrections.jsonl    — נכנס לסט האימון הבא

אחרי כמה מאות תיקונים יש סט התאמה אמיתי לסגנון הספציפי, וזה מה שמייצר
את הפער בין מנוע כללי למנוע שמכיר את הטקסטים שלך.

  python src/review.py queue --docx book.docx      # לאסוף מה שלא הוכרע
  python src/review.py run                         # להכריע, אחד־אחד
  python src/review.py stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine  # noqa: E402
from hebrew import is_hebrew, strip_nikud  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEXICON = ROOT / "lexicon"
QUEUE = ROOT / "data" / "review_pending.jsonl"
CORRECTIONS = ROOT / "data" / "corrections.jsonl"


def collect_from_docx(path: Path, engine: Engine, limit: int | None = None) -> int:
    """אוסף לתור כל מילה שהמנוע לא הכריע, עם ההקשר שלה."""
    import docx

    doc = docx.Document(str(path))
    seen: Counter = Counter()
    rows: list[dict] = []
    n = 0
    for idx, para in enumerate(doc.paragraphs):
        text = para.text
        if not text.strip() or not is_hebrew(text):
            continue
        if limit is not None and n >= limit:
            break
        n += 1
        result = engine.vocalize(text, paragraph_id=f"p{idx}")
        words = [d.form for d in result.decisions]
        for i, d in enumerate(result.decisions):
            if d.confident:
                continue
            skel = strip_nikud(d.form)
            seen[skel] += 1
            if seen[skel] > 3:   # שלוש דוגמאות לכל צורה מספיקות להכרעה
                continue
            rows.append(
                {
                    "skeleton": skel,
                    "paragraph": f"p{idx}",
                    "context": " ".join(words[max(0, i - 5) : i + 6]),
                    "position": i,
                    "note": d.note,
                }
            )

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def collect_from_lexicon(limit: int = 200) -> int:
    """אוסף את הצורות הדו־משמעיות התכופות ביותר מהלקסיקון."""
    path = LEXICON / "review_queue.json"
    if not path.exists():
        return 0
    items = json.loads(path.read_text(encoding="utf-8"))[:limit]
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(
                json.dumps(
                    {
                        "skeleton": item["skeleton"],
                        "paragraph": "*",
                        "context": "",
                        "candidates": [v[0] for v in item["variants"]],
                        "count": item["count"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(items)


def _load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    return [json.loads(line) for line in QUEUE.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _save_queue(rows: list[dict]) -> None:
    QUEUE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def record(skeleton: str, form: str, *, paragraph: str = "*", context: str = "") -> None:
    """כותב הכרעה ל-overrides ולסט האימון. השלד נבדק — אין דרך לשבור זהות."""
    if strip_nikud(form) != skeleton:
        raise ValueError(
            f"הצורה '{form}' אינה תואמת לשלד '{skeleton}'. "
            "תיקון לא משנה אותיות, רק ניקוד."
        )
    path = LEXICON / "overrides.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"*": {}}
    data.setdefault(paragraph, {})[skeleton] = form
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    with CORRECTIONS.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"skeleton": skeleton, "form": form, "paragraph": paragraph,
                 "context": context},
                ensure_ascii=False,
            )
            + "\n"
        )


def run_interactive() -> int:
    rows = _load_queue()
    if not rows:
        print("התור ריק. הרץ `review.py queue` קודם.", file=sys.stderr)
        return 1
    remaining: list[dict] = []
    decided = 0
    for i, row in enumerate(rows):
        print(f"\n[{i + 1}/{len(rows)}]  {row['skeleton']}")
        if row.get("context"):
            print(f"  הקשר: {row['context']}")
        if row.get("candidates"):
            for j, c in enumerate(row["candidates"], 1):
                print(f"    {j}) {c}")
        print("  הקלד ניקוד, מספר מהרשימה, s=דילוג, q=יציאה")
        try:
            answer = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            remaining.extend(rows[i:])
            break
        if answer == "q":
            remaining.extend(rows[i:])
            break
        if answer in ("", "s"):
            remaining.append(row)
            continue
        if answer.isdigit() and row.get("candidates"):
            idx = int(answer) - 1
            if not (0 <= idx < len(row["candidates"])):
                remaining.append(row)
                continue
            answer = row["candidates"][idx]
        try:
            record(row["skeleton"], answer, paragraph=row.get("paragraph", "*"),
                   context=row.get("context", ""))
            decided += 1
        except ValueError as exc:
            print(f"  {exc}")
            remaining.append(row)

    _save_queue(remaining)
    print(f"\nהוכרעו {decided}. נשארו {len(remaining)} בתור.", file=sys.stderr)
    return 0


def stats() -> dict:
    corrections = 0
    if CORRECTIONS.exists():
        corrections = sum(1 for line in CORRECTIONS.read_text(encoding="utf-8").splitlines()
                          if line.strip())
    return {
        "pending": len(_load_queue()),
        "corrections_recorded": corrections,
        "overrides": sum(
            len(v) for k, v in json.loads(
                (LEXICON / "overrides.json").read_text(encoding="utf-8")
            ).items() if isinstance(v, dict)
        ) if (LEXICON / "overrides.json").exists() else 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="לולאת התיקונים")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("queue", help="איסוף מילים לבדיקה")
    q.add_argument("--docx", type=Path, default=None)
    q.add_argument("--from-lexicon", type=int, default=0)
    q.add_argument("--engine", default="lexicon+quotes")
    q.add_argument("--max-paragraphs", type=int, default=None)

    sub.add_parser("run", help="הכרעה אינטראקטיבית")
    sub.add_parser("stats", help="מצב התור")

    args = p.parse_args(argv)
    if args.cmd == "queue":
        total = 0
        if args.docx:
            eng = (Engine.load(with_model="dicta-il/dictabert-large-char-menaked")
                   if args.engine == "full" else Engine.load())
            total += collect_from_docx(args.docx, eng, args.max_paragraphs)
        if args.from_lexicon:
            total += collect_from_lexicon(args.from_lexicon)
        print(f"נוספו {total} פריטים לתור.")
        return 0
    if args.cmd == "run":
        return run_interactive()
    print(json.dumps(stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

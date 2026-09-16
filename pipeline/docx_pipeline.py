"""שלב 5 — הצנרת: docx נכנס, docx מנוקד יוצא.

העיצוב נשמר. python-docx מחזיק כל פסקה כרצף של runs (קטעי עיצוב), והניקוד
מוחזר אליהם לפי אורך השלד של כל run. זה אפשרי בדיוק מפני שהקו האדום
מתקיים: הסרת הניקוד מהפלט מחזירה את המקור תו־בתו, ולכן המיפוי חד־חד־ערכי.

בדיקת הזהות היא תנאי חוסם. אם היא נכשלת ולו בפסקה אחת — אין קובץ פלט.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from engine import Engine  # noqa: E402
from hebrew import identity_diff, is_hebrew, normalize, strip_nikud  # noqa: E402


def redistribute(vocalized: str, run_texts: list[str]) -> list[str]:
    """מחלק טקסט מנוקד חזרה ל-runs לפי אורך השלד של כל run."""
    out: list[str] = []
    pos = 0
    for original in run_texts:
        need = len(original)
        taken = 0
        start = pos
        while pos < len(vocalized) and taken < need:
            # תו שאינו ניקוד נספר מול השלד; ניקוד נסחף איתו.
            if not strip_nikud(vocalized[pos]):
                pos += 1
                continue
            taken += 1
            pos += 1
        # סימני ניקוד שנשארו צמודים אחרי התו האחרון שייכים לו.
        while pos < len(vocalized) and not strip_nikud(vocalized[pos]):
            pos += 1
        out.append(vocalized[start:pos])
    if pos < len(vocalized):
        out[-1] += vocalized[pos:]
    return out


def process(in_path: Path, out_path: Path, *, engine: Engine,
            max_paragraphs: int | None = None,
            report_path: Path | None = None) -> dict:
    import docx

    doc = docx.Document(str(in_path))
    paragraphs = doc.paragraphs
    stats: Counter = Counter()
    sources: Counter = Counter()
    unresolved: Counter = Counter()
    failures: list[dict] = []
    samples: list[dict] = []

    processed = 0
    for idx, para in enumerate(paragraphs):
        text = para.text
        if not text.strip() or not is_hebrew(text):
            continue
        if max_paragraphs is not None and processed >= max_paragraphs:
            break
        processed += 1

        result = engine.vocalize(text, paragraph_id=f"p{idx}")
        diff = identity_diff(text, result.text)
        if diff:
            # לא אמור לקרות — vocalize כבר אוכף זאת. כאן זו רשת ביטחון שנייה.
            failures.append({"paragraph": idx, "text": text[:80]})
            continue

        run_texts = [r.text for r in para.runs]
        if run_texts and "".join(run_texts) == text:
            pieces = redistribute(result.text, run_texts)
            if "".join(strip_nikud(p) for p in pieces) != text:
                failures.append({"paragraph": idx, "reason": "חלוקת runs נכשלה"})
                continue
            for run, piece in zip(para.runs, pieces):
                run.text = piece
        elif para.runs:
            # עיצוב מפוצל שאי אפשר למפות בבטחה — הכול נכנס ל-run הראשון.
            para.runs[0].text = result.text
            for run in para.runs[1:]:
                run.text = ""
            stats["paragraphs_flattened"] += 1

        stats["paragraphs"] += 1
        for d in result.decisions:
            sources[d.source] += 1
        for skel in result.unresolved:
            unresolved[skel] += 1
        if len(samples) < 40:
            samples.append({"paragraph": idx, "source": text, "vocalized": result.text})

    if failures:
        raise SystemExit(
            f"בדיקת הזהות נכשלה ב-{len(failures)} פסקאות. לא נכתב קובץ פלט.\n"
            + json.dumps(failures[:5], ensure_ascii=False, indent=2)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))

    total_words = sum(sources.values())
    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "paragraphs": stats["paragraphs"],
        "paragraphs_flattened": stats["paragraphs_flattened"],
        "words": total_words,
        "by_layer": dict(sources),
        "coverage": {
            k: round(v / total_words, 4) for k, v in sorted(sources.items())
        } if total_words else {},
        "unresolved_words": sum(unresolved.values()),
        "unresolved_rate": round(sum(unresolved.values()) / total_words, 4)
        if total_words else 0.0,
        "top_unresolved": unresolved.most_common(60),
        "identity_failures": 0,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"summary": summary, "samples": samples},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ניקוד קובץ docx")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--engine", default="lexicon+quotes",
                   help="lexicon | lexicon+quotes | full | trained:<path>")
    p.add_argument("--max-paragraphs", type=int, default=None)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    if args.engine == "full":
        eng = Engine.load(with_model="dicta-il/dictabert-large-char-menaked")
    elif args.engine.startswith("trained:"):
        eng = Engine.load(with_model=args.engine.split(":", 1)[1])
    elif args.engine == "lexicon":
        eng = Engine.load(with_quotes=False)
    else:
        eng = Engine.load()

    summary = process(args.input, args.output, engine=eng,
                      max_paragraphs=args.max_paragraphs, report_path=args.report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""מדידת שכבת הכתיב החסר.

הבעיה במדידה: לספר אין טקסט זהב, ולסט ההערכה אין גרסה בכתיב מלא. הפתרון
הוא לייצר אותה — לקחת את הזהב המסורתי, להוסיף לו אמות קריאה, ולבדוק כמה
מהמקור השכבה מחזירה.

    זהב מסורתי        חִבּוּר
    → כתיב מלא         חִיבּוּר      (to_male)
    → שלד              חיבור
    → המנוע במצב haser  ?

מילה נספרת כמוצלחת רק אם הצורה חזרה **בדיוק** לזהב — גם האותיות וגם
הניקוד. המדידה מפרידה בין המילים שבהן הוספה אם קריאה (אלה שבמחלוקת)
לבין כל השאר, כי רק הראשונות מודדות את השכבה.

    python src/evaluate_ktiv.py --limit 500 --engine full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_eval  # noqa: E402
from hebrew import normalize, strip_nikud, word_matches, words  # noqa: E402
from ktiv import to_male  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def make_male(gold: str) -> tuple[str, list[bool]]:
    """(משפט בכתיב מלא, האם לכל מילה נוספה אם קריאה)."""
    out, changed = [], []
    for w in words(normalize(gold)):
        male = to_male(w)
        out.append(male)
        changed.append(strip_nikud(male) != strip_nikud(w))
    return " ".join(out), changed


def run(engine_spec: str, limit: int | None) -> dict:
    import engine as engine_mod

    rows = build_eval.load()
    if limit:
        rows = rows[:limit]

    engines = {
        mode: engine_mod.build_engine_ktiv(engine_spec, mode)
        for mode in ("male", "haser")
    }

    male_texts: list[str] = []
    gold_words: list[list[str]] = []
    changed_flags: list[list[bool]] = []
    for r in rows:
        gold = normalize(r["text"])
        male, changed = make_male(gold)
        male_texts.append(strip_nikud(male))
        gold_words.append(words(gold))
        changed_flags.append(changed)

    stats: dict[str, Counter] = {m: Counter() for m in engines}
    samples: list[dict] = []
    for mode, fn in engines.items():
        preds = fn.batch(male_texts)
        for gws, flags, pred in zip(gold_words, changed_flags, preds):
            pws = words(normalize(pred))
            if len(pws) != len(gws):
                stats[mode]["misaligned"] += 1
                continue
            for gw, pw, was_changed in zip(gws, pws, flags):
                bucket = "mater" if was_changed else "plain"
                stats[mode][f"{bucket}_total"] += 1
                if word_matches(gw, pw):
                    stats[mode][f"{bucket}_exact"] += 1
                elif strip_nikud(gw) == strip_nikud(pw):
                    # השלד חזר נכון, רק הניקוד שגוי — לא כישלון של השכבה.
                    stats[mode][f"{bucket}_right_spelling"] += 1
                if mode == "haser" and was_changed and not word_matches(gw, pw) \
                        and len(samples) < 60:
                    samples.append({"gold": gw, "pred": pw,
                                    "male": to_male(gw)})

    def summarize(c: Counter) -> dict:
        out = {}
        for bucket in ("mater", "plain"):
            total = c[f"{bucket}_total"]
            out[bucket] = {
                "words": total,
                "exact": c[f"{bucket}_exact"],
                "exact_rate": round(c[f"{bucket}_exact"] / total, 4) if total else 0.0,
                "right_spelling_wrong_nikud": c[f"{bucket}_right_spelling"],
                "spelling_rate": round(
                    (c[f"{bucket}_exact"] + c[f"{bucket}_right_spelling"]) / total, 4
                ) if total else 0.0,
            }
        out["misaligned_sentences"] = c["misaligned"]
        return out

    return {
        "engine": engine_spec,
        "sentences": len(rows),
        "modes": {m: summarize(c) for m, c in stats.items()},
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="מדידת שכבת הכתיב החסר")
    p.add_argument("--engine", default="lexicon+quotes")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=REPORTS / "ktiv.json")
    args = p.parse_args(argv)

    if not build_eval.verify():
        print("סט ההערכה חסר או שה-hash לא תואם.", file=sys.stderr)
        return 2
    result = run(args.engine, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

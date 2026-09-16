"""שלב 1 — מנוע ההערכה.

מקבל פונקציית ניקוד ומחזיר מספרים. זה הכלי שכל שלב אחריו נמדד מולו.
בלי מספר — שינוי לא נכנס.

מדדים:
  WER               אחוז המילים שבהן לפחות סימן אחד שגוי
  CER               אחוז הסלוטים (אות + סימניה) השגויים
  WER ללא שי"ן      אותו WER כשמתעלמים מנקודת שי"ן/שמאלית — מבודד את הבעיה
  identity_failures מספר המשפטים שבהם הפלט לא היה זהה למקור. כל אחד כזה
                    נספר כשגיאה מלאה, ובנוסף נחשב כשל חוסם.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_eval  # noqa: E402
from classify import construct_positions, homograph_positions, name_positions  # noqa: E402
from hebrew import (  # noqa: E402
    char_errors,
    identity_diff,
    is_acronym,
    normalize,
    strip_nikud,
    word_matches,
    words,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# הקטגוריות לפילוח. משפט/מילה יכולים להשתייך ליותר מאחת.
SEGMENTS = ("hebrew", "aramaic", "names", "construct", "homographs", "acronyms")


@dataclass
class Bucket:
    words: int = 0
    word_errors: int = 0
    word_errors_no_shin: int = 0
    slots: int = 0
    slot_errors: int = 0

    @property
    def wer(self) -> float:
        return self.word_errors / self.words if self.words else 0.0

    @property
    def wer_no_shin(self) -> float:
        return self.word_errors_no_shin / self.words if self.words else 0.0

    @property
    def cer(self) -> float:
        return self.slot_errors / self.slots if self.slots else 0.0


@dataclass
class Result:
    name: str
    overall: Bucket = field(default_factory=Bucket)
    segments: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    identity_failures: list[dict] = field(default_factory=list)
    sentences: int = 0
    seconds: float = 0.0
    errors_sample: list[dict] = field(default_factory=list)
    top_confusions: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sentences": self.sentences,
            "words": self.overall.words,
            "wer": round(self.overall.wer, 4),
            "wer_no_shin": round(self.overall.wer_no_shin, 4),
            "cer": round(self.overall.cer, 4),
            "identity_failures": len(self.identity_failures),
            "seconds": round(self.seconds, 1),
            "segments": {
                k: {
                    "words": b.words,
                    "wer": round(b.wer, 4),
                    "wer_no_shin": round(b.wer_no_shin, 4),
                    "cer": round(b.cer, 4),
                }
                for k, b in sorted(self.segments.items())
            },
            "top_confusions": [
                {"gold": g, "pred": p, "count": c}
                for (g, p), c in self.top_confusions.most_common(25)
            ],
            "errors_sample": self.errors_sample[:40],
        }


def _align_predicted_words(source: str, predicted: str) -> list[str] | None:
    """מיישר את מילות הפלט למילות הקלט. None אם השלד לא תואם."""
    if identity_diff(source, predicted):
        return None
    return words(normalize(predicted))


def evaluate(
    nikud_fn: Callable[[str], str],
    dataset: Iterable[dict],
    *,
    name: str = "engine",
    homograph_skeletons: set[str] | None = None,
    extra_names: set[str] | None = None,
    collect_errors: bool = True,
) -> Result:
    res = Result(name=name)
    homos = homograph_skeletons or set()
    rows = list(dataset)
    t0 = time.monotonic()

    # נתיב אצווה כשיש — המודל הוא החלק היקר, והוא מרוויח ממנו.
    batch_fn = getattr(nikud_fn, "batch", None)
    preds: list[str | None]
    if batch_fn is not None:
        sources = [strip_nikud(normalize(r["text"])) for r in rows]
        try:
            preds = list(batch_fn(sources))
        except Exception as exc:  # noqa: BLE001
            res.identity_failures.append({"id": "<batch>", "error": repr(exc)})
            preds = [None] * len(rows)
    else:
        preds = []
        for r in rows:
            try:
                preds.append(nikud_fn(strip_nikud(normalize(r["text"]))))
            except Exception as exc:  # noqa: BLE001
                res.identity_failures.append({"id": r.get("id"), "error": repr(exc)})
                preds.append(None)

    for row, pred in zip(rows, preds):
        gold = normalize(row["text"])
        source = strip_nikud(gold)
        res.sentences += 1

        gold_words = words(gold)
        pred_words = _align_predicted_words(source, pred) if pred is not None else None

        if pred_words is None or len(pred_words) != len(gold_words):
            if pred is not None:
                res.identity_failures.append(
                    {
                        "id": row.get("id"),
                        "ref": row.get("ref"),
                        "reason": "שלד הפלט אינו זהה למקור",
                    }
                )
            # כל המילים נספרות כשגויות.
            for w in gold_words:
                res.overall.words += 1
                res.overall.word_errors += 1
                res.overall.word_errors_no_shin += 1
                slots = len([c for c in strip_nikud(w)])
                res.overall.slots += slots
                res.overall.slot_errors += slots
            continue

        is_aram = row.get("layer") == "aramaic"
        name_idx = name_positions(gold, extra_names)
        constr_idx = construct_positions(gold)
        homo_idx = homograph_positions(gold, homos)

        for i, (gw, pw) in enumerate(zip(gold_words, pred_words)):
            ok = word_matches(gw, pw)
            ok_no_shin = word_matches(gw, pw, ignore_shin=True)
            errs, total = char_errors(gw, pw)

            buckets = [res.overall]
            buckets.append(res.segments["aramaic" if is_aram else "hebrew"])
            if i in name_idx:
                buckets.append(res.segments["names"])
            if i in constr_idx:
                buckets.append(res.segments["construct"])
            if i in homo_idx:
                buckets.append(res.segments["homographs"])
            if is_acronym(gw):
                buckets.append(res.segments["acronyms"])

            for b in buckets:
                b.words += 1
                b.slots += total
                b.slot_errors += errs
                if not ok:
                    b.word_errors += 1
                if not ok_no_shin:
                    b.word_errors_no_shin += 1

            if not ok and collect_errors:
                res.top_confusions[(gw, pw)] += 1
                if len(res.errors_sample) < 200:
                    res.errors_sample.append(
                        {
                            "ref": row.get("ref"),
                            "gold": gw,
                            "pred": pw,
                            "context": " ".join(
                                gold_words[max(0, i - 3) : i + 4]
                            ),
                        }
                    )

    res.seconds = time.monotonic() - t0
    return res


# --- דוחות --------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render_report(results: list[Result], *, title: str, manifest: dict,
                  baseline: dict | None = None) -> str:
    lines = [f"# {title}", ""]
    lines += [
        f"סט ההערכה: {manifest['n_sentences']} משפטים · "
        f"{manifest['n_words']} מילים · sha256 `{manifest['sha256'][:16]}…`",
        "",
        "פילוח הסט: "
        + " · ".join(f"{k}={v}" for k, v in sorted(manifest["by_layer"].items())),
        "",
        "## תוצאות",
        "",
        "| מנוע | WER | WER ללא שי\"ן | CER | כשלי זהות | שניות |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        flag = "" if not r.identity_failures else f" ⛔ {len(r.identity_failures)}"
        lines.append(
            f"| {r.name} | {_pct(r.overall.wer)} | {_pct(r.overall.wer_no_shin)} | "
            f"{_pct(r.overall.cer)} | {len(r.identity_failures)}{flag} | "
            f"{r.seconds:.0f} |"
        )

    lines += ["", "## פילוח", "", "| מנוע | " + " | ".join(SEGMENTS) + " |",
              "|---|" + "---|" * len(SEGMENTS)]
    for r in results:
        cells = []
        for seg in SEGMENTS:
            b = r.segments.get(seg)
            cells.append(f"{_pct(b.wer)} ({b.words})" if b and b.words else "—")
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")

    if baseline and baseline.get("wer") is not None:
        base_wer = baseline["wer"]
        base_name = baseline.get("name", "baseline")
        lines += [
            "",
            f"## מול הבסיס — `{base_name}` ב-{_pct(base_wer)}",
            "",
            "| מנוע | WER | שינוי |",
            "|---|---|---|",
        ]
        for r in results:
            delta = r.overall.wer - base_wer
            if abs(delta) < 1e-9:
                cell = "—"
            else:
                cell = ("↓ " if delta < 0 else "↑ ") + _pct(abs(delta))
            lines.append(f"| {r.name} | {_pct(r.overall.wer)} | {cell} |")

    for r in results:
        if not r.top_confusions:
            continue
        lines += ["", f"### שגיאות נפוצות — {r.name}", "",
                  "| זהב | פלט | מופעים |", "|---|---|---|"]
        for (g, p), c in r.top_confusions.most_common(15):
            lines.append(f"| {g} | {p} | {c} |")

    lines += ["", "---", "",
              "נוצר על ידי `src/evaluate.py`. כל שינוי במנוע נמדד מול הטבלה הזו."]
    return "\n".join(lines) + "\n"


def load_homograph_skeletons(lexicon_dir: Path = ROOT / "lexicon") -> set[str]:
    path = lexicon_dir / "forms.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        skel
        for skel, entry in data.get("forms", {}).items()
        if entry.get("category") != "unambiguous"
    }


def load_names(lexicon_dir: Path = ROOT / "lexicon") -> set[str]:
    path = lexicon_dir / "names.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="הערכת מנוע ניקוד")
    p.add_argument("--engines", nargs="+", default=["lexicon"],
                   help="null | lexicon | dictabert | full | trained:<path>")
    p.add_argument("--limit", type=int, default=None, help="תקרת משפטים — לבדיקה")
    p.add_argument("--out", type=Path, default=None, help="קובץ md לכתיבה")
    p.add_argument("--title", default="דוח הערכה")
    p.add_argument("--baseline", type=Path, default=REPORTS / "baseline.json")
    p.add_argument("--lexicon-dir", type=Path, default=ROOT / "lexicon")
    p.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval")
    args = p.parse_args(argv)

    if not build_eval.verify(args.eval_dir):
        print("סט ההערכה חסר או שה-hash לא תואם. הרץ src/build_eval.py.", file=sys.stderr)
        return 2
    manifest = json.loads((args.eval_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    dataset = build_eval.load(args.eval_dir)
    if args.limit:
        dataset = dataset[: args.limit]

    import engine as engine_mod  # ייבוא מאוחר — engine טוען משקולות

    homos = load_homograph_skeletons(args.lexicon_dir)
    names = load_names(args.lexicon_dir)
    results = []
    for spec in args.engines:
        fn = engine_mod.build_engine(spec, args.lexicon_dir)
        print(f"מריץ {spec} על {len(dataset)} משפטים…", file=sys.stderr)
        results.append(
            evaluate(fn, dataset, name=spec, homograph_skeletons=homos, extra_names=names)
        )

    # ההשוואה היא מול המנוע הטוב ביותר בבסיס, לא מול הראשון ברשימה —
    # אחרת כל שינוי נראה מצוין מול מנוע ריק.
    baseline = None
    if args.baseline and args.baseline.exists() and args.baseline != args.out:
        stored = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidates = [r for r in stored.get("results", []) if r.get("wer") is not None]
        if candidates:
            baseline = min(candidates, key=lambda r: r["wer"])

    report = render_report(results, title=args.title, manifest=manifest, baseline=baseline)
    out = args.out or (REPORTS / "evaluation.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {"manifest": manifest, "results": [r.to_dict() for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"נכתב ל-{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""שלב 3 — שכבת הלקסיקון.

מפרק את הקורפוס לטוקנים, סופר `צורה → ניקוד`, ומסווג לשלוש קטגוריות:

  unambiguous  ניקוד אחד ביותר מ-98% מהמופעים → החלה אוטומטית
  context      תלוית הקשר → נפתרת בביגרמים
  ambiguous    אמיתית־דו־משמעית → נכנסת ל-review_queue. לא מנחשים.

התוצרים:
  lexicon/forms.json         שלד → ניקוד + שכיחות + קטגוריה
  lexicon/bigrams.json       (מילה קודמת, שלד) → ניקוד
  lexicon/names.json         שמות פרטיים שנלקטו מהקורפוס
  lexicon/review_queue.json  מה שצריך הכרעה אנושית
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import load_corpus  # noqa: E402
from classify import NAME_TRIGGERS  # noqa: E402
from hebrew import canonical, is_acronym, strip_nikud, words  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEXICON = ROOT / "lexicon"
CORPUS_DIR = ROOT / "data" / "corpus"

UNAMBIGUOUS_RATIO = 0.98
UNAMBIGUOUS_MIN_COUNT = 3
BIGRAM_MIN_COUNT = 2
BIGRAM_MIN_RATIO = 0.80
MAX_VARIANTS_STORED = 6


def build(corpus_dir: Path, out_dir: Path, *, min_count: int = 2,
          layers: list[str] | None = None) -> dict:
    # שלד → ניקוד → ספירה
    forms: dict[str, Counter] = defaultdict(Counter)
    # (שלד קודם, שלד) → ניקוד → ספירה
    bigrams: dict[tuple[str, str], Counter] = defaultdict(Counter)
    name_counts: Counter = Counter()
    layer_forms: dict[str, Counter] = defaultdict(Counter)

    sentences = 0
    tokens = 0
    for row in load_corpus(corpus_dir, layers):
        sentences += 1
        ws = words(row["text"])
        prev_skel = "<s>"
        for w in ws:
            form = canonical(w)
            skel = strip_nikud(form)
            if not skel or is_acronym(w):
                prev_skel = skel or prev_skel
                continue
            tokens += 1
            forms[skel][form] += 1
            layer_forms[row["layer"]][skel] += 1
            bigrams[(prev_skel, skel)][form] += 1
            if prev_skel in NAME_TRIGGERS:
                name_counts[skel] += 1
            prev_skel = skel

    # --- סיווג ---------------------------------------------------------
    # מעבר אחד מראש: אילו שלדים יש להם ולו ביגרם אחד שמכריע אותם.
    resolvable_by_bigram: set[str] = set()
    for (_prev, skel), variants in bigrams.items():
        if skel not in resolvable_by_bigram and _bigram_decides(variants):
            resolvable_by_bigram.add(skel)

    entries: dict[str, dict] = {}
    counts = Counter()
    review: list[dict] = []

    for skel, variants in forms.items():
        total = sum(variants.values())
        if total < min_count:
            continue
        best_form, best_n = variants.most_common(1)[0]
        ratio = best_n / total

        if ratio >= UNAMBIGUOUS_RATIO and total >= UNAMBIGUOUS_MIN_COUNT:
            category = "unambiguous"
        else:
            # יש ביגרם שמכריע את השלד הזה? אז הוא תלוי הקשר, לא דו־משמעי.
            category = "context" if skel in resolvable_by_bigram else "ambiguous"

        counts[category] += 1
        entries[skel] = {
            "category": category,
            "best": best_form,
            "count": total,
            "ratio": round(ratio, 4),
            "variants": [[f, n] for f, n in variants.most_common(MAX_VARIANTS_STORED)],
        }
        if category == "ambiguous":
            review.append(
                {
                    "skeleton": skel,
                    "count": total,
                    "variants": [[f, n] for f, n in variants.most_common(MAX_VARIANTS_STORED)],
                }
            )

    # --- ביגרמים -------------------------------------------------------
    # שומרים רק ביגרמים ששלד היעד שלהם אינו חד־משמעי — לשאר אין תועלת.
    bigram_out: dict[str, str] = {}
    for (prev, skel), variants in bigrams.items():
        entry = entries.get(skel)
        if entry is None or entry["category"] == "unambiguous":
            continue
        decided = _bigram_decides(variants)
        if decided and decided != entry["best"]:
            bigram_out[f"{prev}\t{skel}"] = decided
        elif decided:
            # גם כשהביגרם מסכים עם הצורה השכיחה, שומרים — הוא מאשש בהקשר.
            bigram_out[f"{prev}\t{skel}"] = decided

    # --- שמות ----------------------------------------------------------
    names = sorted(s for s, n in name_counts.items() if n >= 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "sentences": sentences,
        "tokens": tokens,
        "distinct_skeletons": len(forms),
        "stored_skeletons": len(entries),
        "categories": dict(counts),
        "bigrams": len(bigram_out),
        "names": len(names),
        "thresholds": {
            "unambiguous_ratio": UNAMBIGUOUS_RATIO,
            "unambiguous_min_count": UNAMBIGUOUS_MIN_COUNT,
            "bigram_min_count": BIGRAM_MIN_COUNT,
            "bigram_min_ratio": BIGRAM_MIN_RATIO,
            "min_count": min_count,
        },
        "layers": {k: len(v) for k, v in sorted(layer_forms.items())},
    }
    (out_dir / "forms.json").write_text(
        json.dumps({"meta": meta, "forms": entries}, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "bigrams.json").write_text(
        json.dumps(bigram_out, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    review.sort(key=lambda r: -r["count"])
    (out_dir / "review_queue.json").write_text(
        json.dumps(review[:5000], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def _bigram_decides(variants: Counter) -> str | None:
    total = sum(variants.values())
    if total < BIGRAM_MIN_COUNT:
        return None
    form, n = variants.most_common(1)[0]
    return form if n / total >= BIGRAM_MIN_RATIO else None


# --- טעינה לזמן ריצה ----------------------------------------------------


class Lexicon:
    """הלקסיקון כפי שהמנוע משתמש בו."""

    def __init__(self, forms: dict, bigrams: dict[str, str], names: set[str]):
        self.meta = forms.get("meta", {})
        self.forms: dict[str, dict] = forms.get("forms", {})
        self.bigrams = bigrams
        self.names = names

    @classmethod
    def load(cls, path: Path = LEXICON) -> "Lexicon":
        forms_path = path / "forms.json"
        if not forms_path.exists():
            return cls({}, {}, set())
        forms = json.loads(forms_path.read_text(encoding="utf-8"))
        bigrams_path = path / "bigrams.json"
        bigrams = (
            json.loads(bigrams_path.read_text(encoding="utf-8"))
            if bigrams_path.exists()
            else {}
        )
        names_path = path / "names.json"
        names = (
            set(json.loads(names_path.read_text(encoding="utf-8")))
            if names_path.exists()
            else set()
        )
        return cls(forms, bigrams, names)

    def unambiguous(self, skeleton: str) -> str | None:
        e = self.forms.get(skeleton)
        if e and e["category"] == "unambiguous":
            return e["best"]
        return None

    def by_bigram(self, prev_skeleton: str, skeleton: str) -> str | None:
        return self.bigrams.get(f"{prev_skeleton}\t{skeleton}")

    def category(self, skeleton: str) -> str | None:
        e = self.forms.get(skeleton)
        return e["category"] if e else None

    def most_common(self, skeleton: str) -> str | None:
        e = self.forms.get(skeleton)
        return e["best"] if e else None

    def __len__(self) -> int:
        return len(self.forms)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="בניית שכבת הלקסיקון")
    p.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    p.add_argument("--out", type=Path, default=LEXICON)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--layers", nargs="*", default=None)
    args = p.parse_args(argv)
    meta = build(args.corpus, args.out, min_count=args.min_count, layers=args.layers)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

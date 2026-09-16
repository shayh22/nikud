"""שלב 3.4 — התאמת ציטוטים.

מזהה פסוקים ומאמרי חז"ל בטקסט ומעתיק את הניקוד מהמקור המנוקד מילה במילה,
במקום לנקד מחדש. זה בדיוק הכלל מהנוהל — ציטוט נלקח מהמקור — רק אוטומטי.

השיטה: אינדקס n-gram של שלדי העיצורים מעל הקורפוס הצטוט (תנ"ך, משנה,
סידור). בזמן ריצה מחפשים חלון של n מילים, ואם נמצא — מרחיבים את ההתאמה
קדימה כל עוד היא נמשכת, ומעתיקים את הניקוד.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import load_corpus  # noqa: E402
from hebrew import canonical, normalize, strip_nikud, words  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEXICON = ROOT / "lexicon"
CORPUS_DIR = ROOT / "data" / "corpus"

NGRAM = 4          # אורך החלון לחיפוש
MIN_MATCH = 4      # אורך מינימלי להתאמה מקובלת. קצר מזה = רעש.

# מחיצה בין מקורות. חוסמת התאמה שחוצה גבול של פסוק או משנה.
BREAK = "||"

# שכבות שמהן מצטטים בפועל. התלמוד נכנס רק אם מבקשים במפורש (CC-BY-NC).
DEFAULT_LAYERS = ("biblical", "mishnaic", "liturgy")


class QuoteIndex:
    def __init__(self, tokens: list[str], index: dict[str, int],
                 spans: list[list], meta: dict | None = None):
        self.tokens = tokens                       # מילים מנוקדות, רצף אחד
        self.index = index                         # מפתח n-gram → מיקום ב-tokens
        self.spans = spans                         # [start, end, ref] לשיוך
        self.meta = meta or {}
        self._skeletons = [strip_nikud(t) for t in tokens]

    # --- בנייה ---------------------------------------------------------

    @classmethod
    def build(cls, corpus_dir: Path = CORPUS_DIR,
              layers: tuple[str, ...] = DEFAULT_LAYERS,
              *, keep_cantillation: bool = False) -> "QuoteIndex":
        tokens: list[str] = []
        spans: list[list] = []
        for row in load_corpus(corpus_dir, list(layers)):
            ws = [normalize(w) if keep_cantillation else canonical(w)
                  for w in words(row["text"])]
            if len(ws) < MIN_MATCH:
                continue
            start = len(tokens)
            tokens.extend(ws)
            spans.append([start, len(tokens), row["ref"]])
            tokens.append(BREAK)

        skeletons = [strip_nikud(t) for t in tokens]
        index: dict[str, int] = {}
        collisions: set[str] = set()
        for i in range(len(tokens) - NGRAM + 1):
            window = skeletons[i : i + NGRAM]
            if any(w == BREAK or not w for w in window):
                continue
            key = " ".join(window)
            prev = index.get(key)
            if prev is not None:
                # שני מופעים שמנוקדים אחרת — הצירוף אינו מכריע, מוציאים אותו.
                if tokens[i : i + NGRAM] != tokens[prev : prev + NGRAM]:
                    collisions.add(key)
                continue
            index[key] = i
        for key in collisions:
            index.pop(key, None)

        meta = {
            "tokens": len(tokens),
            "ngrams": len(index),
            "dropped_ambiguous": len(collisions),
            "sources": len(spans),
            "layers": list(layers),
            "ngram": NGRAM,
            "min_match": MIN_MATCH,
            "keep_cantillation": keep_cantillation,
        }
        return cls(tokens, index, spans, meta)

    # --- שמירה וטעינה ---------------------------------------------------

    def save(self, path: Path = LEXICON / "quotes.json") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"meta": self.meta, "tokens": self.tokens,
                 "index": self.index, "spans": self.spans},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path = LEXICON / "quotes.json") -> "QuoteIndex | None":
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["tokens"], d["index"], d["spans"], d.get("meta"))

    # --- חיפוש ---------------------------------------------------------

    def _ref_at(self, pos: int) -> str:
        lo, hi = 0, len(self.spans) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end, ref = self.spans[mid]
            if pos < start:
                hi = mid - 1
            elif pos >= end:
                lo = mid + 1
            else:
                return ref
        return ""

    def find_matches(
        self, word_skeletons: list[str]
    ) -> list[tuple[int, int, list[str], str]]:
        """מחזיר [(start, end, מילים מנוקדות, ref)] — קטעים לא חופפים."""
        out: list[tuple[int, int, list[str], str]] = []
        i = 0
        n = len(word_skeletons)
        while i + NGRAM <= n:
            key = " ".join(word_skeletons[i : i + NGRAM])
            pos = self.index.get(key)
            if pos is None:
                i += 1
                continue
            # הרחבה קדימה כל עוד השלדים ממשיכים להתאים.
            length = NGRAM
            while (
                i + length < n
                and pos + length < len(self.tokens)
                and self.tokens[pos + length] != BREAK
                and self._skeletons[pos + length] == word_skeletons[i + length]
            ):
                length += 1
            out.append(
                (i, i + length, self.tokens[pos : pos + length], self._ref_at(pos))
            )
            i += length
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="בניית אינדקס הציטוטים")
    p.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    p.add_argument("--out", type=Path, default=LEXICON / "quotes.json")
    p.add_argument("--layers", nargs="*", default=list(DEFAULT_LAYERS))
    p.add_argument("--keep-cantillation", action="store_true",
                   help='שמירת טעמי המקרא בציטוטי תנ"ך')
    args = p.parse_args(argv)
    idx = QuoteIndex.build(args.corpus, tuple(args.layers),
                           keep_cantillation=args.keep_cantillation)
    idx.save(args.out)
    print(json.dumps(idx.meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

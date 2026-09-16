"""שלב 5 — האינטגרציה.

API אחד: `nikud(text) -> text`. הצנרת הקיימת קוראת לו במקום ל-Nakdimon,
וכל השאר נשאר בדיוק כפי שהוא.

סדר השכבות, מלמעלה למטה (מי שמכריע ראשון מנצח):

    חריגים לפי פסקה   overrides.json
      → כללי הומוגרפים  homographs.json
        → ציטוטים        quotes.json — ניקוד מועתק מהמקור
          → לקסיקון חד־משמעי
            → ביגרמים מהקורפוס
              → המודל
                → כלום — המילה נשארת בלי ניקוד ומסומנת לבדיקה

הקו האדום: אחרי הסרת כל הניקוד הפלט חייב להיות זהה למקור תו־בתו.
`Engine.vocalize` לא מחזיר טקסט שלא עבר את `assert_identity`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lexicon import Lexicon  # noqa: E402
from hebrew import (  # noqa: E402
    assert_identity,
    has_nikud,
    is_acronym,
    normalize,
    split_tokens,
    strip_nikud,
)
from quotes import QuoteIndex  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEXICON = ROOT / "lexicon"
_MODEL_ID = "dicta-il/dictabert-large-char-menaked"

# מקור ההכרעה לכל מילה, לפי סדר עדיפות יורד.
SOURCES = (
    "override",
    "homograph",
    "quote",
    "lexicon",
    "bigram",
    "model",
    "preexisting",
    "acronym",
    "unresolved",
)


@dataclass
class Decision:
    form: str            # הצורה המנוקדת
    source: str          # מאיזו שכבה
    confident: bool = True
    note: str = ""


@dataclass
class VocalizeResult:
    text: str
    decisions: list[Decision] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            out[d.source] = out.get(d.source, 0) + 1
        return out

    @property
    def review_words(self) -> list[str]:
        """מילים שהמנוע לא היה בטוח בהן. הפלט של שלב 6."""
        return [d.form for d in self.decisions if not d.confident]


# --- שכבת המודל ---------------------------------------------------------


class ModelBackend(Protocol):
    name: str

    def vocalize(self, text: str) -> str | None:
        """מקבל טקסט ללא ניקוד ומחזיר מנוקד, או None אם אין לו תשובה."""


class NullBackend:
    """אין מודל. המנוע עובד על לקסיקון וכללים בלבד.

    זה לא מצב שגיאה — שלבים 1–3 של התוכנית אמורים לרוץ כך.
    """

    name = "null"

    def vocalize(self, text: str) -> str | None:
        return None


# DictaBertBackend חי ב-dictabert.py — הקידוד ברמת התו נחוץ גם לאימון.
def DictaBertBackend(*args, **kwargs):  # noqa: N802
    from dictabert import DictaBertBackend as _B

    return _B(*args, **kwargs)


# --- המנוע --------------------------------------------------------------


class Engine:
    def __init__(
        self,
        *,
        lexicon: Lexicon | None = None,
        quotes: QuoteIndex | None = None,
        model: ModelBackend | None = None,
        homographs: dict | None = None,
        overrides: dict | None = None,
        keep_existing_nikud: bool = True,
    ):
        self.lexicon = lexicon or Lexicon({}, {}, set())
        self.quotes = quotes
        self.model = model or NullBackend()
        self.homographs = homographs or {}
        self.overrides = overrides or {}
        self.keep_existing_nikud = keep_existing_nikud

    # --- בנייה ---------------------------------------------------------

    @classmethod
    def load(cls, *, with_model: str | None = None, lexicon_dir: Path = LEXICON,
             with_quotes: bool = True, **kwargs) -> "Engine":
        lexicon = Lexicon.load(lexicon_dir)
        quotes = QuoteIndex.load(lexicon_dir / "quotes.json") if with_quotes else None
        homographs = _load_json(lexicon_dir / "homographs.json", {})
        overrides = _load_json(lexicon_dir / "overrides.json", {})

        model: ModelBackend = NullBackend()
        if with_model:
            model = DictaBertBackend(with_model)
        return cls(
            lexicon=lexicon,
            quotes=quotes,
            model=model,
            homographs=homographs,
            overrides=overrides,
            **kwargs,
        )

    # --- ה-API ---------------------------------------------------------

    def nikud(self, text: str) -> str:
        return self.vocalize(text).text

    def nikud_batch(self, texts: list[str]) -> list[str]:
        """ניקוד אצווה. השכבה היחידה שמרוויחה מזה היא המודל, אבל היא הכבדה."""
        preds: list[str | None] = [None] * len(texts)
        if not isinstance(self.model, NullBackend):
            batch = getattr(self.model, "vocalize_batch", None)
            if batch is not None:
                preds = batch([strip_nikud(normalize(t)) for t in texts])
            else:
                preds = [self.model.vocalize(strip_nikud(normalize(t))) for t in texts]
        return [
            self.vocalize(t, model_prediction=p).text for t, p in zip(texts, preds)
        ]

    def vocalize(self, text: str, *, paragraph_id: str | None = None,
                 model_prediction: str | None = None) -> VocalizeResult:
        source = normalize(text)
        parts = split_tokens(source)
        word_idx = [i for i, (_t, is_word) in enumerate(parts) if is_word]
        originals = [parts[i][0] for i in word_idx]
        skeletons = [strip_nikud(w) for w in originals]

        decisions: list[Decision] = [
            Decision(form=w, source="unresolved", confident=False) for w in originals
        ]

        # 0. מה שכבר מנוקד, וראשי תיבות — לא נוגעים.
        frozen = set()
        for i, w in enumerate(originals):
            if is_acronym(w):
                decisions[i] = Decision(w, "acronym", True, "ראשי תיבות לא מנוקדים")
                frozen.add(i)
            elif self.keep_existing_nikud and has_nikud(w):
                decisions[i] = Decision(w, "preexisting", True, "היה מנוקד במקור")
                frozen.add(i)

        # 6. המודל — השכבה התחתונה. רץ על כל המשפט בבת אחת.
        self._apply_model(source, skeletons, decisions, frozen, model_prediction)

        # 5. ביגרמים מהקורפוס.
        prev = "<s>"
        for i, skel in enumerate(skeletons):
            if i not in frozen:
                form = self.lexicon.by_bigram(prev, skel)
                if form:
                    decisions[i] = Decision(form, "bigram")
            prev = skel

        # 4. לקסיקון חד־משמעי.
        for i, skel in enumerate(skeletons):
            if i in frozen:
                continue
            form = self.lexicon.unambiguous(skel)
            if form:
                decisions[i] = Decision(form, "lexicon")

        # 3. ציטוטים — ניקוד מועתק מהמקור, מילה במילה.
        if self.quotes is not None:
            for start, end, forms, ref in self.quotes.find_matches(skeletons):
                for offset, form in enumerate(forms):
                    i = start + offset
                    if i in frozen:
                        continue
                    decisions[i] = Decision(form, "quote", True, ref)

        # 2. כללי הומוגרפים שנכתבו ביד.
        for i, skel in enumerate(skeletons):
            if i in frozen:
                continue
            rule = self.homographs.get(skel)
            form = _resolve_homograph(rule, skeletons, i)
            if form:
                decisions[i] = Decision(form, "homograph")

        # 1. חריגים לפי פסקה — הסמכות העליונה.
        para_overrides = {}
        if paragraph_id and paragraph_id in self.overrides:
            para_overrides = self.overrides[paragraph_id]
        global_overrides = self.overrides.get("*", {})
        for i, skel in enumerate(skeletons):
            form = para_overrides.get(skel) or global_overrides.get(skel)
            if form:
                decisions[i] = Decision(form, "override")

        # הרכבה. כל צורה נבדקת מול השלד שלה לפני שהיא נכנסת.
        out_parts = list(p for p, _ in parts)
        unresolved: list[str] = []
        for i, part_i in enumerate(word_idx):
            d = decisions[i]
            if d.source == "unresolved":
                # לא מנחשים. המילה נשארת כפי שהיא ונרשמת לבדיקה.
                unresolved.append(skeletons[i])
                out_parts[part_i] = originals[i]
                continue
            if strip_nikud(d.form) != skeletons[i]:
                # שכבה החזירה צורה ששלדה שונה — נפסלת, המילה נשארת כמות שהיא.
                decisions[i] = Decision(originals[i], "unresolved", False,
                                        f"נפסל: שלד לא תואם ({d.source})")
                unresolved.append(skeletons[i])
                out_parts[part_i] = originals[i]
                continue
            out_parts[part_i] = d.form

        result_text = assert_identity(source, "".join(out_parts))
        return VocalizeResult(result_text, decisions, unresolved)

    # --- שכבת המודל, מבודדת ---------------------------------------------

    def _apply_model(self, source: str, skeletons: list[str],
                     decisions: list[Decision], frozen: set[int],
                     predicted: str | None = None) -> None:
        if predicted is None:
            if isinstance(self.model, NullBackend):
                return
            predicted = self.model.vocalize(strip_nikud(source))
        if predicted is None:
            return
        pred_parts = split_tokens(normalize(predicted))
        pred_words = [t for t, is_word in pred_parts if is_word]
        if len(pred_words) != len(skeletons):
            return
        for i, pw in enumerate(pred_words):
            if i in frozen or strip_nikud(pw) != skeletons[i]:
                continue
            decisions[i] = Decision(pw, "model")


def _resolve_homograph(rule, skeletons: list[str], i: int) -> str | None:
    """כלל הומוגרף: מחרוזת = הכרעה קבועה, או dict עם תנאי הקשר.

    {"default": "...", "after": {"של": "..."}, "before": {"שמים": "..."}}
    """
    if rule is None:
        return None
    if isinstance(rule, str):
        return rule
    if not isinstance(rule, dict):
        return None
    after = rule.get("after") or {}
    if i > 0 and skeletons[i - 1] in after:
        return after[skeletons[i - 1]]
    before = rule.get("before") or {}
    if i + 1 < len(skeletons) and skeletons[i + 1] in before:
        return before[skeletons[i + 1]]
    return rule.get("default")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# --- בונה מנועים לפי שם, ל-evaluate.py --------------------------------


# משאבים כבדים — אינדקס הציטוטים לבדו הוא מאות מגהבייט בזיכרון. הרצת
# השוואה בונה כמה מנועים, ובלי המטמון הזה כל אחד היה טוען עותק משלו.
_CACHE: dict[tuple, object] = {}


def _cached(key: tuple, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def load_shared(lexicon_dir: Path = LEXICON):
    """(לקסיקון, ציטוטים, הומוגרפים, חריגים) — נטענים פעם אחת לכל תיקייה."""
    return (
        _cached(("lexicon", lexicon_dir), lambda: Lexicon.load(lexicon_dir)),
        _cached(("quotes", lexicon_dir),
                lambda: QuoteIndex.load(lexicon_dir / "quotes.json")),
        _cached(("homographs", lexicon_dir),
                lambda: _load_json(lexicon_dir / "homographs.json", {})),
        _cached(("overrides", lexicon_dir),
                lambda: _load_json(lexicon_dir / "overrides.json", {})),
    )


def build_engine(spec: str, lexicon_dir: Path = LEXICON) -> Callable[[str], str]:
    """`null` | `lexicon` | `quotes` | `lexicon+quotes` | `dictabert` | `full`
    | `trained:<path>`.

    הפונקציה המוחזרת נושאת גם `.batch` לניקוד אצווה, כשיש בכך תועלת.
    """
    if spec == "null":
        return _with_batch(lambda t: t, lambda ts: list(ts))

    lex, quotes, homographs, overrides = load_shared(lexicon_dir)
    model_id = None
    if spec == "full":
        model_id = _MODEL_ID
    elif spec.startswith("trained:"):
        model_id = spec.split(":", 1)[1]
    elif spec == "dictabert":
        model_id = _MODEL_ID

    model: ModelBackend = NullBackend()
    if model_id:
        model = _cached(("model", model_id), lambda: DictaBertBackend(model_id))

    if spec == "lexicon":
        eng = Engine(lexicon=lex, homographs=homographs, overrides=overrides)
    elif spec == "quotes":
        eng = Engine(quotes=quotes)
    elif spec == "dictabert":
        # המודל לבדו, בלי אף שכבה מעליו — זה מה שהבסיס מודד.
        eng = Engine(model=model)
    elif spec in ("lexicon+quotes", "full") or spec.startswith("trained:"):
        eng = Engine(lexicon=lex, quotes=quotes, model=model,
                     homographs=homographs, overrides=overrides)
    else:
        raise ValueError(f"מנוע לא מוכר: {spec}")
    return _with_batch(eng.nikud, eng.nikud_batch)


def _with_batch(single, batch):
    """מחזיר קריאה יחידה שנושאת גם נתיב אצווה. מתודה קשורה לא מקבלת תכונות."""

    def fn(text: str) -> str:
        return single(text)

    fn.batch = batch
    return fn


# --- ה-API הציבורי -----------------------------------------------------

_DEFAULT: Engine | None = None


def nikud(text: str) -> str:
    """הפונקציה שהצנרת הקיימת קוראת לה."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Engine.load()
    return _DEFAULT.nikud(text)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ניקוד טקסט")
    p.add_argument("text", nargs="*", help="טקסט לניקוד. ריק = קריאה מ-stdin")
    p.add_argument("--engine", default="lexicon+quotes")
    p.add_argument("--stats", action="store_true", help="פילוח לפי שכבה")
    args = p.parse_args()

    text = " ".join(args.text) if args.text else sys.stdin.read()
    if args.stats:
        eng = Engine.load(
            with_model="dicta-il/dictabert-large-char-menaked"
            if args.engine in ("full",)
            else None
        )
        res = eng.vocalize(text)
        print(res.text)
        print(json.dumps(res.by_source, ensure_ascii=False), file=sys.stderr)
    else:
        print(build_engine(args.engine)(text))

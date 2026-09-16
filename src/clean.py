"""ניקוי וסגמנטציה. משותף ל-build_corpus ול-build_eval כדי ששני הסטים
ייבנו בדיוק באותו אופן.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hebrew import HEBREW_LETTERS, VOWELS, iter_letters, normalize, words  # noqa: E402

_HTML = re.compile(r"<[^>]{0,200}>")
_CURLY = re.compile(r"\{[^}]{0,200}\}")          # הערות שוליים ומראי מקום בספריא
_FOOTNOTE_MARK = re.compile(r"[\*†‡°]")
_NOTE_BRACKET = re.compile(r"\[[^\]]{0,4}\]")    # [1] [א] — מראי מקום, לא טקסט
_LATIN = re.compile(r"[A-Za-z]{2,}")
_WS = re.compile(r"\s+")
_NIKUD_ONLY_BRACKETS = str.maketrans("", "", "[]{}")

_SENTENCE_SPLIT = re.compile(r"(?<=[׃\.\?\!])\s+|(?<=[׃\.\?\!])$")


def clean_line(text: str) -> str:
    """מנקה שורה גולמית מספריא. לא נוגע בניקוד ולא באותיות."""
    text = normalize(text)
    text = _HTML.sub(" ", text)
    text = _CURLY.sub(" ", text)
    text = _NOTE_BRACKET.sub(" ", text)
    text = _FOOTNOTE_MARK.sub("", text)
    text = _LATIN.sub(" ", text)
    text = text.translate(_NIKUD_ONLY_BRACKETS)
    text = _WS.sub(" ", text).strip()
    return text


def split_sentences(text: str, *, min_words: int = 3, max_words: int = 60) -> list[str]:
    """פיצול למשפטים. משפט ארוך מדי נחתך על פסיקים כדי לא לחרוג מהקשר המודל."""
    out: list[str] = []
    for chunk in _SENTENCE_SPLIT.split(text):
        if not chunk:
            continue
        chunk = chunk.strip()
        n = len(words(chunk))
        if n < min_words:
            continue
        if n <= max_words:
            out.append(chunk)
            continue
        # חיתוך משני על פסיקים, תוך שמירה על הפיסוק עצמו.
        buf: list[str] = []
        count = 0
        for piece in re.split(r"(?<=,)\s+", chunk):
            buf.append(piece)
            count += len(words(piece))
            if count >= max_words:
                out.append(" ".join(buf).strip())
                buf, count = [], 0
        if buf and count >= min_words:
            out.append(" ".join(buf).strip())
    return out


def vocalization_coverage(text: str) -> tuple[float, float]:
    """(שיעור המילים שיש בהן תנועה, שיעור האותיות שנושאות תנועה)."""
    ws = words(text)
    if not ws:
        return 0.0, 0.0
    with_vowel = 0
    letters = 0
    vowelled_letters = 0
    for w in ws:
        found = False
        for base, marks in iter_letters(w):
            if base in HEBREW_LETTERS:
                letters += 1
                if any(m in VOWELS for m in marks):
                    vowelled_letters += 1
                    found = True
        if found:
            with_vowel += 1
    return with_vowel / len(ws), (vowelled_letters / letters if letters else 0.0)


def is_fully_vocalized(text: str, *, min_word_cov: float = 0.95,
                       min_letter_cov: float = 0.40) -> bool:
    """מסנן שורות עם ניקוד חלקי — הן רעל לאימון וללקסיקון כאחד."""
    word_cov, letter_cov = vocalization_coverage(text)
    return word_cov >= min_word_cov and letter_cov >= min_letter_cov

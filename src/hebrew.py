"""פרימיטיבים לטקסט עברי מנוקד.

המודול הזה הוא הבסיס של כל השאר. הוא מגדיר מה נחשב "ניקוד", איך מסירים
אותו, ואיך בודקים את הקו האדום: אחרי הסרת כל הניקוד הטקסט חייב להיות זהה
למקור תו־בתו.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator, Sequence

# --- טווחי יוניקוד ------------------------------------------------------

# טעמי המקרא. נחשבים ניקוד לצורך הסרה, אבל לא מנוקדים על ידי המנוע.
CANTILLATION = frozenset(chr(c) for c in range(0x0591, 0x05AF + 1))

# תנועות: שווא עד קובוץ, וקמץ קטן.
VOWELS = frozenset(
    [
        "ְ",  # שווא
        "ֱ",  # חטף סגול
        "ֲ",  # חטף פתח
        "ֳ",  # חטף קמץ
        "ִ",  # חיריק
        "ֵ",  # צירה
        "ֶ",  # סגול
        "ַ",  # פתח
        "ָ",  # קמץ
        "ֹ",  # חולם
        "ֺ",  # חולם חסר לוו
        "ֻ",  # קובוץ
        "ׇ",  # קמץ קטן
    ]
)

DAGESH = "ּ"  # דגש / מפיק / שורוק
SHIN_DOT = "ׁ"
SIN_DOT = "ׂ"
RAFE = "ֿ"
METEG = "ֽ"

# סימנים נוספים שמופיעים בטקסטים מנוקדים ואינם אותיות.
MISC_POINTS = frozenset([RAFE, METEG, "ׄ", "ׅ"])

# כל מה שמוסר ב-strip_nikud.
ALL_MARKS = frozenset(VOWELS | {DAGESH, SHIN_DOT, SIN_DOT} | MISC_POINTS | CANTILLATION)

# סימני פיסוק עבריים שהם תווים לכל דבר ואינם ניקוד. אסור להסיר אותם.
MAQAF = "־"
PASEQ = "׀"
SOF_PASUQ = "׃"
GERESH = "׳"
GERSHAYIM = "״"

HEBREW_LETTERS = frozenset(chr(c) for c in range(0x05D0, 0x05EA + 1))

_MARKS_RE = re.compile("[" + "".join(sorted(ALL_MARKS)) + "]")
_HEB_RE = re.compile(r"[א-ת]")

# טוקן = רצף של אותיות עבריות, ניקוד, וגרשיים/מקף שמחוברים למילה.
_TOKEN_RE = re.compile(
    r"[א-ת][ְ-ׇ֑-֯א-ת׳״'\"]*"
)

# --- נרמול --------------------------------------------------------------


def normalize(text: str) -> str:
    """NFC + איחוד תווים שקולים שמופיעים בפועל בקורפוסים."""
    text = unicodedata.normalize("NFC", text)
    # ליגטורות סופיות ותווי הצגה — אין להן מקום בטקסט עבודה.
    text = text.replace("װ", "וו").replace("ױ", "וי").replace("ײ", "יי")
    # גרש/גרשיים בתווי ASCII מנורמלים לתווים העבריים רק בהשוואה, לא בפלט.
    return text


def strip_nikud(text: str, *, keep_cantillation: bool = False) -> str:
    """מחזיר את שלד העיצורים. זו הפונקציה שבדיקת הזהות נשענת עליה."""
    if keep_cantillation:
        marks = ALL_MARKS - CANTILLATION
        return re.sub("[" + "".join(sorted(marks)) + "]", "", text)
    return _MARKS_RE.sub("", text)


def has_nikud(text: str) -> bool:
    return bool(_MARKS_RE.search(text))


def is_hebrew(text: str) -> bool:
    return bool(_HEB_RE.search(text))


def nikud_ratio(text: str) -> float:
    """שיעור האותיות שנושאות תנועה. משמש לסינון ניקוד חלקי."""
    letters = [c for c in text if c in HEBREW_LETTERS]
    if not letters:
        return 0.0
    vocalized = 0
    for _base, marks in iter_letters(text):
        if any(m in VOWELS for m in marks):
            vocalized += 1
    return vocalized / len(letters)


# --- הקו האדום ----------------------------------------------------------


class IdentityError(AssertionError):
    """הפלט אינו זהה למקור אחרי הסרת ניקוד."""


def identity_diff(source: str, vocalized: str) -> list[tuple[int, str, str]]:
    """מחזיר את ההפרשים בין שלד הפלט לשלד המקור. רשימה ריקה = תקין."""
    a = strip_nikud(normalize(source))
    b = strip_nikud(normalize(vocalized))
    if a == b:
        return []
    diffs: list[tuple[int, str, str]] = []
    for i in range(max(len(a), len(b))):
        ca = a[i] if i < len(a) else ""
        cb = b[i] if i < len(b) else ""
        if ca != cb:
            diffs.append((i, a[max(0, i - 20) : i + 20], b[max(0, i - 20) : i + 20]))
            break
    return diffs


def assert_identity(source: str, vocalized: str) -> str:
    """הקו האדום. מחזיר את הפלט אם הוא תקין, ומתפוצץ אחרת.

    אין לעקוף, לרכך, או להתאים את הבדיקה לפלט. טקסט שנאכל גרוע מניקוד שגוי.
    """
    diffs = identity_diff(source, vocalized)
    if diffs:
        _, ctx_src, ctx_out = diffs[0]
        raise IdentityError(
            "בדיקת הזהות נכשלה — הפלט אינו זהה למקור אחרי הסרת ניקוד.\n"
            f"  מקור: ...{ctx_src}...\n"
            f"  פלט:  ...{ctx_out}..."
        )
    return vocalized


# --- אותיות וסימניהן ----------------------------------------------------


def iter_letters(text: str) -> Iterator[tuple[str, tuple[str, ...]]]:
    """מפרק לטקסט ל-(אות בסיס, סימנים). תווים שאינם אותיות מוחזרים ללא סימנים."""
    i = 0
    n = len(text)
    while i < n:
        base = text[i]
        i += 1
        marks: list[str] = []
        while i < n and text[i] in ALL_MARKS:
            marks.append(text[i])
            i += 1
        yield base, tuple(marks)


@dataclass(frozen=True)
class LetterSlot:
    """הסימנים של אות אחת, מפורקים לשלוש המחלקות שהמודל מתייג."""

    base: str
    vowel: str = ""      # תנועה אחת או ריק
    dagesh: bool = False
    shin: str = ""       # "" / SHIN_DOT / SIN_DOT

    @property
    def is_letter(self) -> bool:
        return self.base in HEBREW_LETTERS


def to_slots(text: str) -> list[LetterSlot]:
    """פירוק לסלוטים. סימנים שאינם בשלוש המחלקות (טעמים, מתג) מושמטים."""
    slots = []
    for base, marks in iter_letters(text):
        vowel = ""
        dagesh = False
        shin = ""
        for m in marks:
            if m in VOWELS:
                vowel = vowel or m
            elif m == DAGESH:
                dagesh = True
            elif m == SHIN_DOT:
                shin = SHIN_DOT
            elif m == SIN_DOT:
                shin = SIN_DOT
        slots.append(LetterSlot(base, vowel, dagesh, shin))
    return slots


def slots_to_text(slots: Sequence[LetterSlot]) -> str:
    """הרכבה חזרה. סדר הסימנים: נקודת שי"ן, דגש, תנועה — סדר יוניקוד הקנוני."""
    out = []
    for s in slots:
        out.append(s.base)
        if s.shin:
            out.append(s.shin)
        if s.dagesh:
            out.append(DAGESH)
        if s.vowel:
            out.append(s.vowel)
    return unicodedata.normalize("NFC", "".join(out))


def canonical(text: str) -> str:
    """צורה קנונית להשוואה: NFC, בלי טעמים, בלי מתג/רפה, סדר סימנים קבוע."""
    return slots_to_text(to_slots(normalize(text)))


# --- טוקניזציה ----------------------------------------------------------


def split_tokens(text: str) -> list[tuple[str, bool]]:
    """מפרק ל-(מקטע, האם זו מילה עברית). שרשור המקטעים מחזיר את המקור בדיוק."""
    parts: list[tuple[str, bool]] = []
    pos = 0
    for m in _TOKEN_RE.finditer(text):
        if m.start() > pos:
            parts.append((text[pos : m.start()], False))
        parts.append((m.group(), True))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], False))
    return parts


def words(text: str) -> list[str]:
    return [t for t, is_word in split_tokens(text) if is_word]


def is_acronym(token: str) -> bool:
    """ראשי תיבות: גרשיים בתוך המילה, או גרש בסופה.

    ראשי תיבות לא מנוקדים לעולם — זה כלל של הצנרת הקיימת, והמנוע מכבד אותו.
    """
    body = strip_nikud(token)
    if any(c in body[:-1] for c in (GERSHAYIM, '"')):
        return True
    if body and body[-1] in (GERESH, "'"):
        return True
    return False


# --- השוואה -------------------------------------------------------------


def word_matches(gold: str, pred: str, *, ignore_shin: bool = False) -> bool:
    """האם שתי צורות זהות בניקוד. ignore_shin מבודד את בעיית שי"ן/שי"ן שמאלית."""
    g = to_slots(normalize(gold))
    p = to_slots(normalize(pred))
    if len(g) != len(p):
        return False
    for gs, ps in zip(g, p):
        if gs.base != ps.base:
            return False
        if gs.vowel != ps.vowel or gs.dagesh != ps.dagesh:
            return False
        if not ignore_shin and gs.shin != ps.shin:
            return False
    return True


def char_errors(gold: str, pred: str) -> tuple[int, int]:
    """(מספר סלוטים שגויים, סך הסלוטים). סלוט = אות אחת עם כל סימניה."""
    g = to_slots(normalize(gold))
    p = to_slots(normalize(pred))
    if len(g) != len(p):
        # שלד שונה — כל הסלוטים נחשבים שגויים. לא אמור לקרות אחרי בדיקת זהות.
        return max(len(g), len(p)), max(len(g), len(p))
    errors = 0
    for gs, ps in zip(g, p):
        if (gs.vowel, gs.dagesh, gs.shin) != (ps.vowel, ps.dagesh, ps.shin):
            errors += 1
    return errors, len(g)

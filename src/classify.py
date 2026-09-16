"""סיווגים לפילוח ההערכה: ארמית, שמות פרטיים, סמיכות, הומוגרפים.

כל הסיווגים כאן הם היוריסטיקות מוצהרות, לא אמת מוחלטת. תפקידם לבודד
משפחות שגיאות בדוח ההערכה, לא להכריע לשונית.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hebrew import strip_nikud, to_slots, words  # noqa: E402

# --- ארמית --------------------------------------------------------------

# מילות תפקוד ארמיות תלמודיות. נבחרו כי הן שכיחות ואינן מילים עבריות.
ARAMAIC_MARKERS = {
    "מאי", "האי", "הכי", "הכא", "התם", "ליה", "להו", "לן", "לך", "מילתא",
    "טעמא", "קא", "דקא", "אמאי", "אלמא", "איכא", "ליכא", "מנא", "תנן",
    "תניא", "גמרא", "אביי", "רבא", "אמרי", "וכו", "מידי", "נמי", "הוה",
    "דהא", "אלא", "פשיטא", "תיובתא", "שמעת", "מינה", "קמ", "בעי", "איבעיא",
    "אדרבה", "בשלמא", "לימא", "מסתברא", "הדר", "אתא", "עלמא", "מילי",
}

# תחיליות ארמיות: ד' הזיקה, כד, דלמא.
_ARAMAIC_PREFIX = re.compile(r"^ד[א-ת]{2,}$")


def aramaic_score(text: str) -> float:
    """שיעור הטוקנים שנראים ארמיים. 0 = עברית טהורה."""
    toks = [strip_nikud(w) for w in words(text)]
    if not toks:
        return 0.0
    hits = sum(
        1
        for t in toks
        if t in ARAMAIC_MARKERS or _ARAMAIC_PREFIX.match(t) or t.endswith("ייהו")
    )
    return hits / len(toks)


def is_aramaic(text: str, threshold: float = 0.08) -> bool:
    return aramaic_score(text) >= threshold


# --- שמות פרטיים --------------------------------------------------------

# תארים שאחריהם בא כמעט תמיד שם פרטי.
NAME_TRIGGERS = {
    "רבי", "רב", "רבן", "רבנו", "בן", "בר", "מר", "אבא", "שמעון", "מרן",
    "הרב", "ורבי", "ורב", "דרבי", "דרב",
}

# זרע ראשוני. הלקסיקון מרחיב אותו מהקורפוס.
SEED_NAMES = {
    "אליעזר", "יהושע", "עקיבא", "מאיר", "יהודה", "יוסי", "שמעון", "גמליאל",
    "אלעזר", "ישמעאל", "טרפון", "חנינא", "יוחנן", "אביי", "רבא", "אשי",
    "הונא", "חסדא", "נחמן", "ששת", "פפא", "זירא", "הלל", "שמאי", "אברהם",
    "יצחק", "יעקב", "משה", "אהרן", "דוד", "שלמה", "שמואל", "ירמיהו",
    "ישעיהו", "יחזקאל", "נחמיה", "עזרא", "אסתר", "מרדכי", "רחל", "לאה",
    "שרה", "רבקה", "מרים", "יוסף", "בנימין", "ראובן", "שמעון", "לוי",
}


def name_positions(text: str, extra_names: set[str] | None = None) -> set[int]:
    """אינדקסים של טוקנים שסביר שהם שמות פרטיים."""
    gaz = SEED_NAMES | (extra_names or set())
    toks = [strip_nikud(w) for w in words(text)]
    out: set[int] = set()
    for i, t in enumerate(toks):
        if t in gaz:
            out.add(i)
        elif i > 0 and toks[i - 1] in NAME_TRIGGERS and t not in NAME_TRIGGERS:
            out.add(i)
    return out


# --- סמיכות -------------------------------------------------------------

_HOLAM = "ֹ"
_TSERE = "ֵ"
_PATAH = "ַ"
_QAMATS = "ָ"
_SHEVA = "ְ"


# סמיכויות תדירות שאינן ניתנות לזיהוי מורפולוגי (סגוליים, מילות יחס).
CONSTRUCT_FORMS = {
    "בית", "בן", "בת", "כל", "אנשי", "שם", "דבר", "יד", "פי", "אל",
    "ערב", "בני", "בנות", "ראש", "עין", "אם", "אב", "דם", "חצי",
}


def is_construct(word: str) -> bool:
    """היוריסטיקה מורפולוגית על הצורה המנוקדת.

    שתי הסיומות שמכריעות בפועל:
      רבים זכר בסמיכות  ־ֵי   (לעומת ־ִים בנפרד)
      נקבה יחיד בסמיכות ־ַת   (לעומת ־ָה בנפרד)

    סגוליים כמו בֵּית אינם ניתנים לזיהוי כך, ולכן יש גם רשימה סגורה קצרה.
    ההיוריסטיקה מוטה לדיוק על חשבון כיסוי — היא משרתת פילוח בדוח, לא הכרעה.
    """
    if strip_nikud(word) in CONSTRUCT_FORMS:
        return True
    slots = [s for s in to_slots(word) if s.is_letter]
    if len(slots) < 2:
        return False
    last, prev = slots[-1], slots[-2]
    if last.base == "י" and prev.vowel == _TSERE:          # ־ֵי
        return True
    if last.base == "ת" and last.vowel == _PATAH:          # ־ַת
        return True
    if last.base == "ת" and prev.vowel == _HOLAM:          # ־וֹת בסמיכות
        return False
    return False


def construct_positions(text: str) -> set[int]:
    ws = words(text)
    return {i for i, w in enumerate(ws) if i + 1 < len(ws) and is_construct(w)}


# --- הומוגרפים ----------------------------------------------------------


def homograph_positions(text: str, homograph_skeletons: set[str]) -> set[int]:
    """טוקנים ששלד העיצורים שלהם נושא יותר מניקוד אחד בלקסיקון."""
    toks = [strip_nikud(w) for w in words(text)]
    return {i for i, t in enumerate(toks) if t in homograph_skeletons}

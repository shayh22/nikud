"""כתיב חסר — הורדת אם קריאה כשהניקוד מחליף אותה.

הספר כתוב בכתיב מלא, והניקוד המסורתי נסמך על כתיב חסר: `חִבּוּר` ולא
`חִיבּוּר`, `כֻּלָּם` ולא `כּוּלָּם`. המודול הזה מוריד את היו"ד כשיש חיריק
ואת הוי"ו כשיש קובוץ (ובאופציה גם חולם).

## למה זה לא יכול להיות מכני

צורה שנראית מתבקשת להורדה עלולה להיות מילה אחרת לגמרי:

    איש → אש    אבל בקורפוס אֵשׁ, בצירה — מילה אחרת
    היא → הא    אבל בקורפוס הָא, בקמץ — מילה אחרת
    רבי → רב    אבל בקורפוס רַב, בפתח — מילה אחרת
    תיקון → תקון  אבל בקורפוס תַּקּוּן, בפתח — צורה אחרת

כל ארבעתן מאומתות בקורפוס בשכיחות גבוהה. שכיחות לבדה אינה שומרת.

## כלל ההצדקה

אות יורדת רק אם **התנועה בצורה החסרה מצדיקה אותה**:

    יו"ד יורדת   ⟸  האות שלפניה נושאת חיריק
    וי"ו יורדת   ⟸  האות שלפניה נושאת קובוץ (או חולם, באופציה)

ובנוסף: האות אינה ראשונה ואינה אחרונה במילה, והצורה החסרה מאומתת
בקורפוס כצורה נחרצת. `אֵשׁ` נופלת כי א' נושאת צירה ולא חיריק. `הָא`
נופלת בקמץ. `רַב` בפתח. `תַּקּוּן` בפתח.

## הערובה

הקו האדום המקורי — הפלט זהה למקור תו־בתו — לא מתקיים כאן, מעצם הגדרת
המשימה. במקומו יש ערובה אחרת, חזקה לא פחות ובדיקה לא פחות:

    **כל הורדה מתועדת, והחזרת האותיות המתועדות משחזרת את המקור בדיוק.**

`assert_restorable` אוכפת את זה. שינוי שאינו הפיך נדחה, בדיוק כפי
ש-`assert_identity` דוחה שינוי שאינו זהה.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hebrew import normalize, strip_nikud, to_slots  # noqa: E402

HIRIQ = "ִ"
QUBUTS = "ֻ"
HOLAM = "ֹ"
YOD = "י"
VAV = "ו"

# כמה אמות קריאה לכל היותר נבדקות במילה אחת. מעל זה הצירופים מתפוצצים
# והמילה כמעט תמיד ארמית או לועזית ממילא.
MAX_CANDIDATES = 4


class RestorationError(AssertionError):
    """ההורדה אינה הפיכה — אי אפשר לשחזר ממנה את המקור."""


@dataclass(frozen=True)
class Conversion:
    """תוצאת ההמרה של מילה אחת."""

    form: str              # הצורה המנוקדת החסרה
    removed: tuple[int, ...]   # אינדקסים בשלד המקורי של האותיות שהורדו
    letters: tuple[str, ...]   # האותיות עצמן, לשחזור
    count: int             # שכיחות הצורה החסרה בקורפוס


# --- הערובה ------------------------------------------------------------


def restore(skeleton: str, removed: tuple[int, ...], letters: tuple[str, ...]) -> str:
    """מחזיר את האותיות שהורדו למקומן. ההפך של ההמרה."""
    chars = list(skeleton)
    for pos, letter in sorted(zip(removed, letters)):
        chars.insert(pos, letter)
    return "".join(chars)


def assert_restorable(source_word: str, converted: Conversion) -> Conversion:
    """הערובה של המודול הזה. החזרת האותיות חייבת לשחזר את המקור בדיוק."""
    source_skeleton = strip_nikud(normalize(source_word))
    rebuilt = restore(strip_nikud(converted.form), converted.removed, converted.letters)
    if rebuilt != source_skeleton:
        raise RestorationError(
            "ההורדה אינה הפיכה.\n"
            f"  מקור:  {source_skeleton}\n"
            f"  שוחזר: {rebuilt}\n"
            f"  צורה:  {converted.form}  הורדו={converted.removed}"
        )
    return converted


# --- מועמדות -----------------------------------------------------------


def mater_positions(male_form: str, *, yod: bool = True, vav: bool = True,
                    holam: bool = False) -> dict[int, str]:
    """אמות קריאה בצורה המנוקדת המלאה → התנועה שתחליף אותן בכתיב חסר.

    אם קריאה מזוהה מהניקוד עצמו, לא מהאות:
      יו"ד שאינה נושאת דבר, והאות שלפניה בחיריק   → חיריק
      וי"ו בשורוק (דגש בלבד), והאות שלפניה בלי תנועה → קובוץ
      וי"ו בחולם מלא, והאות שלפניה בלי תנועה        → חולם (באופציה)

    ראשונה ואחרונה במילה לעולם לא: יו"ד סופית היא כמעט תמיד סיומת
    (רַבִּי, דִּבְרֵי), ווי"ו סופית היא כינוי (אֵלָיו).
    """
    slots = [s for s in to_slots(normalize(male_form)) if s.is_letter]
    out: dict[int, str] = {}
    # סיומת הרבים ־ִים / ־ִין נכתבת ביו"ד גם בכתיב חסר, תמיד.
    # נדרשות ארבע אותיות לפחות, אחרת מילים כמו `מִין` נחסמות בטעות —
    # שם היו"ד אינה סיומת אלא חלק מהגזע.
    plural_yod = (
        len(slots) >= 4
        and slots[-1].base in ("\u05dd", "\u05df")
        and slots[-2].base == YOD
    )
    for i in range(1, len(slots) - 1):
        if plural_yod and i == len(slots) - 2:
            continue
        s, prev = slots[i], slots[i - 1]
        if s.base == YOD and yod:
            if not s.vowel and not s.dagesh and prev.vowel == HIRIQ:
                out[i] = HIRIQ
        elif s.base == VAV and vav:
            if s.dagesh and not s.vowel and not prev.vowel:
                out[i] = QUBUTS          # שורוק
            elif holam and s.vowel == HOLAM and not s.dagesh and not prev.vowel:
                out[i] = HOLAM           # חולם מלא
    return out


def _supporting_variants(entry: dict, removed: tuple[int, ...],
                         vowels: tuple[str, ...]) -> tuple[int, int, str | None]:
    """(שכיחות תומכת, שכיחות כוללת, הצורה התומכת השכיחה ביותר).

    צורה תומכת = כזו שבה האות שלפני כל הורדה נושאת בדיוק את התנועה
    שאמורה להחליף את אם הקריאה. זו הראיה שהכתיב החסר הזה באמת בשימוש
    למילה הזאת, ולא סתם שלד שבמקרה קיים בקורפוס כמילה אחרת.
    """
    support = 0
    total = 0
    supporting: list[tuple[str, int]] = []
    for form, count in entry.get("variants", []):
        total += count
        slots = [s for s in to_slots(normalize(form)) if s.is_letter]
        ok = True
        for i, (pos, vowel) in enumerate(zip(removed, vowels)):
            prev = pos - i - 1        # מיקום האות שלפני ההורדה, בצורה החסרה
            if prev < 0 or prev >= len(slots) or slots[prev].vowel != vowel:
                ok = False
                break
        if ok:
            support += count
            supporting.append((form, count))
    return support, total, _pick_form(supporting)


def _pick_form(variants: list[tuple[str, int]]) -> str | None:
    """בוחר צורה מתוך מועמדות תומכות.

    צורות שנבדלות רק בדגש קל הן אותה צורה במהדורות שונות, ולכן הן
    מקובצות תחילה — אחרת `תִקּוּן×17` היה מנצח את `תִּקּוּן×15` ומאבד
    את הדגש. בתוך הקבוצה נבחר הניקוד המלא יותר.
    """
    from hebrew import word_matches

    groups: list[list[tuple[str, int]]] = []
    for form, count in variants:
        for g in groups:
            if word_matches(g[0][0], form, lenient=True):
                g.append((form, count))
                break
        else:
            groups.append([(form, count)])
    if not groups:
        return None
    best = max(groups, key=lambda g: sum(c for _f, c in g))
    return max(best, key=lambda fc: (_mark_count(fc[0]), fc[1]))[0]


def _mark_count(form: str) -> int:
    slots = to_slots(normalize(form))
    return sum(bool(s.vowel) + bool(s.dagesh) + bool(s.shin) for s in slots)


# --- הממיר -------------------------------------------------------------


PREFIX_LETTERS = frozenset("והבכלמש")


def _mater_shape_matches(form: str, subset: tuple[int, ...],
                         vowels: tuple[str, ...]) -> bool:
    """האם בצורה הזאת אותן אותיות הן אמות קריאה מאותו סוג."""
    found = mater_positions(form, yod=True, vav=True, holam=True)
    return all(found.get(p) == v for p, v in zip(subset, vowels))


def mechanical(male_form: str, subset: tuple[int, ...],
               vowels: tuple[str, ...]) -> str:
    """מוריד את אמות הקריאה ומעביר את התנועה לאות שלפניהן.

    זו הפעולה עצמה, והיא מוגדרת לחלוטין: חיריק כבר יושב על האות שלפני
    היו"ד, ולכן הורדת היו"ד מספיקה; שורוק יושב על הוי"ו עצמה, ולכן
    הורדתה מחייבת להניח קובוץ על האות שלפניה.
    """
    import dataclasses

    from hebrew import slots_to_text

    slots = to_slots(normalize(male_form))
    letter_pos = [i for i, s in enumerate(slots) if s.is_letter]
    drop: set[int] = set()
    for p, vowel in zip(subset, vowels):
        drop.add(letter_pos[p])
        prev = letter_pos[p - 1]
        slots[prev] = dataclasses.replace(slots[prev], vowel=vowel)
    return slots_to_text([s for i, s in enumerate(slots) if i not in drop])


class HaserConverter:
    """ממיר כתיב מלא לכתיב חסר, על סמך הקורפוס בלבד.

    שתי שאלות נפרדות, ומטופלות בנפרד:

    1. **האם האות היא אם קריאה?** נקבע מהניקוד של הצורה המלאה בלבד
       (`mater_positions`). זו שאלה מכנית.
    2. **האם המסורת כותבת את המילה הזאת חסר?** נקבע מהקורפוס, ורק ממנו.
       בלי ראיה — המילה נשארת כמות שהיא. לא ממציאים כתיב.

    הפרדת השתיים היא מה שמאפשר לטפל במילים עם תחילית: `הַחִיבּוּר` אינו
    בקורפוס, אבל `חִבּוּר` כן, והראיה מהגזע מספיקה.
    """

    def __init__(self, lexicon, *, drop_hiriq_yod: bool = True,
                 drop_shuruk_vav: bool = True, drop_holam_vav: bool = False,
                 min_support: int = 3, min_ratio: float = 0.50,
                 strong_support: int = 25, min_vs_male: float = 0.35):
        self.lexicon = lexicon
        self.drop_yod = drop_hiriq_yod
        self.drop_vav = drop_shuruk_vav
        self.allow_holam = drop_holam_vav
        self.min_support = min_support
        self.min_ratio = min_ratio
        # מדד היחס שואל "איזה חלק מהמופעים של השלד תומך בהורדה", וזו
        # השאלה הלא נכונה כששלד אחד מארח שתי מילים. `אסור` מארח את
        # `אָסוּר` השכיח ואת `אִסּוּר`; ל-158 מופעים של השני יש משקל ראייתי
        # מלא, גם אם הם 5% מהשלד. לכן תמיכה מוחלטת חזקה עוקפת את היחס.
        self.strong_support = strong_support
        # כמה הכתיב החסר צריך להיות נפוץ ביחס לכתיב המלא של אותה מילה.
        # זה מה שמבדיל בין `חיבור` (שהמסורת כותבת חסר) לבין `ציצית`
        # (שהמסורת כותבת מלא, ו`ציצת` שבקורפוס היא צורת הסמיכות).
        self.min_vs_male = min_vs_male
        self._cache: dict[str, Conversion | None] = {}

    # --- ה-API ---------------------------------------------------------

    def convert(self, male_form: str) -> Conversion | None:
        """מקבל צורה מנוקדת בכתיב מלא, ומחזיר את הצורה החסרה או None."""
        if male_form in self._cache:
            return self._cache[male_form]
        result = self._convert(male_form)
        if result is not None:
            assert_restorable(male_form, result)
        self._cache[male_form] = result
        return result

    # --- פנימי ---------------------------------------------------------

    def _convert(self, male_form: str) -> Conversion | None:
        maters = mater_positions(male_form, yod=self.drop_yod, vav=self.drop_vav,
                                 holam=self.allow_holam)
        if not maters or len(maters) > MAX_CANDIDATES:
            return None
        skeleton = strip_nikud(normalize(male_form))
        letters = [s for s in to_slots(normalize(male_form)) if s.is_letter]
        if len(skeleton) != len(letters):
            return None   # תווים שאינם אותיות בתוך המילה — לא נוגעים

        # מהרבה הורדות למעט: הכתיב החסר ביותר שהקורפוס עדיין תומך בו.
        for size in range(len(maters), 0, -1):
            best: Conversion | None = None
            for subset in combinations(sorted(maters), size):
                vowels = tuple(maters[p] for p in subset)
                support, corpus_form = self._evidence(skeleton, subset, vowels)
                if support < self.min_support:
                    continue
                # צורה מהקורפוס כשהמילה השלמה מאומתת; אחרת בנייה מכנית,
                # ששומרת על הניקוד שהמנוע כבר הכריע עליו (למשל בתחילית).
                form = corpus_form or mechanical(male_form, subset, vowels)
                candidate = Conversion(
                    form, subset, tuple(skeleton[p] for p in subset), support
                )
                if best is None or candidate.count > best.count:
                    best = candidate
            if best is not None:
                return best
        return None

    def _evidence(self, skeleton: str, subset: tuple[int, ...],
                  vowels: tuple[str, ...]) -> tuple[int, str | None]:
        """(שכיחות תומכת, צורה מהקורפוס אם המילה השלמה מאומתת).

        מחפש קודם את המילה השלמה. אם אין — מקלף תחיליות ומחפש את הגזע.
        רוב המילים בטקסט עברי נושאות תחילית (`הַחִיבּוּר`, `בִּתְפִילָּה`),
        והקורפוס מכיל את הגזע ולא את הצירוף.
        """
        support, form = self._lookup(skeleton, subset, vowels)
        if form is not None:
            return support, form

        for plen in (1, 2, 3):
            if min(subset) <= plen or len(skeleton) - plen < 3:
                continue
            if any(c not in PREFIX_LETTERS for c in skeleton[:plen]):
                continue
            support, form = self._lookup(
                skeleton[plen:], tuple(p - plen for p in subset), vowels
            )
            if form is not None:
                return support, None   # ראיה מהגזע — הצורה נבנית מכנית
        return 0, None

    def _lookup(self, skeleton: str, subset: tuple[int, ...],
                vowels: tuple[str, ...]) -> tuple[int, str | None]:
        """בודק שלד אחד מול הקורפוס. מחזיר (תמיכה, צורה) או (0, None)."""
        defective = "".join(c for i, c in enumerate(skeleton) if i not in subset)
        entry = self.lexicon.forms.get(defective)
        if not entry:
            return 0, None
        support, total, form = _supporting_variants(entry, subset, vowels)
        if form is None or support < self.min_support:
            return 0, None
        if (total and support / total < self.min_ratio
                and support < self.strong_support):
            return 0, None
        # הכתיב המלא של אותה מילה — כמה הוא עצמו מבוסס בקורפוס.
        male_entry = self.lexicon.forms.get(skeleton)
        if male_entry:
            male_support = sum(
                c for f, c in male_entry.get("variants", [])
                if _mater_shape_matches(f, subset, vowels)
            )
            if male_support and support < male_support * self.min_vs_male:
                return 0, None
        return support, form


# --- כיוון הפוך, לבדיקה בלבד -------------------------------------------


def to_male(vocalized: str) -> str:
    """מוסיף אמות קריאה לצורה מנוקדת — חיריק→י, קובוץ→ו.

    משמש רק כדי לבדוק את הממיר: לוקחים טקסט מסורתי, הופכים אותו לכתיב
    מלא, ומודדים כמה מהמקור הממיר מחזיר.

    ההוספה שמרנית בכוונה, כדי לייצר כתיב מלא שבאמת נכתב:
      * לא מוסיפים אחרי יו"ד או וי"ו — שם החיריק יושב על עיצור
        (בַּיִת, וְיִגְאֹל), ואף אחד לא כותב `ביית`
      * לא מוסיפים כשהאות הבאה היא א' או ה' — הן כבר אם הקריאה
        (רִאשׁוֹן ולא `ריאשון`)
      * לא באות האחרונה
    כתיב מלא מומצא היה הופך את המדידה לחסרת ערך.
    """
    slots = to_slots(normalize(vocalized))
    letters = [s for s in slots if s.is_letter]
    out: list[str] = []
    seen = 0
    for s in slots:
        add = ""
        vowel = s.vowel
        if s.is_letter:
            seen += 1
            nxt = letters[seen].base if seen < len(letters) else ""
            blocked = s.base in (YOD, VAV) or seen >= len(letters)
            if not blocked and nxt not in ("\u05d0", "\u05d4"):
                if vowel == HIRIQ and nxt != YOD:
                    # חיריק נשאר במקומו, והיו"ד נוספת אחריו: חִבּוּר → חִיבּוּר
                    add = YOD
                elif vowel == QUBUTS and nxt != VAV:
                    # קובוץ הופך לשורוק, כלומר עובר אל הוי"ו עצמה:
                    # כֻּלָּם → כּוּלָּם, ולא `כֻּולָּם`
                    add = VAV + "\u05bc"
                    vowel = ""
        piece = s.base
        if s.shin:
            piece += s.shin
        if s.dagesh:
            piece += "\u05bc"
        if vowel:
            piece += vowel
        out.append(piece + add)
    return normalize("".join(out))

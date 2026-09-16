"""קטלוג המקורות. כל מקור נושא תיוג שכבה ודגל רישוי.

הדגל `commercial_ok` מכריע כאן ולא אחר כך, כפי שדורש סעיף 4 של התוכנית:
עדיף לאמן פעם אחת על הסט הנכון מאשר לגלות בדיעבד שהמודל נגוע.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    title: str              # שם הקטגוריה או הספר ב-shape API
    layer: str              # biblical / mishnaic / aramaic / liturgy
    commercial_ok: bool     # False = CC-BY-NC, לספר פרטי בלבד
    note: str = ""
    # העדפת מהדורה: המהדורה הראשונה שכותרתה מכילה אחת מהמחרוזות תיבחר.
    prefer_versions: tuple[str, ...] = field(default_factory=tuple)


CATALOG: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            key="tanakh",
            title="Tanakh",
            layer="biblical",
            commercial_ok=True,
            note="Miqra according to the Masorah — CC-BY-SA. מנוקד ומוטעם.",
            prefer_versions=("Miqra according to the Masorah",),
        ),
        Source(
            key="mishnah",
            title="Mishnah",
            layer="mishnaic",
            commercial_ok=True,
            note="Torat Emet 357 — נחלת הכלל. מנוקד עם דגש ושי\"ן.",
            prefer_versions=("Torat Emet",),
        ),
        Source(
            key="talmud",
            title="Bavli",
            layer="aramaic",
            commercial_ok=False,
            note="William Davidson Edition — CC-BY-NC. ניקוד של דיקטה. "
            "לספר פרטי בסדר, למוצר מסחרי אסור.",
            prefer_versions=("William Davidson Edition - Vocalized Aramaic",),
        ),
        Source(
            key="siddur",
            title="Siddur Ashkenaz",
            layer="liturgy",
            commercial_ok=True,
            note="סידור אשכנז מנוקד.",
        ),
        Source(
            key="mishneh_torah",
            # shape API דורש את נתיב הקטגוריה המלא. "Mishneh Torah" לבדו
            # מחזיר "No index or category found".
            title="Halakhah/Mishneh Torah",
            layer="mishnaic",
            commercial_ok=True,
            note="רמב\"ם, Torat Emet — נחלת הכלל. פרוזה הלכתית מנוקדת, "
            "המשלב הקרוב ביותר לספרות רבנית מאוחרת.",
            prefer_versions=("Torat Emet",),
        ),
    ]
}

# מסלול ברירת המחדל: כל מה שמותר גם למוצר מסחרי.
COMMERCIAL_SAFE = tuple(k for k, s in CATALOG.items() if s.commercial_ok)
# המסלול המלא, לספר פרטי.
ALL_SOURCES = tuple(CATALOG)

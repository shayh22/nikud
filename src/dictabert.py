"""עטיפה ל-DictaBERT-char-menaked — קידוד, פענוח, ותיוג לאימון.

הערה חשובה: ה-`predict` שמגיע עם המודל נשען על `return_offsets_mapping`
של הטוקנייזר, ובגרסאות transformers חדשות המיפוי הזה כבר לא ברמת התו —
הפלט יוצא משובש. לכן כאן בונים את ה-input_ids תו־בתו בעצמנו. זה גם מה
שנדרש בלאו הכי לבדיקת הזהות: יישור מדויק בין כל תו לתחזית שלו.

ראש התיוג הוא ברמת התו, עם מחלקות נפרדות לניקוד ולנקודת שי"ן — בדיוק
המבנה שסעיף 6 של התוכנית מבקש לאימון־המשך.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hebrew import (  # noqa: E402
    DAGESH,
    HEBREW_LETTERS,
    SHIN_DOT,
    SIN_DOT,
    VOWELS,
    identity_diff,
    normalize,
    strip_nikud,
)

MODEL_ID = "dicta-il/dictabert-large-char-menaked"
MAX_CHARS = 1800          # תקרת התו של הטוקנייזר היא 2048, משאירים שוליים
SHIN = "ש"


# --- מיפוי בין טקסט למחלקות --------------------------------------------


def _mark_key(marks: str) -> str:
    """מפתח יציב לקבוצת סימנים, ללא תלות בסדר שבו נכתבו."""
    return "".join(sorted(marks))


def char_labels(text: str, nikud_classes: list[str], shin_classes: list[str],
                mat_lect_token: str = "<MAT_LECT>") -> tuple[list[str], list[int], list[int]]:
    """מפרק טקסט מנוקד ל-(תווי בסיס, מחלקת ניקוד לכל תו, מחלקת שי"ן לכל תו).

    מחלקה -100 = אין תיוג (תו שאינו אות עברית) — מתעלמים ממנה ב-loss.
    """
    # המחלקות המורכבות (דגש+תנועה) נכתבות בסדר דגש־ואז־תנועה, אבל NFC
    # ממיין סימנים לפי מחלקת הצירוף הקנונית ועלול להפוך אותו. לכן המפתח
    # הוא קבוצת הסימנים הממוינת, לא המחרוזת כפי שהיא.
    nikud_index = {_mark_key(c): i for i, c in enumerate(nikud_classes)}
    shin_index = {_mark_key(c): i for i, c in enumerate(shin_classes)}
    mat_lect_id = nikud_index.get(_mark_key(mat_lect_token), -100)

    bases: list[str] = []
    nikud_ids: list[int] = []
    shin_ids: list[int] = []

    text = normalize(text)
    i = 0
    while i < len(text):
        base = text[i]
        i += 1
        marks: list[str] = []
        while i < len(text) and (
            text[i] in VOWELS or text[i] in (DAGESH, SHIN_DOT, SIN_DOT)
        ):
            marks.append(text[i])
            i += 1

        bases.append(base)
        if base not in HEBREW_LETTERS:
            nikud_ids.append(-100)
            shin_ids.append(-100)
            continue

        vowel = next((m for m in marks if m in VOWELS), "")
        dagesh = DAGESH if DAGESH in marks else ""
        # המודל מקודד דגש+תנועה כמחלקה אחת, בסדר דגש־ואז־תנועה.
        cls = nikud_index.get(_mark_key(dagesh + vowel))
        if cls is None:
            cls = nikud_index.get(_mark_key(vowel), nikud_index.get("", 0))
        # אם/ו/י ללא תנועה כלל — אם קריאה. המודל למד מחלקה נפרדת לזה.
        if not vowel and not dagesh and base in "אוי" and mat_lect_id >= 0:
            cls = mat_lect_id
        nikud_ids.append(cls)

        if base == SHIN:
            dot = next((m for m in marks if m in (SHIN_DOT, SIN_DOT)), SHIN_DOT)
            shin_ids.append(shin_index.get(_mark_key(dot), 0))
        else:
            shin_ids.append(-100)

    return bases, nikud_ids, shin_ids


def apply_labels(text: str, nikud_ids: list[int], shin_ids: list[int],
                 nikud_classes: list[str], shin_classes: list[str],
                 mat_lect_token: str = "<MAT_LECT>") -> str:
    """מרכיב טקסט מנוקד מתחזיות ברמת התו. מובטח שהשלד לא משתנה."""
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch not in HEBREW_LETTERS:
            out.append(ch)
            continue
        mark = nikud_classes[nikud_ids[i]] if 0 <= nikud_ids[i] < len(nikud_classes) else ""
        if mark == mat_lect_token:
            mark = ""
        dot = ""
        if ch == SHIN and 0 <= shin_ids[i] < len(shin_classes):
            dot = shin_classes[shin_ids[i]]
        out.append(ch + dot + mark)
    return normalize("".join(out))


# --- קידוד -------------------------------------------------------------


def encode(text: str, tokenizer) -> list[int]:
    """input_ids ברמת התו. אורך = len(text) + 2 (CLS/SEP)."""
    unk = tokenizer.unk_token_id
    ids = [tokenizer.cls_token_id]
    for ch in text:
        tid = tokenizer.convert_tokens_to_ids(ch)
        ids.append(unk if tid is None else tid)
    ids.append(tokenizer.sep_token_id)
    return ids


def split_for_model(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """חותך טקסט ארוך לחלקים ששומרים על גבולות מילים. השרשור מחזיר את המקור."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        else:
            cut += 1  # המרווח נשאר בסוף החלק, כדי שהשרשור יהיה מדויק
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


# --- ה-backend ---------------------------------------------------------


class DictaBertBackend:
    """השכבה התחתונה של המנוע.

    כרטיס המודל קובע במפורש שהוא אומן על עברית מודרנית ואינו מיועד לשכבות
    מוקדמות — מקראית, רבנית או טרום־מודרנית. לכן הוא כאן התחתית, לא הכול.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None,
                 batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.name = f"dictabert({model_id})"
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        cfg = self.model.config
        self.nikud_classes = list(cfg.nikud_classes)
        self.shin_classes = list(cfg.shin_classes)
        self.mat_lect_token = getattr(cfg, "mat_lect_token", "<MAT_LECT>")

    def vocalize(self, text: str) -> str | None:
        out = self.vocalize_batch([text])
        return out[0]

    def vocalize_batch(self, texts: list[str]) -> list[str | None]:
        torch = self._torch
        # כל טקסט מפורק לחלקים; ממפים חזרה אחרי ההרצה.
        pieces: list[str] = []
        owner: list[int] = []
        for i, t in enumerate(texts):
            t = strip_nikud(normalize(t))
            for chunk in split_for_model(t):
                pieces.append(chunk)
                owner.append(i)

        decoded: list[str] = []
        for start in range(0, len(pieces), self.batch_size):
            batch = pieces[start : start + self.batch_size]
            encoded = [encode(p, self.tokenizer) for p in batch]
            width = max(len(e) for e in encoded)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.tensor(
                [e + [pad] * (width - len(e)) for e in encoded], dtype=torch.long
            )
            attn = torch.tensor(
                [[1] * len(e) + [0] * (width - len(e)) for e in encoded], dtype=torch.long
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    return_dict=True,
                ).logits
            nikud = logits.nikud_logits.argmax(-1).cpu().tolist()
            shin = logits.shin_logits.argmax(-1).cpu().tolist()
            for row, piece in enumerate(batch):
                # התו ה-i בטקסט יושב במקום i+1 בגלל [CLS].
                n_ids = nikud[row][1 : len(piece) + 1]
                s_ids = shin[row][1 : len(piece) + 1]
                decoded.append(
                    apply_labels(piece, n_ids, s_ids, self.nikud_classes,
                                 self.shin_classes, self.mat_lect_token)
                )

        results: list[str | None] = ["" for _ in texts]
        for piece_text, idx in zip(decoded, owner):
            results[idx] = (results[idx] or "") + piece_text
        # אם המודל אכל טקסט — פוסלים את כל התשובה שלו, לא מתקנים אותה.
        for i, (src, got) in enumerate(zip(texts, results)):
            if got is None or identity_diff(strip_nikud(normalize(src)), got):
                results[i] = None
        return results

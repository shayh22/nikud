"""שלב 4 — אימון־המשך.

בסיס: `dicta-il/dictabert-large-char-menaked`, ברישיון CC-BY-4.0.
ראש התיוג ברמת התו, עם מחלקות נפרדות לניקוד ולנקודת שי"ן.

הכללים שהתוכנית מטילה על השלב הזה, ומומשו כאן:

* **להתחיל מריצה קטנה.** `--smoke` מריץ 10% מהקורפוס ו-epoch אחד. אם
  ה-WER לא זז — יש באג בצינור הנתונים, ולא צריך לגלות את זה אחרי 12 שעות.
* **צ'קפוינט כל N צעדים**, הערכה אוטומטית על סט ההערכה המוקפא בכל
  צ'קפוינט, ושמירת הטוב ביותר בלבד.
* **שכחה קטסטרופלית.** `--replay-jsonl` מערבב אחוז מטקסט מודרני מנוקד.
  בלי זה המודל מאבד את היכולת שכבר יש לו, והסקריפט מזהיר על כך במפורש.
* **המשכיות.** `--resume` ממשיך מנקודת עצירה.

סט ההערכה לא נכנס לאימון בשום מצב — `build_corpus.py` כבר סינן אותו,
והסקריפט מאמת זאת שוב לפני שהוא מתחיל.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_eval  # noqa: E402
from build_corpus import load_corpus  # noqa: E402
from dictabert import MODEL_ID, char_labels, encode  # noqa: E402
from hebrew import normalize, strip_nikud  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CHECKPOINTS = ROOT / "models" / "checkpoints"
EVAL_DIR = ROOT / "data" / "eval"

MAX_LEN = 512          # תווים לדוגמה. משפטים ארוכים יותר נחתכים בבניית הקורפוס.


@dataclass
class TrainConfig:
    model_id: str = MODEL_ID
    lr: float = 1e-5               # נמוך בכוונה — אימון־המשך, לא אימון מאפס
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 1
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    eval_every: int = 500
    max_steps: int | None = None
    corpus_fraction: float = 1.0
    replay_fraction: float = 0.15
    freeze_embeddings: bool = True
    seed: int = 20260916


# --- נתונים -------------------------------------------------------------


def load_examples(corpus_dir: Path, *, fraction: float, seed: int,
                  layers: list[str] | None = None) -> list[str]:
    rows = [r["text"] for r in load_corpus(corpus_dir, layers)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if fraction < 1.0:
        rows = rows[: max(1, int(len(rows) * fraction))]
    return rows


def guard_no_leakage(examples: list[str], eval_dir: Path = EVAL_DIR) -> None:
    """הסט המוקפא לא נכנס לאימון. נבדק שוב כאן, לא רק ב-build_corpus."""
    if not build_eval.verify(eval_dir):
        raise SystemExit(
            "סט ההערכה חסר או שה-hash לא תואם. בלי סט מוקפא אין מדידה, "
            "ובלי מדידה אין אימון."
        )
    eval_skeletons = {
        strip_nikud(normalize(r["text"])) for r in build_eval.load(eval_dir)
    }
    overlap = sum(1 for t in examples if strip_nikud(normalize(t)) in eval_skeletons)
    if overlap:
        raise SystemExit(
            f"עצירה: {overlap} משפטים מסט ההערכה נמצאים בסט האימון. "
            "יש לבנות מחדש את הקורפוס עם build_corpus.py."
        )


class Batcher:
    """אצוות ממוינות לפי אורך — פחות ריפוד, יותר צעדים בשנייה."""

    def __init__(self, texts: list[str], tokenizer, nikud_classes, shin_classes,
                 mat_lect_token, batch_size: int, seed: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.nikud_classes = nikud_classes
        self.shin_classes = shin_classes
        self.mat_lect_token = mat_lect_token
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def epoch(self):
        order = sorted(range(len(self.texts)), key=lambda i: len(self.texts[i]))
        batches = [
            order[i : i + self.batch_size]
            for i in range(0, len(order), self.batch_size)
        ]
        self.rng.shuffle(batches)
        for batch in batches:
            yield self._collate([self.texts[i] for i in batch])

    def _collate(self, texts: list[str]):
        import torch

        encoded, nikud_labels, shin_labels = [], [], []
        for gold in texts:
            gold = normalize(gold)[:MAX_LEN]
            plain = strip_nikud(gold)
            _bases, n_ids, s_ids = char_labels(
                gold, self.nikud_classes, self.shin_classes, self.mat_lect_token
            )
            # התוויות נמדדות על התווים של השלד, לא של הטקסט המנוקד.
            if len(n_ids) != len(plain):
                continue
            ids = encode(plain, self.tokenizer)
            encoded.append(ids)
            nikud_labels.append([-100] + n_ids + [-100])
            shin_labels.append([-100] + s_ids + [-100])

        if not encoded:
            return None
        width = max(len(e) for e in encoded)
        pad = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor(
                [e + [pad] * (width - len(e)) for e in encoded], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [[1] * len(e) + [0] * (width - len(e)) for e in encoded],
                dtype=torch.long,
            ),
            "nikud_labels": torch.tensor(
                [x + [-100] * (width - len(x)) for x in nikud_labels], dtype=torch.long
            ),
            "shin_labels": torch.tensor(
                [x + [-100] * (width - len(x)) for x in shin_labels], dtype=torch.long
            ),
        }


# --- אימון --------------------------------------------------------------


def train(cfg: TrainConfig, *, corpus_dir: Path, out_dir: Path,
          replay_jsonl: Path | None, resume: Path | None,
          eval_dir: Path = EVAL_DIR, smoke: bool = False) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    if smoke:
        cfg.corpus_fraction = min(cfg.corpus_fraction, 0.10)
        cfg.epochs = 1
        cfg.eval_every = 100

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    examples = load_examples(corpus_dir, fraction=cfg.corpus_fraction, seed=cfg.seed)
    if not examples:
        raise SystemExit(f"אין קורפוס ב-{corpus_dir}. הרץ build_corpus.py קודם.")
    guard_no_leakage(examples, eval_dir)

    if replay_jsonl and replay_jsonl.exists():
        replay = [
            json.loads(line)["text"]
            for line in replay_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        take = int(len(examples) * cfg.replay_fraction)
        examples += (replay * (take // max(1, len(replay)) + 1))[:take]
        print(f"עורבבו {take} משפטים מודרניים נגד שכחה קטסטרופלית.", file=sys.stderr)
    else:
        print(
            "אזהרה: אין --replay-jsonl. המודל עלול לאבד את יכולת הניקוד "
            "בעברית מודרנית שכבר יש לו (שכחה קטסטרופלית). התוכנית מבקשת "
            "לערבב אחוז מטקסט מודרני מנוקד.",
            file=sys.stderr,
        )

    source = str(resume) if resume else cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModel.from_pretrained(source, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    if cfg.freeze_embeddings:
        for p in model.bert.embeddings.parameters():
            p.requires_grad = False

    nikud_classes = list(model.config.nikud_classes)
    shin_classes = list(model.config.shin_classes)
    mat_lect = getattr(model.config, "mat_lect_token", "<MAT_LECT>")

    batcher = Batcher(examples, tokenizer, nikud_classes, shin_classes, mat_lect,
                      cfg.batch_size, cfg.seed)
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size / cfg.grad_accum)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * cfg.warmup_ratio), total_steps
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {"wer": float("inf"), "step": -1}
    step = 0
    t0 = time.monotonic()

    # MenakedLabels חי במודול הקוד המרוחק של המודל, שנטען דינמית ואין לו
    # שם יבוא יציב. לוקחים אותו מהמודול של המחלקה שנטענה בפועל.
    MenakedLabels = sys.modules[type(model).__module__].MenakedLabels
    print(f"אימון: {len(examples)} דוגמאות · {total_steps} צעדים · {device}",
          file=sys.stderr)

    stop = False
    for epoch in range(cfg.epochs):
        if stop:
            break
        accum = 0
        optim.zero_grad()
        for batch in batcher.epoch():
            if batch is None:
                continue
            labels = MenakedLabels(
                nikud_labels=batch["nikud_labels"].to(device),
                shin_labels=batch["shin_labels"].to(device),
            )
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
                return_dict=True,
            )
            (out.loss / cfg.grad_accum).backward()
            accum += 1
            if accum < cfg.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            accum = 0
            step += 1

            if step % 20 == 0:
                print(
                    f"  epoch {epoch} step {step}/{total_steps} "
                    f"loss={out.loss.item():.4f} "
                    f"{(time.monotonic() - t0) / step:.2f}s/step",
                    file=sys.stderr,
                )

            if step % cfg.eval_every == 0 or step == total_steps:
                wer = _evaluate_checkpoint(model, tokenizer, device, eval_dir)
                history.append({"step": step, "loss": out.loss.item(), "wer": wer})
                print(f"  >> צ'קפוינט {step}: WER={wer:.4f}", file=sys.stderr)
                if wer < best["wer"]:
                    best = {"wer": wer, "step": step}
                    # שומרים את הטוב ביותר בלבד.
                    model.save_pretrained(out_dir / "best")
                    tokenizer.save_pretrained(out_dir / "best")
                _write_state(out_dir, cfg, history, best, step, total_steps)
                model.train()

            if step >= total_steps:
                stop = True
                break

    _write_state(out_dir, cfg, history, best, step, total_steps)
    return {"best": best, "history": history, "steps": step}


def _evaluate_checkpoint(model, tokenizer, device, eval_dir: Path = EVAL_DIR) -> float:
    """WER על סט ההערכה המוקפא. לא על סט האימון."""
    import torch

    from dictabert import apply_labels
    from hebrew import word_matches, words

    model.eval()
    rows = build_eval.load(eval_dir)
    nikud_classes = list(model.config.nikud_classes)
    shin_classes = list(model.config.shin_classes)
    mat_lect = getattr(model.config, "mat_lect_token", "<MAT_LECT>")

    total = errors = 0
    batch_size = 16
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        plains = [strip_nikud(normalize(r["text"]))[:MAX_LEN] for r in chunk]
        encoded = [encode(p, tokenizer) for p in plains]
        width = max(len(e) for e in encoded)
        pad = tokenizer.pad_token_id
        input_ids = torch.tensor(
            [e + [pad] * (width - len(e)) for e in encoded], dtype=torch.long
        ).to(device)
        attn = torch.tensor(
            [[1] * len(e) + [0] * (width - len(e)) for e in encoded], dtype=torch.long
        ).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn,
                           return_dict=True).logits
        nikud = logits.nikud_logits.argmax(-1).cpu().tolist()
        shin = logits.shin_logits.argmax(-1).cpu().tolist()
        for row, (r, plain) in enumerate(zip(chunk, plains)):
            pred = apply_labels(plain, nikud[row][1 : len(plain) + 1],
                                shin[row][1 : len(plain) + 1],
                                nikud_classes, shin_classes, mat_lect)
            gold_w = words(normalize(r["text"]))
            pred_w = words(pred)
            if len(gold_w) != len(pred_w):
                total += len(gold_w)
                errors += len(gold_w)
                continue
            for g, p in zip(gold_w, pred_w):
                total += 1
                if not word_matches(g, p):
                    errors += 1
    return errors / total if total else 1.0


def _write_state(out_dir: Path, cfg: TrainConfig, history, best, step, total) -> None:
    (out_dir / "state.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "history": history,
                "best": best,
                "step": step,
                "total_steps": total,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="אימון־המשך על הקורפוס הרבני")
    p.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    p.add_argument("--out", type=Path, default=CHECKPOINTS)
    p.add_argument("--replay-jsonl", type=Path, default=None,
                   help="טקסט מודרני מנוקד, נגד שכחה קטסטרופלית")
    p.add_argument("--resume", type=Path, default=None, help="המשך מצ'קפוינט")
    p.add_argument("--smoke", action="store_true",
                   help="ריצה קטנה: 10%% מהקורפוס, epoch אחד. להריץ את זה ראשון.")
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--corpus-fraction", type=float, default=1.0)
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = p.parse_args(argv)

    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        corpus_fraction=args.corpus_fraction,
    )
    result = train(cfg, corpus_dir=args.corpus, out_dir=args.out,
                   replay_jsonl=args.replay_jsonl, resume=args.resume,
                   eval_dir=args.eval_dir, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

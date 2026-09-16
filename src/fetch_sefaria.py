"""שלב 2 — משיכה מספריא.

חובות הסעיף: cache מקומי, כיבוד rate limit, המשך מנקודת עצירה, שמירת הגולמי
לפני כל עיבוד, ותיעוד רישיון לכל מקור *בזמן* המשיכה.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import CATALOG, COMMERCIAL_SAFE, Source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
API = "https://www.sefaria.org/api"
UA = "nikud-engine/1.0 (research; contact via repo)"

MIN_INTERVAL = 0.35   # שניות בין בקשות — כיבוד rate limit
MAX_RETRIES = 5


# --- שכבת HTTP עם cache והמשך ------------------------------------------


class Fetcher:
    def __init__(self, raw_dir: Path = RAW, *, min_interval: float = MIN_INTERVAL):
        self.raw = raw_dir
        self.raw.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.min_interval = min_interval
        self._last = 0.0
        self.hits = 0
        self.misses = 0

    def _cache_path(self, kind: str, key: str) -> Path:
        safe = urllib.parse.quote(key, safe="")[:180]
        d = self.raw / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.json"

    def get(self, kind: str, key: str, url: str) -> Any:
        """מחזיר JSON. שומר את הגולמי בדיסק; קריאה שנייה לא יוצאת לרשת."""
        path = self._cache_path(kind, key)
        if path.exists():
            try:
                self.hits += 1
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                path.unlink()  # cache פגום — נמשוך שוב
        data = self._request(url)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.misses += 1
        return data

    def _request(self, url: str) -> Any:
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, timeout=60)
                self._last = time.monotonic()
                if r.status_code == 429:
                    time.sleep(delay + random.random())
                    delay *= 2
                    continue
                if r.status_code == 404:
                    return {"_error": "not_found", "_url": url}
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                self._last = time.monotonic()
                if attempt == MAX_RETRIES - 1:
                    return {"_error": str(exc), "_url": url}
                time.sleep(delay + random.random())
                delay *= 2
        return {"_error": "exhausted", "_url": url}


# --- מבנה הספרים --------------------------------------------------------


def _flatten_shape(node: Any, out: list[dict]) -> None:
    """shape API מחזיר מבנה מקונן לספרים מורכבים. משטחים לעלים."""
    if isinstance(node, list):
        for item in node:
            _flatten_shape(item, out)
        return
    if not isinstance(node, dict):
        return
    chapters = node.get("chapters")
    if isinstance(chapters, list) and chapters and isinstance(chapters[0], dict):
        _flatten_shape(chapters, out)
        return
    if node.get("title"):
        out.append(node)


def list_sections(fetcher: Fetcher, title: str) -> list[str]:
    """מחזיר רשימת refs ברמת הפרק/הדף עבור כותרת או קטגוריה."""
    shape = fetcher.get("shape", title, f"{API}/shape/{urllib.parse.quote(title)}")
    if isinstance(shape, dict) and shape.get("_error"):
        return []
    leaves: list[dict] = []
    _flatten_shape(shape, leaves)

    refs: list[str] = []
    for leaf in leaves:
        book = leaf.get("title") or leaf.get("book")
        chapters = leaf.get("chapters")
        if not book:
            continue
        if isinstance(chapters, int):
            # מבנה שטוח בן רמה אחת — הספר כולו הוא ref אחד.
            refs.append(book)
        elif isinstance(chapters, list):
            if _is_talmud(book):
                # מסכתות ממוספרות בדפים: 2a, 2b, 3a...
                for i in range(len(chapters)):
                    daf = 2 + i // 2
                    side = "a" if i % 2 == 0 else "b"
                    refs.append(f"{book} {daf}{side}")
            else:
                for i in range(len(chapters)):
                    refs.append(f"{book} {i + 1}")
        else:
            refs.append(book)
    return refs


def _is_talmud(book: str) -> bool:
    return book.startswith("Jerusalem Talmud") is False and book in _BAVLI_TRACTATES


_BAVLI_TRACTATES = {
    "Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah", "Yoma",
    "Sukkah", "Beitzah", "Taanit", "Megillah", "Moed Katan", "Chagigah",
    "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
    "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot",
    "Shevuot", "Avodah Zarah", "Horayot", "Zevachim", "Menachot", "Chullin",
    "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah", "Tamid", "Niddah",
}


# --- משיכת טקסט --------------------------------------------------------


@dataclass
class FetchedSection:
    ref: str
    source_key: str
    layer: str
    version_title: str
    license: str
    version_source: str
    lines: list[str]


def _pick_version(versions: list[dict], source: Source) -> dict | None:
    if not versions:
        return None
    for wanted in source.prefer_versions:
        for v in versions:
            if wanted.lower() in (v.get("versionTitle") or "").lower():
                return v
    # אחרת: המהדורה העברית הראשונה שיש בה ניקוד.
    for v in versions:
        if (v.get("languageFamilyName") or v.get("language")) in ("hebrew", "he"):
            return v
    return versions[0]


def _flatten_text(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        if node.strip():
            out.append(node)
    elif isinstance(node, list):
        for item in node:
            _flatten_text(item, out)


def fetch_section(fetcher: Fetcher, ref: str, source: Source) -> FetchedSection | None:
    url = (
        f"{API}/v3/texts/{urllib.parse.quote(ref)}"
        "?version=hebrew&return_format=text_only"
    )
    data = fetcher.get("texts", ref, url)
    if not isinstance(data, dict) or data.get("_error"):
        return None
    version = _pick_version(data.get("versions") or [], source)
    if not version:
        return None
    lines: list[str] = []
    _flatten_text(version.get("text"), lines)
    if not lines:
        return None
    return FetchedSection(
        ref=data.get("ref") or ref,
        source_key=source.key,
        layer=source.layer,
        version_title=version.get("versionTitle") or "",
        license=version.get("license") or "unknown",
        version_source=version.get("versionSource") or "",
        lines=lines,
    )


# --- רישיונות -----------------------------------------------------------


class LicenseLog:
    """נכתב בזמן המשיכה, לא אחריה."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: dict[tuple[str, str], dict] = {}
        json_path = path.with_suffix(".json")
        self.json_path = json_path
        if json_path.exists():
            for e in json.loads(json_path.read_text(encoding="utf-8")):
                self.entries[(e["source"], e["version_title"])] = e

    def record(self, sec: FetchedSection) -> None:
        key = (sec.source_key, sec.version_title)
        entry = self.entries.get(key)
        if entry is None:
            src = CATALOG[sec.source_key]
            self.entries[key] = {
                "source": sec.source_key,
                "title": src.title,
                "layer": src.layer,
                "version_title": sec.version_title,
                "license": sec.license,
                "version_source": sec.version_source,
                "commercial_ok": src.commercial_ok,
                "sections": 1,
                "example_ref": sec.ref,
            }
        else:
            entry["sections"] += 1

    def flush(self) -> None:
        entries = sorted(self.entries.values(), key=lambda e: (e["source"], e["version_title"]))
        self.json_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows = [
            "# רישיונות המקורות",
            "",
            "נכתב אוטומטית על ידי `src/fetch_sefaria.py` בזמן המשיכה.",
            "",
            "| מקור | מהדורה | רישיון | מסחרי? | מקטעים | דוגמה |",
            "|---|---|---|---|---|---|",
        ]
        for e in entries:
            ok = "כן" if e["commercial_ok"] else "**לא**"
            rows.append(
                f"| {e['title']} | {e['version_title']} | {e['license']} | "
                f"{ok} | {e['sections']} | `{e['example_ref']}` |"
            )
        nc = [e for e in entries if not e["commercial_ok"]]
        if nc:
            rows += [
                "",
                "## אזהרה",
                "",
                "המקורות הבאים אינם מתאימים למוצר או שירות מסחרי:",
                "",
            ]
            rows += [f"- **{e['title']}** ({e['version_title']}) — {e['license']}" for e in nc]
            rows += [
                "",
                "לבניית מסלול מסחרי: `python src/fetch_sefaria.py --commercial-safe`.",
            ]
        self.path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- ריצה ---------------------------------------------------------------


def run(source_keys: Iterable[str], *, limit_sections: int | None, out: Path,
        min_interval: float = MIN_INTERVAL) -> int:
    fetcher = Fetcher(min_interval=min_interval)
    licenses = LicenseLog(ROOT / "data" / "LICENSES.md")
    state_path = RAW / "_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    done: set[str] = set(state.get("done", []))

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("a", encoding="utf-8") as fh:
        for key in source_keys:
            source = CATALOG[key]
            refs = list_sections(fetcher, source.title)
            if limit_sections:
                refs = refs[:limit_sections]
            print(f"[{key}] {len(refs)} מקטעים", file=sys.stderr)
            for i, ref in enumerate(refs):
                marker = f"{key}::{ref}"
                if marker in done:
                    continue
                sec = fetch_section(fetcher, ref, source)
                if sec is not None:
                    licenses.record(sec)
                    fh.write(
                        json.dumps(
                            {
                                "ref": sec.ref,
                                "source": sec.source_key,
                                "layer": sec.layer,
                                "version": sec.version_title,
                                "license": sec.license,
                                "lines": sec.lines,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                done.add(marker)
                if i % 25 == 0:
                    fh.flush()
                    licenses.flush()
                    state_path.write_text(
                        json.dumps({"done": sorted(done)}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(
                        f"  {i + 1}/{len(refs)}  cache={fetcher.hits} net={fetcher.misses}",
                        file=sys.stderr,
                    )
    licenses.flush()
    state_path.write_text(json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8")
    print(f"נכתבו {written} מקטעים ל-{out}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="משיכת טקסטים מנוקדים מספריא")
    p.add_argument("--sources", nargs="*", default=None, help="מפתחות מהקטלוג")
    p.add_argument(
        "--commercial-safe",
        action="store_true",
        help="רק מקורות שמותרים גם למוצר מסחרי (בלי CC-BY-NC)",
    )
    p.add_argument("--limit-sections", type=int, default=None,
                   help="תקרת מקטעים לכל מקור — לריצת בדיקה")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "sections.jsonl")
    p.add_argument("--min-interval", type=float, default=MIN_INTERVAL)
    args = p.parse_args(argv)

    if args.sources:
        keys = args.sources
    elif args.commercial_safe:
        keys = list(COMMERCIAL_SAFE)
    else:
        keys = list(CATALOG)
    unknown = [k for k in keys if k not in CATALOG]
    if unknown:
        p.error(f"מקורות לא מוכרים: {unknown}. קיימים: {list(CATALOG)}")
    run(keys, limit_sections=args.limit_sections, out=args.out,
        min_interval=args.min_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

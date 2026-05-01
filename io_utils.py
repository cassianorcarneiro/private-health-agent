# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# I/O Utilities: Exam reading, PII sanitization, and source ranking.
# Kept separate from agent.py to remain independently testable.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
PDF_EXTS = {".pdf"}
SHEET_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}


# ---------- PDF and Spreadsheet Reading ----------------------------------------------------------

def read_pdf(path: Path) -> str:
    """Extracts text and tables from PDF. Tries pdfplumber first, falls back to pypdf."""
    try:
        import pdfplumber
        chunks: List[str] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                table_text_parts: List[str] = []
                for t in tables:
                    for row in t:
                        clean = [str(c).strip() if c is not None else "" for c in row]
                        if any(clean):
                            table_text_parts.append(" | ".join(clean))
                section = f"--- Page {i} ---\n{text}".strip()
                if table_text_parts:
                    section += "\n[Tables]\n" + "\n".join(table_text_parts)
                chunks.append(section)
        return "\n\n".join(chunks).strip()
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n\n".join(
                f"--- Page {i+1} ---\n{(p.extract_text() or '').strip()}"
                for i, p in enumerate(reader.pages)
            ).strip()
        except Exception as e:
            return f"(Failed to read PDF {path.name}: {e})"


def read_sheet(path: Path) -> str:
    """Reads CSV/TSV/XLSX. Returns readable tabular text."""
    try:
        import pandas as pd
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            xls = pd.read_excel(path, sheet_name=None)
            parts = []
            for name, df in xls.items():
                parts.append(f"--- Sheet: {name} ---\n{df.to_string(index=False)}")
            return "\n\n".join(parts)
        return df.to_string(index=False)
    except Exception as e:
        return f"(Failed to read spreadsheet {path.name}: {e})"


# ---------- Loading with base64 cache (avoids re-reading images every turn) ------------------------

class ExamLoader:
    """Accumulates loaded exams throughout the session. Caches base64 of images."""

    def __init__(self, max_chars: int = 20000):
        self.max_chars = max_chars
        self.text_parts: List[str] = []
        self.image_paths: List[str] = []
        self._image_b64_cache: Dict[str, str] = {}

    def add(self, paths: List[str]) -> Dict[str, Any]:
        added_text_chars = 0
        added_images = 0
        missing: List[str] = []

        for raw in paths:
            p = Path(raw).expanduser()
            if not p.exists():
                missing.append(str(p))
                continue
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                resolved = str(p.resolve())
                if resolved not in self.image_paths:
                    self.image_paths.append(resolved)
                    # Warm up base64 cache
                    b64 = self._encode_image_b64(resolved)
                    if b64:
                        self._image_b64_cache[resolved] = b64
                        added_images += 1
            elif ext in PDF_EXTS:
                text = read_pdf(p)
                self.text_parts.append(f"### File: {p.name}\n{text}")
                added_text_chars += len(text)
            elif ext in SHEET_EXTS:
                text = read_sheet(p)
                self.text_parts.append(f"### File: {p.name}\n{text}")
                added_text_chars += len(text)
            else:
                missing.append(f"{p} (unsupported extension)")

        return {
            "added_text_chars": added_text_chars,
            "added_images": added_images,
            "missing": missing,
        }

    def clear(self):
        self.text_parts.clear()
        self.image_paths.clear()
        self._image_b64_cache.clear()

    @property
    def text(self) -> str:
        full = "\n\n".join(self.text_parts).strip()
        if len(full) > self.max_chars:
            full = full[: self.max_chars] + "\n\n[...text truncated due to context limit...]"
        return full

    def images_base64(self) -> List[str]:
        """Returns pre-computed cache without re-reading from disk."""
        return [
            b64 for p in self.image_paths
            if (b64 := self._image_b64_cache.get(p))
        ]

    def has_any(self) -> bool:
        return bool(self.text_parts) or bool(self.image_paths)

    @staticmethod
    def _encode_image_b64(path: str) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None


# ---------- PII Sanitization before sending to planner ---------------------------------------

# Conservative patterns. The goal isn't strict legal compliance (GDPR/HIPAA),
# but reducing the risk of the planner copying identifiable snippets into search queries.

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Brazilian CPF (XXX.XXX.XXX-XX) and variations
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[TAX_ID]"),
    # Approximate Brazilian RG
    (re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b"), "[RG]"),
    # Common date formats
    (re.compile(r"\b\d{2}/\d{2}/\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    # Phone numbers (with or without parentheses)
    (re.compile(r"\(?\d{2}\)?[\s-]?9?\d{4}-?\d{4}"), "[PHONE]"),
    # E-mails
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # Long-form medical record/protocol/registration numbers
    (re.compile(r"\b(?:record|protocol|registration|prontu[áa]rio)[\s#:]*\d{4,}\b", re.IGNORECASE), "[ID]"),
]


def sanitize_pii(text: str) -> str:
    """Replaces common Brazilian PII patterns with placeholders."""
    if not text:
        return text
    out = text
    for pattern, repl in _PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


# ---------- Search Result Re-ranking ----------------------------------------------------

def domain_of(url: str) -> str:
    """Extracts the effective domain from a URL without external dependencies."""
    if not url:
        return ""
    m = re.match(r"^https?://([^/]+)", url)
    if not m:
        return ""
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def rank_sources(
    results: List[Dict[str, Any]],
    preferred_domains: List[str],
    max_items: int,
) -> List[Dict[str, Any]]:
    """
    Reorders results: trusted sources (substring match in domain) come first.
    Maintains relative order within each bucket.
    """
    preferred_lower = [d.lower() for d in preferred_domains]

    def score(item: Dict[str, Any]) -> int:
        if item.get("error"):
            return -1
        url = item.get("url") or item.get("href") or ""
        host = domain_of(url)
        if not host:
            return 0
        for d in preferred_lower:
            if d in host:
                return 2
        return 1

    # Annotate score, stable sort by -score
    annotated = [(score(r), i, r) for i, r in enumerate(results)]
    annotated.sort(key=lambda t: (-t[0], t[1]))
    ranked = [r for s, _, r in annotated if s >= 0]
    return ranked[:max_items]


def filter_sources_by_intent(
    sources: List[Dict[str, Any]],
    intents_allowed: List[str],
) -> List[Dict[str, Any]]:
    """Filters sources by the 'intent' field annotated when the query was run."""
    return [s for s in sources if s.get("intent") in intents_allowed]


def summarize_sources(sources: List[Dict[str, Any]], max_items: int) -> str:
    """Formats sources into compact text for inclusion in prompts."""
    lines: List[str] = []
    for item in sources[:max_items]:
        if item.get("error"):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or item.get("href") or "").strip()
        body = (item.get("body") or item.get("snippet") or item.get("content") or "").strip()
        if not (title or url or body):
            continue
        lines.append(f"- {title}\n  {url}\n  {body[:400]}")
    return "\n".join(lines) if lines else "(No useful sources returned.)"
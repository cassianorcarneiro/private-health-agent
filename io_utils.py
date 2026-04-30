# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Utilitários de I/O: leitura de exames, sanitização de PII, ranking de fontes.
# Mantidos fora do agent.py para serem testáveis isoladamente.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
PDF_EXTS = {".pdf"}
SHEET_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}


# ---------- Leitura de PDFs e planilhas ----------------------------------------------------------

def read_pdf(path: Path) -> str:
    """Extrai texto e tabelas de PDF. pdfplumber primeiro, fallback para pypdf."""
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
                section = f"--- Página {i} ---\n{text}".strip()
                if table_text_parts:
                    section += "\n[Tabelas]\n" + "\n".join(table_text_parts)
                chunks.append(section)
        return "\n\n".join(chunks).strip()
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n\n".join(
                f"--- Página {i+1} ---\n{(p.extract_text() or '').strip()}"
                for i, p in enumerate(reader.pages)
            ).strip()
        except Exception as e:
            return f"(Falha ao ler PDF {path.name}: {e})"


def read_sheet(path: Path) -> str:
    """Lê CSV/TSV/XLSX. Retorna texto tabular legível."""
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
                parts.append(f"--- Aba: {name} ---\n{df.to_string(index=False)}")
            return "\n\n".join(parts)
        return df.to_string(index=False)
    except Exception as e:
        return f"(Falha ao ler planilha {path.name}: {e})"


# ---------- Loading com cache de base64 (evita reler imagem a cada turno) ------------------------

class ExamLoader:
    """Acumula exames carregados ao longo da sessão. Cacheia base64 das imagens."""

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
                    # Pré-aquece cache de base64
                    b64 = self._encode_image_b64(resolved)
                    if b64:
                        self._image_b64_cache[resolved] = b64
                        added_images += 1
            elif ext in PDF_EXTS:
                text = read_pdf(p)
                self.text_parts.append(f"### Arquivo: {p.name}\n{text}")
                added_text_chars += len(text)
            elif ext in SHEET_EXTS:
                text = read_sheet(p)
                self.text_parts.append(f"### Arquivo: {p.name}\n{text}")
                added_text_chars += len(text)
            else:
                missing.append(f"{p} (extensão não suportada)")

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
            full = full[: self.max_chars] + "\n\n[...texto truncado por limite de contexto...]"
        return full

    def images_base64(self) -> List[str]:
        """Retorna o cache pré-computado, sem reler do disco."""
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


# ---------- Sanitização de PII antes de mandar pro planner ---------------------------------------

# Padrões conservadores. O objetivo não é compliance LGPD/HIPAA — é reduzir o
# risco de o planner copiar literalmente trechos identificáveis para queries.

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # CPF (XXX.XXX.XXX-XX) e variações
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    # RG aproximado
    (re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b"), "[RG]"),
    # Datas em formatos comuns
    (re.compile(r"\b\d{2}/\d{2}/\d{2,4}\b"), "[DATA]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATA]"),
    # Telefones (com ou sem parênteses); usa lookbehind/lookahead leves
    (re.compile(r"\(?\d{2}\)?[\s-]?9?\d{4}-?\d{4}"), "[TEL]"),
    # E-mails
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # Números de prontuário/protocolo de aparência longa
    (re.compile(r"\b(?:prontu[áa]rio|protocolo|registro)[\s#:]*\d{4,}\b", re.IGNORECASE), "[ID]"),
]


def sanitize_pii(text: str) -> str:
    """Substitui padrões comuns de PII brasileiros por placeholders."""
    if not text:
        return text
    out = text
    for pattern, repl in _PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


# ---------- Re-ranking de resultados de busca ----------------------------------------------------

def domain_of(url: str) -> str:
    """Extrai o domínio efetivo de uma URL, sem dependências externas."""
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
    Reordena resultados: fontes confiáveis (substring match em domínio) vêm primeiro.
    Mantém a ordem relativa dentro de cada bucket.
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

    # Anota score, ordena estável por -score
    annotated = [(score(r), i, r) for i, r in enumerate(results)]
    annotated.sort(key=lambda t: (-t[0], t[1]))
    ranked = [r for s, _, r in annotated if s >= 0]
    return ranked[:max_items]


def filter_sources_by_intent(
    sources: List[Dict[str, Any]],
    intents_allowed: List[str],
) -> List[Dict[str, Any]]:
    """Filtra fontes pelo campo 'intent' anotado quando a query foi rodada."""
    return [s for s in sources if s.get("intent") in intents_allowed]


def summarize_sources(sources: List[Dict[str, Any]], max_items: int) -> str:
    """Formata as fontes em texto compacto para entrar nos prompts."""
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
    return "\n".join(lines) if lines else "(Nenhuma fonte útil retornada.)"

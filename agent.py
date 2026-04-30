# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE HEALTH AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import os
import re
import json
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from operator import add

from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import ollama

from config import Config

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Disclaimer (impresso ao iniciar o programa)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

DISCLAIMER_PT = """\
[bold red]⚠️  AVISO IMPORTANTE — LEIA ANTES DE USAR ⚠️[/bold red]

Este agente é uma ferramenta [bold]experimental[/bold] de apoio educacional,
construída sobre modelos de linguagem (LLMs) — incluindo o MedGemma do Google,
que é um [bold]modelo de pesquisa[/bold], NÃO um dispositivo médico.

[bold yellow]ESTE AGENTE NÃO:[/bold yellow]
  • substitui um médico, farmacêutico ou qualquer profissional de saúde;
  • emite diagnósticos clínicos válidos;
  • prescreve medicamentos, doses ou condutas terapêuticas;
  • foi avaliado por nenhum órgão regulatório (ANVISA, FDA, EMA, etc.);
  • garante precisão, completude ou atualidade das informações.

[bold yellow]RISCOS CONHECIDOS:[/bold yellow]
  • LLMs podem [bold]alucinar[/bold] (inventar fatos, valores de referência,
    interações medicamentosas e diagnósticos com aparência plausível);
  • a interpretação de exames depende de contexto clínico que o modelo
    não tem (história, exame físico, comorbidades, medicações em uso);
  • valores de referência variam por laboratório, idade, sexo e método;
  • buscas na web podem trazer fontes desatualizadas ou de baixa qualidade.

[bold yellow]COMO USAR COM SEGURANÇA:[/bold yellow]
  • Trate as respostas como [bold]hipóteses para discutir com seu médico[/bold].
  • Em sintomas de alarme (dor torácica, falta de ar, déficit neurológico,
    sangramento, febre alta persistente, etc.) procure atendimento [bold]agora[/bold].
  • Não interrompa nem inicie medicamentos com base nesta ferramenta.
  • Não compartilhe dados de pacientes de terceiros sem consentimento.

[bold yellow]PRIVACIDADE:[/bold yellow]
  • Os modelos rodam localmente via Ollama; o conteúdo dos seus exames
    [bold]não sai do seu computador[/bold] — exceto quando a busca web
    estiver ligada (apenas as queries são enviadas ao buscador).

Ao continuar, você declara que entendeu estes limites e usará a
ferramenta por sua conta e risco, apenas para fins pessoais e educacionais.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Graph state
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class AgentState(TypedDict):
    history: List[Dict[str, str]]

    question: str
    use_web_search: bool

    # Conteúdo extraído de exames (PDF/planilha) e caminhos de imagens.
    exam_text: str
    image_paths: List[str]

    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    drafts: Annotated[List[str], add]
    final_answer: str


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Schemas
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class SearchPlan(BaseModel):
    queries: List[str] = Field(..., description="Short medical web-search queries (3 to 6).")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Helpers para leitura de exames
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
PDF_EXTS = {".pdf"}
SHEET_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}


def _read_pdf(path: Path) -> str:
    """Extrai texto de PDF usando pdfplumber (melhor para tabelas) com fallback para pypdf."""
    try:
        import pdfplumber
        chunks = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # Tabelas — comum em exames laboratoriais
                tables = page.extract_tables() or []
                table_text = ""
                for t in tables:
                    for row in t:
                        clean = [str(c).strip() if c is not None else "" for c in row]
                        table_text += " | ".join(clean) + "\n"
                chunks.append(f"--- Página {i} ---\n{text}\n{table_text}".strip())
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


def _read_sheet(path: Path) -> str:
    """Lê CSV/TSV/XLSX e retorna como texto tabular legível."""
    try:
        import pandas as pd
        ext = path.suffix.lower()
        if ext in {".csv"}:
            df = pd.read_csv(path)
        elif ext in {".tsv"}:
            df = pd.read_csv(path, sep="\t")
        else:
            # Lê todas as abas
            xls = pd.read_excel(path, sheet_name=None)
            parts = []
            for name, df in xls.items():
                parts.append(f"--- Aba: {name} ---\n{df.to_string(index=False)}")
            return "\n\n".join(parts)
        return df.to_string(index=False)
    except Exception as e:
        return f"(Falha ao ler planilha {path.name}: {e})"


def load_exam_files(paths: List[str], max_chars: int) -> Dict[str, Any]:
    """
    Recebe uma lista de caminhos e separa em (texto extraído de PDFs/planilhas)
    e (caminhos de imagens) que serão enviadas ao modelo de visão.
    """
    text_parts: List[str] = []
    image_paths: List[str] = []
    missing: List[str] = []

    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            missing.append(str(p))
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            image_paths.append(str(p.resolve()))
        elif ext in PDF_EXTS:
            text_parts.append(f"### Arquivo: {p.name}\n{_read_pdf(p)}")
        elif ext in SHEET_EXTS:
            text_parts.append(f"### Arquivo: {p.name}\n{_read_sheet(p)}")
        else:
            missing.append(f"{p} (extensão não suportada)")

    full_text = "\n\n".join(text_parts).strip()
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[...texto truncado por limite de contexto...]"

    return {"text": full_text, "images": image_paths, "missing": missing}


def _encode_image_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Core class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@dataclass
class HealthMultiAgent:

    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.history: List[Dict[str, str]] = []

        # Exames carregados pelo comando /load — persistem entre perguntas
        # até o usuário rodar /clear.
        self.loaded_exam_text: str = ""
        self.loaded_image_paths: List[str] = []

        self._check_model()
        self.app = self.build_graph()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Check model
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def _check_model(self):
        try:
            models_response = ollama.list()
            model_details = []

            if hasattr(models_response, "models") and models_response.models:
                for model in models_response.models:
                    model_details.append({
                        "name": model.model,
                        "size": model.size,
                        "modified": model.modified_at,
                        "parameters": getattr(model.details, "parameter_size", "N/A") if model.details else "N/A",
                    })

            if not model_details:
                self.console.print("❌ [red]Nenhum modelo encontrado no Ollama.[/red]")
                self.console.print("   Instale com: [cyan]ollama pull medgemma:4b[/cyan]")
                raise RuntimeError("No models available")

            # Resolve modelo de texto
            self.config.ollama_model = self._resolve_model(
                self.config.ollama_model, model_details, label="texto"
            )
            # Resolve modelo de visão
            self.config.ollama_vision_model = self._resolve_model(
                self.config.ollama_vision_model, model_details, label="visão"
            )

        except Exception as e:
            self.console.print(f"❌ Erro ao conectar ao Ollama: {e}", style="bold red")
            self.console.print("\n🔧 [yellow]Possíveis soluções:[/yellow]")
            self.console.print("1. Verifique se o Ollama está rodando: [cyan]ollama serve[/cyan]")
            self.console.print("2. Instale o MedGemma:                [cyan]ollama pull medgemma:4b[/cyan]")
            raise

    def _resolve_model(self, requested: str, model_details: list, label: str) -> str:
        """Acha o modelo configurado na lista local do Ollama; senão, usa fallback."""
        match = [m for m in model_details if requested.lower() in m["name"].lower()]
        if match:
            chosen = match[0]
            self.console.print(Panel(
                f"✅ [green]Modelo de {label}:[/green] {chosen['name']}\n"
                f"📊 [cyan]Tamanho:[/cyan] {chosen['size']/1024/1024/1024:.1f} GB\n"
                f"⚙️  [yellow]Parâmetros:[/yellow] {chosen['parameters']}",
                title=f"🩺 Modelo ({label})",
                border_style="green",
            ))
            return chosen["name"]
        else:
            chosen = model_details[0]
            self.console.print(Panel(
                f"⚠️  [yellow]'{requested}' não encontrado.[/yellow]\n"
                f"Usando fallback: [bold]{chosen['name']}[/bold]\n"
                f"[dim]Para melhores resultados em saúde:[/dim] [cyan]ollama pull medgemma:4b[/cyan]",
                title=f"🩺 Modelo ({label}) — fallback",
                border_style="yellow",
            ))
            return chosen["name"]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # LLM sessions
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def _llm(self, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=temperature,
        )

    def _vision_invoke(self, prompt: str, image_paths: List[str], temperature: float) -> str:
        """
        Chama o modelo de visão (MedGemma) usando o cliente nativo do Ollama,
        que aceita imagens em base64. Usado quando há imagens carregadas.
        """
        images_b64 = [b for b in (_encode_image_b64(p) for p in image_paths) if b]
        if not images_b64:
            # Sem imagens válidas — degrada para texto.
            return self._llm(temperature).invoke(prompt).content.strip()
        try:
            resp = ollama.chat(
                model=self.config.ollama_vision_model,
                messages=[{"role": "user", "content": prompt, "images": images_b64}],
                options={"temperature": temperature},
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            return f"(Falha ao processar imagens com modelo de visão: {e})"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Utilities
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    @staticmethod
    def _safe_json_extract(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l:r + 1])
        raise ValueError("Could not parse JSON from model output.")

    @staticmethod
    def _summarize_sources(results: List[Dict[str, Any]], max_items: int) -> str:
        lines: List[str] = []
        n = 0
        for item in results:
            if n >= max_items:
                break
            if item.get("error"):
                continue
            title = (item.get("title") or "").strip()
            url = (item.get("url") or item.get("href") or "").strip()
            body = (item.get("body") or item.get("snippet") or item.get("content") or "").strip()
            if not (title or url or body):
                continue
            lines.append(f"- {title}\n  {url}\n  {body[:400]}")
            n += 1
        return "\n".join(lines) if lines else "(Nenhuma fonte útil retornada.)"

    def _history_block(self, max_turns: int = 6) -> str:
        recent = self.history[-2 * max_turns:]
        out = [f"{m['role'].upper()}: {m['content']}" for m in recent]
        return "\n".join(out) if out else "(sem contexto prévio)"

    def _exam_block(self, state: AgentState) -> str:
        text = state.get("exam_text", "")
        imgs = state.get("image_paths", [])
        if not text and not imgs:
            return "(nenhum exame carregado)"
        parts = []
        if text:
            parts.append(f"[TEXTO EXTRAÍDO DOS EXAMES]\n{text}")
        if imgs:
            parts.append(f"[IMAGENS ANEXADAS] {len(imgs)} arquivo(s): " +
                         ", ".join(Path(p).name for p in imgs))
        return "\n\n".join(parts)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Graph nodes
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def node_plan_search(self, state: AgentState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [DEBUG] node_plan_search...[/dim cyan]")

        if not state.get("use_web_search", True):
            return {"search_queries": [], "search_results": [], "drafts": [], "final_answer": ""}

        llm = self._llm(self.config.temperature_planner)
        preferred = ", ".join(self.config.preferred_medical_sources[:8])

        prompt = (
            "Você é um planejador de buscas médicas. Gere de 3 a 6 queries CURTAS, em inglês ou português, "
            "para responder à pergunta do usuário usando fontes confiáveis "
            f"(preferencialmente: {preferred}).\n"
            "Inclua, quando relevante: nome de doença, sintoma, exame ou medicamento; "
            "valores de referência; interações; diretrizes (guidelines).\n"
            'Retorne APENAS JSON: {"queries":["..."]}\n\n'
            f"Histórico recente:\n{self._history_block()}\n\n"
            f"Resumo dos exames carregados:\n{(state.get('exam_text','')[:1500] or '(nenhum)')}\n\n"
            f"Pergunta do usuário:\n{state['question']}\n"
        )

        raw = llm.invoke(prompt).content
        try:
            data = self._safe_json_extract(raw)
            plan = SearchPlan.model_validate(data)
            queries = [q.strip() for q in plan.queries if q.strip()][: self.config.max_queries]
            if not queries:
                queries = [state["question"][:120]]
        except (ValueError, ValidationError):
            queries = [state["question"][:120]]

        return {"search_queries": queries, "search_results": [], "drafts": [], "final_answer": ""}

    def node_web_search(self, state: AgentState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [DEBUG] node_web_search...[/dim cyan]")

        if not state.get("use_web_search", True) or not state["search_queries"]:
            return {"search_results": []}

        out: List[Dict[str, Any]] = []
        for q in state["search_queries"]:
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(q, max_results=self.config.ddgs_max_results_per_query)
                    for r in results:
                        out.append({"query": q, **r} if isinstance(r, dict) else {"query": q, "raw": r})
            except Exception as e:
                out.append({"query": q, "error": str(e)})
        return {"search_results": out}

    # ----- Agentes especializados ---------------------------------------------------------------

    def node_exam_analyst(self, state: AgentState) -> Dict[str, Any]:
        """
        Agente 1: ANALISTA DE EXAMES.
        Lê o conteúdo extraído de PDFs/planilhas e, se houver imagens, usa o
        modelo de visão (MedGemma) para interpretá-las.
        """
        self.console.print("[dim cyan]>> [DEBUG] Exam Analyst...[/dim cyan]")

        if not state.get("exam_text") and not state.get("image_paths"):
            return {"drafts": ["[Analista de Exames]\n(Nenhum exame foi anexado nesta conversa.)"]}

        sources = self._summarize_sources(state["search_results"], 6) if state.get("use_web_search") else "(busca web desativada)"

        prompt = (
            "Você é um ANALISTA DE EXAMES médicos. Sua tarefa é, com base nos dados abaixo, "
            "extrair achados clinicamente relevantes de forma estruturada.\n\n"
            "Para CADA achado relevante, indique: parâmetro, valor, unidade, valor de referência (se presente), "
            "se está alterado (alto/baixo/normal/indeterminado), e o que isso costuma sugerir.\n"
            "Se houver imagens (radiografia, dermatoscopia, fundo de olho, lâmina), descreva achados visíveis "
            "em linguagem médica, mas SEM emitir diagnóstico definitivo.\n"
            "NÃO INVENTE valores de referência: se não estiverem no exame, diga 'referência não informada'.\n"
            "Termine com 1-3 frases sobre o quadro geral sugerido pelos exames.\n\n"
            f"Pergunta do usuário:\n{state['question']}\n\n"
            f"Exames:\n{self._exam_block(state)}\n\n"
            f"Fontes da web (se houver):\n{sources}\n"
        )

        if state.get("image_paths"):
            draft = self._vision_invoke(prompt, state["image_paths"], self.config.temperature_drafters)
        else:
            draft = self._llm(self.config.temperature_drafters).invoke(prompt).content.strip()

        return {"drafts": [f"[Analista de Exames]\n{draft}"]}

    def node_clinical_reasoner(self, state: AgentState) -> Dict[str, Any]:
        """
        Agente 2: RACIOCÍNIO CLÍNICO / DIAGNÓSTICOS DIFERENCIAIS.
        Sintomas + achados → lista de hipóteses, ordenadas por probabilidade,
        com bandeiras vermelhas (red flags).
        """
        self.console.print("[dim cyan]>> [DEBUG] Clinical Reasoner...[/dim cyan]")

        sources = self._summarize_sources(state["search_results"], self.config.max_sources_in_prompt) \
            if state.get("use_web_search") else "(busca web desativada)"

        prompt = (
            "Você é um agente de RACIOCÍNIO CLÍNICO. Com base na pergunta, no histórico, "
            "nos exames (se houver) e nas fontes, produza:\n"
            "1) Resumo do problema em 2-3 linhas;\n"
            "2) Lista de DIAGNÓSTICOS DIFERENCIAIS (3 a 6), ordenados por probabilidade, "
            "cada um com: nome da hipótese, achados que apoiam, achados que contrariam, "
            "exames que ajudariam a confirmar/excluir;\n"
            "3) RED FLAGS (sinais que exigem avaliação médica imediata) — sempre liste se aplicável;\n"
            "4) O que NÃO dá para concluir só com os dados disponíveis.\n\n"
            "REGRAS: nunca diga 'você tem X'. Use 'compatível com', 'sugestivo de', 'a considerar'. "
            "Marque incertezas explicitamente.\n\n"
            f"Histórico:\n{self._history_block()}\n\n"
            f"Pergunta:\n{state['question']}\n\n"
            f"Exames:\n{self._exam_block(state)}\n\n"
            f"Fontes:\n{sources}\n"
        )

        draft = self._llm(self.config.temperature_drafters).invoke(prompt).content.strip()
        return {"drafts": [f"[Raciocínio Clínico]\n{draft}"]}

    def node_pharma_advisor(self, state: AgentState) -> Dict[str, Any]:
        """
        Agente 3: MEDICAMENTOS, INTERAÇÕES, CONDUTA GERAL.
        Quando a pergunta envolver fármacos, foca em mecanismo, classes,
        efeitos adversos comuns/graves, interações, contraindicações.
        Quando NÃO envolver, foca em medidas não-farmacológicas e quando procurar atendimento.
        """
        self.console.print("[dim cyan]>> [DEBUG] Pharma Advisor...[/dim cyan]")

        sources = self._summarize_sources(state["search_results"], self.config.max_sources_in_prompt) \
            if state.get("use_web_search") else "(busca web desativada)"

        prompt = (
            "Você é um agente de informações sobre MEDICAMENTOS e CONDUTA GERAL.\n"
            "Se a pergunta/contexto envolver um ou mais fármacos, descreva para CADA um:\n"
            "  • classe terapêutica e mecanismo;\n"
            "  • indicações comuns;\n"
            "  • efeitos adversos frequentes e GRAVES (separe os dois);\n"
            "  • interações medicamentosas relevantes;\n"
            "  • contraindicações principais e populações de risco (gestantes, idosos, hepato/nefropatas).\n"
            "Se NÃO envolver fármacos, traga medidas não-farmacológicas, autocuidado e critérios "
            "claros de quando procurar pronto-atendimento.\n"
            "NUNCA forneça doses específicas como prescrição. Se citar dose, deixe claro que é "
            "informação genérica de bula e que a dose individual deve ser definida pelo médico.\n"
            "Sinalize fortemente quando a recomendação pessoal exigir avaliação profissional.\n\n"
            f"Histórico:\n{self._history_block()}\n\n"
            f"Pergunta:\n{state['question']}\n\n"
            f"Exames:\n{self._exam_block(state)}\n\n"
            f"Fontes:\n{sources}\n"
        )

        draft = self._llm(self.config.temperature_drafters).invoke(prompt).content.strip()
        return {"drafts": [f"[Medicamentos & Conduta]\n{draft}"]}

    def node_aggregate(self, state: AgentState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [DEBUG] Aggregator...[/dim cyan]")

        llm = self._llm(self.config.temperature_aggregator)
        sources = self._summarize_sources(state["search_results"], self.config.max_sources_in_prompt)
        drafts = "\n\n".join(state["drafts"])

        prompt = (
            "Você é o AGREGADOR final. Combine os rascunhos abaixo em UMA resposta clara em "
            "português, organizada nas seções:\n"
            "  1. Achados nos exames (se houver)\n"
            "  2. Hipóteses a considerar (com incertezas)\n"
            "  3. Red flags / quando procurar atendimento\n"
            "  4. Sobre medicamentos / conduta geral\n"
            "  5. Próximos passos sugeridos\n\n"
            "REGRAS:\n"
            " • não invente nada que não esteja nos rascunhos ou nas fontes;\n"
            " • marque incertezas com 'possivelmente', 'a confirmar', 'depende de avaliação clínica';\n"
            " • NÃO emita diagnóstico definitivo nem prescreva;\n"
            " • encerre com a frase: 'Esta análise é educacional e não substitui consulta médica.'\n\n"
            f"Histórico:\n{self._history_block()}\n\n"
            f"Pergunta:\n{state['question']}\n\n"
            f"Exames:\n{self._exam_block(state)}\n\n"
            f"Fontes:\n{sources}\n\n"
            f"Rascunhos dos especialistas:\n{drafts}\n"
        )

        final = llm.invoke(prompt).content.strip()
        return {"final_answer": final}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Graph build + run
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("plan_search", self.node_plan_search)
        g.add_node("web_search", self.node_web_search)
        g.add_node("exam_analyst", self.node_exam_analyst)
        g.add_node("clinical_reasoner", self.node_clinical_reasoner)
        g.add_node("pharma_advisor", self.node_pharma_advisor)
        g.add_node("aggregate", self.node_aggregate)

        g.set_entry_point("plan_search")
        g.add_edge("plan_search", "web_search")

        # fan-out
        g.add_edge("web_search", "exam_analyst")
        g.add_edge("web_search", "clinical_reasoner")
        g.add_edge("web_search", "pharma_advisor")

        # fan-in
        g.add_edge("exam_analyst", "aggregate")
        g.add_edge("clinical_reasoner", "aggregate")
        g.add_edge("pharma_advisor", "aggregate")

        g.add_edge("aggregate", END)
        return g.compile()

    def ask(self, question: str, use_web_search: bool = True) -> str:
        init_state: AgentState = {
            "history": self.history,
            "question": question,
            "use_web_search": use_web_search,
            "exam_text": self.loaded_exam_text,
            "image_paths": self.loaded_image_paths,
            "search_queries": [],
            "search_results": [],
            "drafts": [],
            "final_answer": "",
        }

        out = self.app.invoke(init_state)
        answer = out["final_answer"]

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Confirmação do disclaimer e CLI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def show_disclaimer_and_confirm(console: Console) -> bool:
    console.print(Panel(
        DISCLAIMER_PT,
        title="🩺 AGENTE DE SAÚDE PESSOAL — DISCLAIMER",
        border_style="red",
    ))
    try:
        ans = input('Digite "EU CONCORDO" para continuar (qualquer outra coisa encerra): ').strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.upper() == "EU CONCORDO"


def parse_load_command(line: str) -> List[str]:
    """
    Aceita:
        /load /caminho/exame.pdf
        /load "exame com espaço.pdf" outro.xlsx imagem.png
    Suporta caminhos entre aspas duplas.
    """
    body = line[len("/load"):].strip()
    if not body:
        return []
    # quoted "..."  ou tokens sem espaço
    tokens = re.findall(r'"([^"]+)"|(\S+)', body)
    return [a or b for a, b in tokens]


def main():
    config = Config()
    console = Console()

    # 1) Disclaimer obrigatório
    if not show_disclaimer_and_confirm(console):
        console.print("[yellow]Encerrado. Você não confirmou o disclaimer.[/yellow]")
        return

    # 2) Inicializa o agente (verifica modelos)
    assistant = HealthMultiAgent(config=config)

    assistant.console.print(Panel(
        'Comandos disponíveis:\n'
        '  [cyan]/load <arquivo1> <arquivo2> ...[/cyan]   anexa exames (PDF, XLSX, CSV, PNG, JPG)\n'
        '  [cyan]/show[/cyan]                              mostra arquivos anexados\n'
        '  [cyan]/clear[/cyan]                             remove todos os exames carregados\n'
        '  [cyan]/search on|off[/cyan]                     liga/desliga busca na web\n'
        '  [cyan]/disclaimer[/cyan]                        reexibe o aviso\n'
        '  [cyan]exit[/cyan] ou [cyan]quit[/cyan]                            encerra\n\n'
        '[dim]Dica: anexe seus exames com /load antes de fazer perguntas sobre eles.[/dim]',
        title="🩺 Comandos",
        border_style="white",
    ))

    use_search = True

    while True:
        assistant.console.print()
        try:
            q = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            assistant.console.print("\n[yellow]Encerrando...[/yellow]")
            break
        assistant.console.print()

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # ----- comandos -----
        if q.lower() == "/disclaimer":
            assistant.console.print(Panel(DISCLAIMER_PT, title="🩺 DISCLAIMER", border_style="red"))
            continue

        if q.lower().startswith("/search"):
            if "off" in q.lower():
                use_search = False
                assistant.console.print("🔴 [red]Busca web desativada.[/red]")
            elif "on" in q.lower():
                use_search = True
                assistant.console.print("🟢 [green]Busca web ativada.[/green]")
            continue

        if q.lower().startswith("/load"):
            paths = parse_load_command(q)
            if not paths:
                assistant.console.print("[yellow]Uso: /load arquivo1.pdf arquivo2.xlsx ...[/yellow]")
                continue
            loaded = load_exam_files(paths, max_chars=config.max_exam_text_chars)
            if loaded["text"]:
                # Acumula entre vários /load
                assistant.loaded_exam_text = (
                    assistant.loaded_exam_text + "\n\n" + loaded["text"]
                ).strip()
            assistant.loaded_image_paths.extend(loaded["images"])

            msg = []
            if loaded["text"]:
                msg.append(f"📄 Texto extraído (~{len(loaded['text'])} chars).")
            if loaded["images"]:
                msg.append(f"🖼️  {len(loaded['images'])} imagem(ns) anexada(s).")
            if loaded["missing"]:
                msg.append(f"⚠️ Não carregados: {', '.join(loaded['missing'])}")
            assistant.console.print(Panel("\n".join(msg) or "(nada carregado)",
                                          title="Exames carregados", border_style="cyan"))
            continue

        if q.lower() == "/show":
            if not assistant.loaded_exam_text and not assistant.loaded_image_paths:
                assistant.console.print("[dim](nenhum exame carregado)[/dim]")
            else:
                preview = assistant.loaded_exam_text[:600]
                if len(assistant.loaded_exam_text) > 600:
                    preview += "\n[...]"
                imgs = ", ".join(Path(p).name for p in assistant.loaded_image_paths) or "(nenhuma)"
                assistant.console.print(Panel(
                    f"[bold]Texto:[/bold]\n{preview or '(nenhum)'}\n\n[bold]Imagens:[/bold] {imgs}",
                    title="Anexos atuais", border_style="cyan",
                ))
            continue

        if q.lower() == "/clear":
            assistant.loaded_exam_text = ""
            assistant.loaded_image_paths = []
            assistant.console.print("[green]Anexos removidos.[/green]")
            continue

        # ----- pergunta normal -----
        a = assistant.ask(q, use_web_search=use_search)

        assistant.console.print()
        assistant.console.print(Panel(
            Markdown(a),
            title="🩺 Resposta",
            border_style="blue",
        ))


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    main()

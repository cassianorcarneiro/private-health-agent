# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE HEALTH AI AGENT — V2 (sequential pipeline)
# CASSIANO RIBEIRO CARNEIRO
#
# Pipeline:
#   Triage → ExamExtractor → SearchPlanner → WebSearch → ClinicalReasoner → PharmaChecker → Synthesizer
# Cada etapa produz JSON estruturado validado por Pydantic. Etapas posteriores
# consomem outputs estruturados (não texto livre) das anteriores.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import ollama

from config import Config
from schemas import (
    TriageResult, ExamExtraction, SearchPlan,
    ClinicalReasoning, PharmaResult, FinalAnswer,
)
from io_utils import (
    ExamLoader, sanitize_pii, rank_sources,
    filter_sources_by_intent, summarize_sources,
)
from llm_client import LLMClient, StructuredOutputError


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Disclaimer
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
    [bold]não sai do seu computador[/bold].
  • Quando a busca web está ligada, apenas as queries (sanitizadas) são
    enviadas ao buscador, e ainda assim o sistema tenta remover dados
    pessoais antes — mas a sanitização é de melhor esforço, não garantia.

Ao continuar, você declara que entendeu estes limites e usará a
ferramenta por sua conta e risco, apenas para fins pessoais e educacionais.
"""


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Graph state
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class PipelineState(TypedDict, total=False):
    # Input
    history: List[Dict[str, str]]
    question: str
    use_web_search: bool
    exam_text: str
    image_paths: List[str]
    images_b64: List[str]

    # Outputs estruturados de cada etapa
    triage: TriageResult
    extraction: ExamExtraction
    search_plan: SearchPlan
    search_results: List[Dict[str, Any]]
    clinical: ClinicalReasoning
    pharma: PharmaResult
    final: FinalAnswer

    # Sinalizadores de controle de fluxo
    short_circuit: bool
    pipeline_errors: List[str]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Helpers para carregamento de prompts
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def load_prompts(prompts_dir: str) -> Dict[str, str]:
    base = Path(prompts_dir)
    if not base.exists():
        # fallback: tenta o diretório ao lado do agent.py
        alt = Path(__file__).parent / prompts_dir
        if alt.exists():
            base = alt
    needed = {
        "triage": "01_triage.txt",
        "extractor": "02_extractor.txt",
        "planner": "03_planner.txt",
        "clinical": "04_clinical.txt",
        "pharma": "05_pharma.txt",
        "synthesizer": "06_synthesizer.txt",
    }
    out: Dict[str, str] = {}
    for name, fname in needed.items():
        p = base / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Prompt '{fname}' não encontrado em {base.resolve()}. "
                f"Os prompts são versionados em arquivos texto."
            )
        out[name] = p.read_text(encoding="utf-8").strip()
    return out


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Core class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@dataclass
class HealthAgent:
    config: Config

    def __post_init__(self):
        self.console = Console()
        self.history: List[Dict[str, str]] = []
        self.exam_loader = ExamLoader(max_chars=self.config.max_exam_text_chars)
        self.prompts = load_prompts(self.config.prompts_dir)

        self._check_model()

        self.llm = LLMClient(
            base_url=self.config.ollama_base_url,
            default_model=self.config.ollama_model,
            json_max_retries=self.config.json_max_retries,
        )

        self.app = self.build_graph()

    # ---- Model resolution ----------------------------------------------------------------------

    def _check_model(self):
        try:
            models_response = ollama.Client(host=self.config.ollama_base_url).list()
            model_details = []
            if hasattr(models_response, "models") and models_response.models:
                for model in models_response.models:
                    model_details.append({
                        "name": model.model,
                        "size": getattr(model, "size", 0),
                        "modified": getattr(model, "modified_at", None),
                        "parameters": getattr(model.details, "parameter_size", "N/A") if model.details else "N/A",
                    })
            if not model_details:
                self.console.print("❌ [red]Nenhum modelo encontrado no Ollama.[/red]")
                self.console.print("   Instale com: [cyan]ollama pull medgemma:4b[/cyan]")
                raise RuntimeError("No models available")

            self.config.ollama_model = self._resolve_model(self.config.ollama_model, model_details, "texto")
            self.config.ollama_vision_model = self._resolve_model(self.config.ollama_vision_model, model_details, "visão")
            if self.config.ollama_model_clinical:
                self.config.ollama_model_clinical = self._resolve_model(
                    self.config.ollama_model_clinical, model_details, "clínico"
                )
            if self.config.ollama_model_pharma:
                self.config.ollama_model_pharma = self._resolve_model(
                    self.config.ollama_model_pharma, model_details, "pharma"
                )
        except Exception as e:
            self.console.print(f"❌ Erro ao conectar ao Ollama: {e}", style="bold red")
            self.console.print("\n🔧 [yellow]Possíveis soluções:[/yellow]")
            self.console.print("1. Verifique se o Ollama está rodando: [cyan]ollama serve[/cyan]")
            self.console.print("2. Instale o MedGemma:                [cyan]ollama pull medgemma:4b[/cyan]")
            raise

    def _resolve_model(self, requested: str, model_details: list, label: str) -> str:
        """
        Match exato primeiro (mais seguro), depois substring como fallback.
        Evita o problema de 'medgemma' casar arbitrariamente com 4b ou 27b.
        """
        req_low = requested.lower()

        exact = [m for m in model_details if m["name"].lower() == req_low]
        if exact:
            chosen = exact[0]
            self._log_model(label, chosen, exact_match=True)
            return chosen["name"]

        startswith = [m for m in model_details if m["name"].lower().startswith(req_low)]
        if startswith:
            chosen = startswith[0]
            self._log_model(label, chosen, exact_match=False)
            return chosen["name"]

        contains = [m for m in model_details if req_low in m["name"].lower()]
        if contains:
            chosen = contains[0]
            self._log_model(label, chosen, exact_match=False)
            return chosen["name"]

        # Fallback final
        chosen = model_details[0]
        self.console.print(Panel(
            f"⚠️  [yellow]'{requested}' não encontrado.[/yellow]\n"
            f"Usando fallback: [bold]{chosen['name']}[/bold]\n"
            f"[dim]Para melhor desempenho:[/dim] [cyan]ollama pull medgemma:4b[/cyan]",
            title=f"🩺 Modelo ({label}) — fallback",
            border_style="yellow",
        ))
        return chosen["name"]

    def _log_model(self, label: str, chosen: dict, exact_match: bool):
        size_gb = (chosen["size"] or 0) / 1024 / 1024 / 1024
        suffix = "" if exact_match else " [dim](match não exato)[/dim]"
        self.console.print(Panel(
            f"✅ [green]Modelo de {label}:[/green] {chosen['name']}{suffix}\n"
            f"📊 [cyan]Tamanho:[/cyan] {size_gb:.1f} GB\n"
            f"⚙️  [yellow]Parâmetros:[/yellow] {chosen['parameters']}",
            title=f"🩺 Modelo ({label})",
            border_style="green",
        ))

    # ---- Utilidades ----------------------------------------------------------------------------

    def _history_block(self) -> str:
        recent = self.history[-2 * self.config.history_max_turns :]
        out = [f"{m['role'].upper()}: {m['content']}" for m in recent]
        return "\n".join(out) if out else "(sem contexto prévio)"

    def _exam_text_for_prompt(self) -> str:
        return self.exam_loader.text or "(nenhum exame anexado)"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # NODES
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    # ---- 1. Triage -----------------------------------------------------------------------------

    def node_triage(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [1/6] Triage...[/dim cyan]")

        prompt = (
            self.prompts["triage"]
            + f"\n\nHistórico recente:\n{self._history_block()}"
            + f"\n\nResumo dos exames anexados (preview):\n"
            + (state.get("exam_text", "")[:1500] or "(nenhum)")
            + f"\n\nPergunta do usuário:\n{state['question']}"
        )
        try:
            triage = self.llm.chat_structured(
                prompt=prompt,
                schema=TriageResult,
                temperature=self.config.temperature_triage,
            )
        except StructuredOutputError as e:
            self.console.print(f"[yellow]Triage falhou validação: {e}. Assumindo categoria genérica.[/yellow]")
            triage = TriageResult(category="general_health")

        # Short-circuit em emergência: pula direto para o Synthesizer com estado mínimo.
        short = bool(triage.is_emergency or triage.red_flags)

        if short:
            self.console.print("[bold red]   ⚠ Red flags detectados — pipeline curto-circuitado.[/bold red]")

        return {
            "triage": triage,
            "short_circuit": short,
            "pipeline_errors": [],
        }

    # ---- 2. Exam Extractor ---------------------------------------------------------------------

    def node_extract(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [2/6] Extração de exames...[/dim cyan]")

        if not state.get("exam_text") and not state.get("images_b64"):
            return {"extraction": ExamExtraction()}

        prompt = (
            self.prompts["extractor"]
            + f"\n\n[TEXTO EXTRAÍDO DOS EXAMES]\n{state.get('exam_text','(nenhum)')}"
            + f"\n\n[PERGUNTA DO USUÁRIO PARA CONTEXTO]\n{state['question']}"
        )
        try:
            extraction = self.llm.chat_structured(
                prompt=prompt,
                schema=ExamExtraction,
                temperature=self.config.temperature_extractor,
                model=self.config.ollama_vision_model,
                images_b64=state.get("images_b64") or None,
            )
        except StructuredOutputError as e:
            self.console.print(f"[yellow]Extração falhou validação: {e}. Continuando sem achados estruturados.[/yellow]")
            extraction = ExamExtraction(
                extraction_quality="low",
                extraction_notes=f"Falha na extração estruturada: {e}",
            )

        n_lab = len(extraction.lab_findings)
        n_img = len(extraction.image_findings)
        if n_lab or n_img:
            self.console.print(f"   📋 {n_lab} achado(s) laboratorial(is), {n_img} achado(s) de imagem.")

        return {"extraction": extraction}

    # ---- 3. Search Planner ---------------------------------------------------------------------

    def node_plan_search(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [3/6] Planejamento de busca...[/dim cyan]")

        if not state.get("use_web_search", True):
            return {"search_plan": SearchPlan(queries=[]), "search_results": []}

        triage = state["triage"]
        extraction = state.get("extraction") or ExamExtraction()

        # Sumariza achados ANORMAIS de forma compacta para o planner.
        abnormal = extraction.abnormal_findings()
        abnormal_summary = (
            "\n".join(
                f"- {f.parameter}: {f.value} {f.unit or ''} (status: {f.status})"
                for f in abnormal[:10]
            ) or "(nenhum achado anormal)"
        )

        # Sanitização de PII no input que vai informar as queries
        question_for_search = (
            sanitize_pii(state["question"])
            if self.config.sanitize_pii_in_search else state["question"]
        )

        prompt = (
            self.prompts["planner"]
            + f"\n\n[CATEGORIA TRIADA]: {triage.category}"
            + f"\n[MEDICAMENTOS MENCIONADOS]: {', '.join(triage.mentioned_drugs) or '(nenhum)'}"
            + f"\n[TÓPICOS-CHAVE]: {', '.join(triage.key_topics) or '(nenhum)'}"
            + f"\n\n[ACHADOS ANORMAIS DOS EXAMES]:\n{abnormal_summary}"
            + f"\n\n[PERGUNTA DO USUÁRIO (sanitizada)]:\n{question_for_search}"
        )
        try:
            plan = self.llm.chat_structured(
                prompt=prompt,
                schema=SearchPlan,
                temperature=self.config.temperature_planner,
            )
        except StructuredOutputError:
            # Fallback: uma única query a partir da pergunta sanitizada
            plan = SearchPlan(queries=[
                {"query": question_for_search[:120], "intent": "clinical"}
            ])

        # Cap pelo config
        plan.queries = plan.queries[: self.config.max_queries]
        return {"search_plan": plan}

    # ---- 4. Web Search -------------------------------------------------------------------------

    def node_web_search(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [4/6] Busca web...[/dim cyan]")

        plan = state.get("search_plan") or SearchPlan()
        if not state.get("use_web_search", True) or not plan.queries:
            return {"search_results": []}

        out: List[Dict[str, Any]] = []
        for q_item in plan.queries:
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(q_item.query, max_results=self.config.ddgs_max_results_per_query)
                    for r in results:
                        if isinstance(r, dict):
                            out.append({"query": q_item.query, "intent": q_item.intent, **r})
            except Exception as e:
                out.append({"query": q_item.query, "intent": q_item.intent, "error": str(e)})

        # Re-ranking por domínio confiável.
        ranked = rank_sources(
            out,
            preferred_domains=self.config.preferred_medical_sources,
            max_items=self.config.max_sources_in_prompt * 2,  # margem para filtragem por intent
        )
        return {"search_results": ranked}

    # ---- 5. Clinical Reasoner ------------------------------------------------------------------

    def node_clinical(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [5/6] Raciocínio clínico...[/dim cyan]")

        triage = state["triage"]
        extraction = state.get("extraction") or ExamExtraction()

        # Filtra fontes por intent — clinical reasoner NÃO precisa ler bula.
        clinical_sources = filter_sources_by_intent(
            state.get("search_results", []),
            intents_allowed=["clinical", "reference"],
        )
        sources_text = summarize_sources(clinical_sources, self.config.max_sources_in_prompt)

        # Achados estruturados em JSON (não texto solto) — é a chave do design.
        findings_json = extraction.model_dump_json(indent=2)

        prompt = (
            self.prompts["clinical"]
            + f"\n\n[CATEGORIA TRIADA]: {triage.category}"
            + f"\n[MEDICAMENTOS MENCIONADOS PELO USUÁRIO]: {', '.join(triage.mentioned_drugs) or '(nenhum)'}"
            + f"\n\n[ACHADOS ESTRUTURADOS DOS EXAMES (JSON)]:\n{findings_json}"
            + f"\n\n[HISTÓRICO RECENTE]:\n{self._history_block()}"
            + f"\n\n[PERGUNTA DO USUÁRIO]:\n{state['question']}"
            + f"\n\n[FONTES CLÍNICAS RELEVANTES]:\n{sources_text}"
        )

        model = self.config.ollama_model_clinical or self.config.ollama_model
        try:
            clinical = self.llm.chat_structured(
                prompt=prompt,
                schema=ClinicalReasoning,
                temperature=self.config.temperature_clinical,
                model=model,
            )
        except StructuredOutputError as e:
            self.console.print(f"[yellow]Raciocínio clínico falhou: {e}.[/yellow]")
            clinical = ClinicalReasoning(
                summary="(falha na geração estruturada do raciocínio clínico)",
                data_limitations=[f"Erro: {e}"],
            )
        return {"clinical": clinical}

    # ---- 6. Pharma Checker ---------------------------------------------------------------------

    def node_pharma(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [6/6] Análise farmacológica...[/dim cyan]")

        triage = state["triage"]
        clinical = state.get("clinical") or ClinicalReasoning()

        # Pharma só vê fontes de drug/interaction.
        pharma_sources = filter_sources_by_intent(
            state.get("search_results", []),
            intents_allowed=["drug", "interaction"],
        )
        sources_text = summarize_sources(pharma_sources, self.config.max_sources_in_prompt)

        # Diferenciais resumidos (só nome + probabilidade)
        diffs_summary = (
            "\n".join(f"- {d.name} ({d.probability})" for d in clinical.differentials[:6])
            or "(nenhum diferencial relevante)"
        )

        prompt = (
            self.prompts["pharma"]
            + f"\n\n[MEDICAMENTOS MENCIONADOS]: {', '.join(triage.mentioned_drugs) or '(nenhum)'}"
            + f"\n[CATEGORIA TRIADA]: {triage.category}"
            + f"\n\n[DIFERENCIAIS DO RACIOCÍNIO CLÍNICO]:\n{diffs_summary}"
            + f"\n\n[FONTES FARMACOLÓGICAS]:\n{sources_text}"
            + f"\n\n[PERGUNTA DO USUÁRIO]:\n{state['question']}"
        )
        model = self.config.ollama_model_pharma or self.config.ollama_model
        try:
            pharma = self.llm.chat_structured(
                prompt=prompt,
                schema=PharmaResult,
                temperature=self.config.temperature_pharma,
                model=model,
            )
        except StructuredOutputError as e:
            self.console.print(f"[yellow]Análise farmacológica falhou: {e}.[/yellow]")
            pharma = PharmaResult(
                general_advice="(falha na geração estruturada — peça orientação ao seu médico/farmacêutico).",
            )
        return {"pharma": pharma}

    # ---- 7. Synthesizer ------------------------------------------------------------------------

    def node_synthesize(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> Sintetizando resposta final...[/dim cyan]")

        triage = state["triage"]
        extraction = state.get("extraction") or ExamExtraction()
        clinical = state.get("clinical") or ClinicalReasoning()
        pharma = state.get("pharma") or PharmaResult()

        prompt = (
            self.prompts["synthesizer"]
            + "\n\n[ENTRADAS ESTRUTURADAS]"
            + f"\n\n[1] Triage:\n{triage.model_dump_json(indent=2)}"
            + f"\n\n[2] Extração de exames:\n{extraction.model_dump_json(indent=2)}"
            + f"\n\n[3] Raciocínio clínico:\n{clinical.model_dump_json(indent=2)}"
            + f"\n\n[4] Análise farmacológica:\n{pharma.model_dump_json(indent=2)}"
            + f"\n\n[HISTÓRICO RECENTE]:\n{self._history_block()}"
            + f"\n\n[PERGUNTA DO USUÁRIO]:\n{state['question']}"
        )
        try:
            final = self.llm.chat_structured(
                prompt=prompt,
                schema=FinalAnswer,
                temperature=self.config.temperature_synthesizer,
            )
        except StructuredOutputError as e:
            # Fallback robusto: monta uma resposta degradada mas coerente
            fallback_md = self._fallback_synthesis(triage, extraction, clinical, pharma, error=str(e))
            final = FinalAnswer(
                answer_markdown=fallback_md,
                confidence_level="low",
                must_seek_care=triage.is_emergency or bool(triage.red_flags),
            )

        # Garantia da frase educacional final (defesa em profundidade)
        required_tail = "Esta análise é educacional e não substitui consulta médica."
        if required_tail not in final.answer_markdown:
            final.answer_markdown = final.answer_markdown.rstrip() + f"\n\n*{required_tail}*"

        return {"final": final}

    def _fallback_synthesis(
        self,
        triage: TriageResult, extraction: ExamExtraction,
        clinical: ClinicalReasoning, pharma: PharmaResult, error: str,
    ) -> str:
        """Resposta degradada quando o synthesizer falha. Usa só os dados estruturados."""
        parts: List[str] = []
        if triage.red_flags:
            parts.append("## 🚨 Atenção\n" + "\n".join(f"- {r}" for r in triage.red_flags))
            parts.append("Procure avaliação médica presencial.")
        if extraction.lab_findings:
            parts.append("## 📋 Achados nos exames")
            for f in extraction.lab_findings:
                ref = f" (ref: {f.reference_range})" if f.reference_range else ""
                parts.append(f"- **{f.parameter}**: {f.value} {f.unit or ''}{ref} — *{f.status}*")
        if clinical.differentials:
            parts.append("## 🤔 Hipóteses a considerar")
            for d in clinical.differentials:
                parts.append(f"- **{d.name}** ({d.probability}) — {d.rationale}")
        if pharma.drug_info:
            parts.append("## 💊 Sobre os medicamentos")
            for d in pharma.drug_info:
                parts.append(f"- **{d.name}** ({d.drug_class}) — {d.mechanism_short}")
        parts.append(f"\n*Nota: o agregador automático falhou ({error[:100]}); "
                     f"esta resposta foi montada diretamente dos dados estruturados.*")
        parts.append("\n*Esta análise é educacional e não substitui consulta médica.*")
        return "\n\n".join(parts)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Roteamento condicional do grafo
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    @staticmethod
    def _after_triage(state: PipelineState) -> str:
        if state.get("short_circuit"):
            return "synthesize"  # pula direto para o final em emergências
        return "extract"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Build & run
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def build_graph(self):
        g = StateGraph(PipelineState)

        g.add_node("triage", self.node_triage)
        g.add_node("extract", self.node_extract)
        g.add_node("plan_search", self.node_plan_search)
        g.add_node("web_search", self.node_web_search)
        g.add_node("clinical", self.node_clinical)
        g.add_node("pharma", self.node_pharma)
        g.add_node("synthesize", self.node_synthesize)

        g.set_entry_point("triage")
        # Roteamento condicional pós-triagem (curto-circuito em emergência)
        g.add_conditional_edges("triage", self._after_triage, {
            "extract": "extract",
            "synthesize": "synthesize",
        })
        g.add_edge("extract", "plan_search")
        g.add_edge("plan_search", "web_search")
        g.add_edge("web_search", "clinical")
        g.add_edge("clinical", "pharma")
        g.add_edge("pharma", "synthesize")
        g.add_edge("synthesize", END)

        return g.compile()

    def ask(self, question: str, use_web_search: bool = True) -> FinalAnswer:
        init_state: PipelineState = {
            "history": self.history,
            "question": question,
            "use_web_search": use_web_search,
            "exam_text": self.exam_loader.text,
            "image_paths": self.exam_loader.image_paths,
            "images_b64": self.exam_loader.images_base64(),
        }

        try:
            out = self.app.invoke(init_state)
        except Exception as e:
            self.console.print(f"[bold red]Erro inesperado no pipeline: {e}[/bold red]")
            return FinalAnswer(
                answer_markdown=(
                    "## ⚠️ Erro técnico\n\n"
                    f"Ocorreu um erro processando sua pergunta: `{e}`\n\n"
                    "O histórico da conversa foi preservado. Tente reformular ou "
                    "verifique se o Ollama continua rodando.\n\n"
                    "*Esta análise é educacional e não substitui consulta médica.*"
                ),
                confidence_level="low",
                must_seek_care=False,
            )

        final: FinalAnswer = out["final"]

        # Histórico só guarda o markdown final, e cai pelo último N*2 limit no _history_block
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": final.answer_markdown})
        # Trim duro pra não acumular RAM em sessões longas
        max_items = self.config.history_max_turns * 4
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]

        return final


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# REPL
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def show_disclaimer_and_confirm(console: Console) -> bool:
    console.print(Panel(DISCLAIMER_PT, title="🩺 AGENTE DE SAÚDE PESSOAL — DISCLAIMER", border_style="red"))
    try:
        ans = input('Digite "EU CONCORDO" para continuar (qualquer outra coisa encerra): ').strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.upper() == "EU CONCORDO"


def parse_load_command(line: str) -> List[str]:
    body = line[len("/load"):].strip()
    if not body:
        return []
    tokens = re.findall(r'"([^"]+)"|(\S+)', body)
    return [a or b for a, b in tokens]


def main():
    config = Config()
    console = Console()

    if not show_disclaimer_and_confirm(console):
        console.print("[yellow]Encerrado. Você não confirmou o disclaimer.[/yellow]")
        return

    try:
        agent = HealthAgent(config=config)
    except Exception:
        return

    agent.console.print(Panel(
        'Comandos disponíveis:\n'
        '  [cyan]/load <arq1> <arq2> ...[/cyan]   anexa exames (PDF, XLSX, CSV, PNG, JPG)\n'
        '  [cyan]/show[/cyan]                      mostra arquivos anexados\n'
        '  [cyan]/clear[/cyan]                     remove todos os exames carregados\n'
        '  [cyan]/search on|off[/cyan]             liga/desliga busca na web\n'
        '  [cyan]/disclaimer[/cyan]                reexibe o aviso\n'
        '  [cyan]/debug[/cyan]                     mostra outputs estruturados da última pergunta\n'
        '  [cyan]exit[/cyan] ou [cyan]quit[/cyan]                       encerra\n\n'
        '[dim]Anexe seus exames com /load antes de fazer perguntas sobre eles.[/dim]',
        title="🩺 Comandos", border_style="white",
    ))

    use_search = True
    last_state: Optional[Dict[str, Any]] = None  # para /debug

    while True:
        agent.console.print()
        try:
            q = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            agent.console.print("\n[yellow]Encerrando...[/yellow]")
            break
        agent.console.print()

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # ----- comandos -----
        if q.lower() == "/disclaimer":
            agent.console.print(Panel(DISCLAIMER_PT, title="🩺 DISCLAIMER", border_style="red"))
            continue

        if q.lower().startswith("/search"):
            if "off" in q.lower():
                use_search = False
                agent.console.print("🔴 [red]Busca web desativada.[/red]")
            elif "on" in q.lower():
                use_search = True
                agent.console.print("🟢 [green]Busca web ativada.[/green]")
            continue

        if q.lower().startswith("/load"):
            paths = parse_load_command(q)
            if not paths:
                agent.console.print("[yellow]Uso: /load arquivo1.pdf arquivo2.xlsx ...[/yellow]")
                continue
            res = agent.exam_loader.add(paths)
            msg = []
            if res["added_text_chars"]:
                msg.append(f"📄 +{res['added_text_chars']} chars de texto.")
            if res["added_images"]:
                msg.append(f"🖼️  +{res['added_images']} imagem(ns) (base64 cacheado).")
            if res["missing"]:
                msg.append(f"⚠️ Não carregados: {', '.join(res['missing'])}")
            agent.console.print(Panel("\n".join(msg) or "(nada carregado)",
                                      title="Exames carregados", border_style="cyan"))
            continue

        if q.lower() == "/show":
            if not agent.exam_loader.has_any():
                agent.console.print("[dim](nenhum exame carregado)[/dim]")
            else:
                preview = agent.exam_loader.text[:600]
                if len(agent.exam_loader.text) > 600:
                    preview += "\n[...]"
                imgs = ", ".join(Path(p).name for p in agent.exam_loader.image_paths) or "(nenhuma)"
                agent.console.print(Panel(
                    f"[bold]Texto:[/bold]\n{preview or '(nenhum)'}\n\n[bold]Imagens:[/bold] {imgs}",
                    title="Anexos atuais", border_style="cyan",
                ))
            continue

        if q.lower() == "/clear":
            agent.exam_loader.clear()
            agent.console.print("[green]Anexos removidos.[/green]")
            continue

        if q.lower() == "/debug":
            if not last_state:
                agent.console.print("[dim]Nenhuma pergunta processada ainda.[/dim]")
            else:
                for k, v in last_state.items():
                    if hasattr(v, "model_dump_json"):
                        agent.console.print(Panel(
                            v.model_dump_json(indent=2),
                            title=f"[debug] {k}", border_style="magenta",
                        ))
            continue

        # ----- pergunta normal -----
        final = agent.ask(q, use_web_search=use_search)
        last_state = {"final": final}

        agent.console.print()
        border = "red" if final.must_seek_care else "blue"
        agent.console.print(Panel(
            Markdown(final.answer_markdown),
            title=f"🩺 Resposta (confiança: {final.confidence_level})",
            border_style=border,
        ))


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    main()

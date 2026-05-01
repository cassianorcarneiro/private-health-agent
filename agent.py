# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE HEALTH AI AGENT — V2 (sequential pipeline)
# REPOSITORY: https://github.com/cassianorcarneiro/private-health-agent
# CASSIANO RIBEIRO CARNEIRO
#
# Pipeline:
#   Triage → ExamExtractor → SearchPlanner → WebSearch → ClinicalReasoner → PharmaChecker → Synthesizer
# Each step produces Pydantic-validated structured JSON. Subsequent steps
# consume structured outputs (not free text) from the previous ones.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

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

DISCLAIMER_EN = """\
[bold red]⚠️  IMPORTANT NOTICE — READ BEFORE USE ⚠️[/bold red]

This agent is an [bold]experimental[/bold] educational support tool,
built on large language models (LLMs) — including Google's MedGemma,
which is a [bold]research model[/bold], NOT a medical device.

[bold yellow]THIS AGENT DOES NOT:[/bold yellow]
  • replace a doctor, pharmacist, or any healthcare professional;
  • issue valid clinical diagnoses;
  • prescribe medications, dosages, or therapeutic procedures;
  • undergo evaluation by any regulatory body (FDA, EMA, ANVISA, etc.);
  • guarantee the accuracy, completeness, or timeliness of information.

[bold yellow]KNOWN RISKS:[/bold yellow]
  • LLMs may [bold]hallucinate[/bold] (fabricating facts, reference values,
    drug interactions, and plausible-looking diagnoses);
  • lab result interpretation depends on clinical context the model
    lacks (history, physical exam, comorbidities, current medications);
  • reference values vary by laboratory, age, sex, and testing method;
  • web searches may return outdated or low-quality sources.

[bold yellow]HOW TO USE SAFELY:[/bold yellow]
  • Treat responses as [bold]hypotheses to discuss with your doctor[/bold].
  • For warning symptoms (chest pain, shortness of breath, neurological deficit,
    bleeding, persistent high fever, etc.), seek medical care [bold]immediately[/bold].
  • Do not stop or start medications based on this tool.
  • Do not share third-party patient data without consent.

[bold yellow]PRIVACY:[/bold yellow]
  • Models run locally via Ollama; your exam content [bold]does not leave your computer[/bold].
  • When web search is enabled, only (sanitized) queries are sent to the
    search engine. The system attempts to remove personal data beforehand,
    but sanitization is a best-effort attempt, not a guarantee.

By continuing, you declare that you understand these limits and will use the
tool at your own risk, for personal and educational purposes only.
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

    # Structured outputs from each step
    triage: TriageResult
    extraction: ExamExtraction
    search_plan: SearchPlan
    search_results: List[Dict[str, Any]]
    clinical: ClinicalReasoning
    pharma: PharmaResult
    final: FinalAnswer

    # Flow control flags
    short_circuit: bool
    pipeline_errors: List[str]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Prompt Loading Helpers
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def load_prompts(prompts_dir: str) -> Dict[str, str]:
    base = Path(prompts_dir)
    if not base.exists():
        # fallback: try the directory next to agent.py
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
                f"Prompt '{fname}' not found in {base.resolve()}. "
                f"Prompts are versioned in text files."
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

        self.console.print("[dim]→ Loading prompts...[/dim]")
        self.prompts = load_prompts(self.config.prompts_dir)

        self.console.print("[dim]→ Checking models in Ollama...[/dim]")
        self._check_model()

        self.console.print("[dim]→ Initializing LLM client...[/dim]")
        self.llm = LLMClient(
            base_url=self.config.ollama_base_url,
            default_model=self.config.ollama_model,
            json_max_retries=self.config.json_max_retries,
        )

        self.console.print("[dim]→ Building agent graph...[/dim]")
        self.app = self.build_graph()
        self.console.print("[green]✓ Ready.[/green]\n")

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
                self.console.print("❌ [red]No models found in Ollama.[/red]")
                self.console.print("   Install with: [cyan]ollama pull medgemma:4b[/cyan]")
                raise RuntimeError("No models available")

            self.config.ollama_model = self._resolve_model(self.config.ollama_model, model_details, "text")
            self.config.ollama_vision_model = self._resolve_model(self.config.ollama_vision_model, model_details, "vision")
            if self.config.ollama_model_clinical:
                self.config.ollama_model_clinical = self._resolve_model(
                    self.config.ollama_model_clinical, model_details, "clinical"
                )
            if self.config.ollama_model_pharma:
                self.config.ollama_model_pharma = self._resolve_model(
                    self.config.ollama_model_pharma, model_details, "pharma"
                )
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.config.ollama_base_url}.\n"
                f"   Original error: {e}\n\n"
                "🔧 Possible solutions:\n"
                "   1. Verify if Ollama is running:  ollama serve\n"
                "   2. Install MedGemma:             ollama pull medgemma:4b\n"
                "   3. Confirm the URL in config.ollama_base_url"
            ) from e

    def _resolve_model(self, requested: str, model_details: list, label: str) -> str:
        """
        Exact match first (safest), then substring as fallback.
        Prevents 'medgemma' from matching arbitrarily with 4b or 27b.
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

        # Final Fallback
        chosen = model_details[0]
        self.console.print(Panel(
            f"⚠️  [yellow]'{requested}' not found.[/yellow]\n"
            f"Using fallback: [bold]{chosen['name']}[/bold]\n"
            f"[dim]For better performance:[/dim] [cyan]ollama pull medgemma:4b[/cyan]",
            title=f"🩺 Model ({label}) — fallback",
            border_style="yellow",
        ))
        return chosen["name"]

    def _log_model(self, label: str, chosen: dict, exact_match: bool):
        size_gb = (chosen["size"] or 0) / 1024 / 1024 / 1024
        suffix = "" if exact_match else " [dim](inexact match)[/dim]"
        self.console.print(Panel(
            f"✅ [green]{label.capitalize()} Model:[/green] {chosen['name']}{suffix}\n"
            f"📊 [cyan]Size:[/cyan] {size_gb:.1f} GB\n"
            f"⚙️  [yellow]Parameters:[/yellow] {chosen['parameters']}",
            title=f"🩺 Model ({label})",
            border_style="green",
        ))

    # ---- Utilities ----------------------------------------------------------------------------

    def _history_block(self) -> str:
        recent = self.history[-2 * self.config.history_max_turns :]
        out = [f"{m['role'].upper()}: {m['content']}" for m in recent]
        return "\n".join(out) if out else "(no previous context)"

    def _exam_text_for_prompt(self) -> str:
        return self.exam_loader.text or "(no exams attached)"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # NODES
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    # ---- 1. Triage -----------------------------------------------------------------------------

    def node_triage(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [1/6] Triage...[/dim cyan]")

        prompt = (
            self.prompts["triage"]
            + f"\n\nRecent history:\n{self._history_block()}"
            + f"\n\nAttached exam summary (preview):\n"
            + (state.get("exam_text", "")[:1500] or "(none)")
            + f"\n\nUser question:\n{state['question']}"
        )
        try:
            triage = self.llm.chat_structured(
                prompt=prompt,
                schema=TriageResult,
                temperature=self.config.temperature_triage,
            )
        except StructuredOutputError as e:
            self.console.print(f"[yellow]Triage validation failed: {e}. Assuming generic health category.[/yellow]")
            triage = TriageResult(category="general_health")

        # Short-circuit in case of emergency: skip straight to Synthesizer with minimal state.
        short = bool(triage.is_emergency or triage.red_flags)

        if short:
            self.console.print("[bold red]   ⚠ Red flags detected — pipeline short-circuited.[/bold red]")

        return {
            "triage": triage,
            "short_circuit": short,
            "pipeline_errors": [],
        }

    # ---- 2. Exam Extractor ---------------------------------------------------------------------

    def node_extract(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [2/6] Exam extraction...[/dim cyan]")

        if not state.get("exam_text") and not state.get("images_b64"):
            return {"extraction": ExamExtraction()}

        prompt = (
            self.prompts["extractor"]
            + f"\n\n[EXTRACTED EXAM TEXT]\n{state.get('exam_text','(none)')}"
            + f"\n\n[USER QUESTION FOR CONTEXT]\n{state['question']}"
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
            self.console.print(f"[yellow]Extraction validation failed: {e}. Continuing without structured findings.[/yellow]")
            extraction = ExamExtraction(
                extraction_quality="low",
                extraction_notes=f"Failed structured extraction: {e}",
            )

        n_lab = len(extraction.lab_findings)
        n_img = len(extraction.image_findings)
        if n_lab or n_img:
            self.console.print(f"   📋 {n_lab} lab finding(s), {n_img} image finding(s).")

        return {"extraction": extraction}

    # ---- 3. Search Planner ---------------------------------------------------------------------

    def node_plan_search(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [3/6] Search planning...[/dim cyan]")

        if not state.get("use_web_search", True):
            return {"search_plan": SearchPlan(queries=[]), "search_results": []}

        triage = state["triage"]
        extraction = state.get("extraction") or ExamExtraction()

        # Summarize ABNORMAL findings concisely for the planner.
        abnormal = extraction.abnormal_findings()
        abnormal_summary = (
            "\n".join(
                f"- {f.parameter}: {f.value} {f.unit or ''} (status: {f.status})"
                for f in abnormal[:10]
            ) or "(no abnormal findings)"
        )

        # Sanitize PII in the input that informs the queries
        question_for_search = (
            sanitize_pii(state["question"])
            if self.config.sanitize_pii_in_search else state["question"]
        )

        prompt = (
            self.prompts["planner"]
            + f"\n\n[TRIAGED CATEGORY]: {triage.category}"
            + f"\n[MENTIONED DRUGS]: {', '.join(triage.mentioned_drugs) or '(none)'}"
            + f"\n[KEY TOPICS]: {', '.join(triage.key_topics) or '(none)'}"
            + f"\n\n[ABNORMAL EXAM FINDINGS]:\n{abnormal_summary}"
            + f"\n\n[USER QUESTION (sanitized)]:\n{question_for_search}"
        )
        try:
            plan = self.llm.chat_structured(
                prompt=prompt,
                schema=SearchPlan,
                temperature=self.config.temperature_planner,
            )
        except StructuredOutputError:
            # Fallback: a single query from the sanitized question
            plan = SearchPlan(queries=[
                {"query": question_for_search[:120], "intent": "clinical"}
            ])

        # Cap based on config
        plan.queries = plan.queries[: self.config.max_queries]
        return {"search_plan": plan}

    # ---- 4. Web Search -------------------------------------------------------------------------

    def node_web_search(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [4/6] Web search...[/dim cyan]")

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

        # Re-ranking by trusted domain.
        ranked = rank_sources(
            out,
            preferred_domains=self.config.preferred_medical_sources,
            max_items=self.config.max_sources_in_prompt * 2,  # margin for intent filtering
        )
        return {"search_results": ranked}

    # ---- 5. Clinical Reasoner ------------------------------------------------------------------

    def node_clinical(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [5/6] Clinical reasoning...[/dim cyan]")

        triage = state["triage"]
        extraction = state.get("extraction") or ExamExtraction()

        # Filter sources by intent — clinical reasoner does NOT need drug leaflets.
        clinical_sources = filter_sources_by_intent(
            state.get("search_results", []),
            intents_allowed=["clinical", "reference"],
        )
        sources_text = summarize_sources(clinical_sources, self.config.max_sources_in_prompt)

        # Structured JSON findings (not loose text) — the core design key.
        findings_json = extraction.model_dump_json(indent=2)

        prompt = (
            self.prompts["clinical"]
            + f"\n\n[TRIAGED CATEGORY]: {triage.category}"
            + f"\n[MEDICATIONS MENTIONED BY USER]: {', '.join(triage.mentioned_drugs) or '(none)'}"
            + f"\n\n[STRUCTURED EXAM FINDINGS (JSON)]:\n{findings_json}"
            + f"\n\n[RECENT HISTORY]:\n{self._history_block()}"
            + f"\n\n[USER QUESTION]:\n{state['question']}"
            + f"\n\n[RELEVANT CLINICAL SOURCES]:\n{sources_text}"
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
            self.console.print(f"[yellow]Clinical reasoning failed: {e}.[/yellow]")
            clinical = ClinicalReasoning(
                summary="(failed structured generation of clinical reasoning)",
                data_limitations=[f"Error: {e}"],
            )
        return {"clinical": clinical}

    # ---- 6. Pharma Checker ---------------------------------------------------------------------

    def node_pharma(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [6/6] Pharmacological analysis...[/dim cyan]")

        triage = state["triage"]
        clinical = state.get("clinical") or ClinicalReasoning()

        # Pharma only looks at drug/interaction sources.
        pharma_sources = filter_sources_by_intent(
            state.get("search_results", []),
            intents_allowed=["drug", "interaction"],
        )
        sources_text = summarize_sources(pharma_sources, self.config.max_sources_in_prompt)

        # Summarized differentials (name + probability only)
        diffs_summary = (
            "\n".join(f"- {d.name} ({d.probability})" for d in clinical.differentials[:6])
            or "(no relevant differentials)"
        )

        prompt = (
            self.prompts["pharma"]
            + f"\n\n[MENTIONED DRUGS]: {', '.join(triage.mentioned_drugs) or '(none)'}"
            + f"\n[TRIAGED CATEGORY]: {triage.category}"
            + f"\n\n[CLINICAL REASONING DIFFERENTIALS]:\n{diffs_summary}"
            + f"\n\n[PHARMACOLOGICAL SOURCES]:\n{sources_text}"
            + f"\n\n[USER QUESTION]:\n{state['question']}"
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
            self.console.print(f"[yellow]Pharmacological analysis failed: {e}.[/yellow]")
            pharma = PharmaResult(
                general_advice="(structured generation failed — please seek guidance from your doctor/pharmacist).",
            )
        return {"pharma": pharma}

    # ---- 7. Synthesizer ------------------------------------------------------------------------

    def node_synthesize(self, state: PipelineState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> Synthesizing final response...[/dim cyan]")

        triage = state["triage"]
        extraction = state.get("extraction") or ExamExtraction()
        clinical = state.get("clinical") or ClinicalReasoning()
        pharma = state.get("pharma") or PharmaResult()

        prompt = (
            self.prompts["synthesizer"]
            + "\n\n[STRUCTURED INPUTS]"
            + f"\n\n[1] Triage:\n{triage.model_dump_json(indent=2)}"
            + f"\n\n[2] Exam extraction:\n{extraction.model_dump_json(indent=2)}"
            + f"\n\n[3] Clinical reasoning:\n{clinical.model_dump_json(indent=2)}"
            + f"\n\n[4] Pharmacological analysis:\n{pharma.model_dump_json(indent=2)}"
            + f"\n\n[RECENT HISTORY]:\n{self._history_block()}"
            + f"\n\n[USER QUESTION]:\n{state['question']}"
        )
        try:
            final = self.llm.chat_structured(
                prompt=prompt,
                schema=FinalAnswer,
                temperature=self.config.temperature_synthesizer,
            )
        except StructuredOutputError as e:
            # Robust fallback: assemble a degraded but coherent response
            fallback_md = self._fallback_synthesis(triage, extraction, clinical, pharma, error=str(e))
            final = FinalAnswer(
                answer_markdown=fallback_md,
                confidence_level="low",
                must_seek_care=triage.is_emergency or bool(triage.red_flags),
            )

        # Educational disclaimer final guarantee (defense in depth)
        required_tail = "This analysis is educational and does not replace medical consultation."
        if required_tail not in final.answer_markdown:
            final.answer_markdown = final.answer_markdown.rstrip() + f"\n\n*{required_tail}*"

        return {"final": final}

    def _fallback_synthesis(
        self,
        triage: TriageResult, extraction: ExamExtraction,
        clinical: ClinicalReasoning, pharma: PharmaResult, error: str,
    ) -> str:
        """Degraded response for when the synthesizer fails. Uses structured data only."""
        parts: List[str] = []
        if triage.red_flags:
            parts.append("## 🚨 Warning\n" + "\n".join(f"- {r}" for r in triage.red_flags))
            parts.append("Seek in-person medical evaluation.")
        if extraction.lab_findings:
            parts.append("## 📋 Exam Findings")
            for f in extraction.lab_findings:
                ref = f" (ref: {f.reference_range})" if f.reference_range else ""
                parts.append(f"- **{f.parameter}**: {f.value} {f.unit or ''}{ref} — *{f.status}*")
        if clinical.differentials:
            parts.append("## 🤔 Hypotheses to Consider")
            for d in clinical.differentials:
                parts.append(f"- **{d.name}** ({d.probability}) — {d.rationale}")
        if pharma.drug_info:
            parts.append("## 💊 Medication Information")
            for d in pharma.drug_info:
                parts.append(f"- **{d.name}** ({d.drug_class}) — {d.mechanism_short}")
        parts.append(f"\n*Note: the automatic aggregator failed ({error[:100]}); "
                     f"this response was assembled directly from structured data.*")
        parts.append("\n*This analysis is educational and does not replace medical consultation.*")
        return "\n\n".join(parts)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Conditional Graph Routing
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    @staticmethod
    def _after_triage(state: PipelineState) -> str:
        if state.get("short_circuit"):
            return "synthesize"  # skip straight to final in emergencies
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
        # Conditional routing post-triage (short-circuit in emergencies)
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
            self.console.print(f"[bold red]Unexpected error in pipeline: {e}[/bold red]")
            return FinalAnswer(
                answer_markdown=(
                    "## ⚠️ Technical Error\n\n"
                    f"An error occurred while processing your question: `{e}`\n\n"
                    "Conversation history was preserved. Try rephrasing or "
                    "verify if Ollama is still running.\n\n"
                    "*This analysis is educational and does not replace medical consultation.*"
                ),
                confidence_level="low",
                must_seek_care=False,
            )

        final: FinalAnswer = out["final"]

        # History only stores final markdown
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": final.answer_markdown})
        # Hard trim to avoid RAM accumulation in long sessions
        max_items = self.config.history_max_turns * 4
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]

        return final


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# REPL (Read-Eval-Print Loop)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def show_disclaimer_and_confirm(console: Console) -> bool:
    console.print(Panel(DISCLAIMER_EN, title="🩺 PERSONAL HEALTH AGENT — DISCLAIMER", border_style="red"))
    try:
        ans = input('Type "I AGREE" to continue (anything else will quit): ').strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.upper() == "I AGREE"


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
        console.print("[yellow]Closed. You did not confirm the disclaimer.[/yellow]")
        return

    try:
        agent = HealthAgent(config=config)
    except FileNotFoundError as e:
        console.print(f"\n[bold red]❌ Prompt file missing:[/bold red]\n   {e}")
        console.print(
            "\n[yellow]Prompts are located in ./prompts/ next to agent.py.[/yellow]\n"
            "Verify if the folder was copied together with the .py files."
        )
        return
    except RuntimeError as e:
        console.print(f"\n[bold red]❌ {e}[/bold red]")
        return
    except Exception:
        console.print("\n[bold red]❌ Unexpected initialization failure:[/bold red]")
        console.print_exception()
        return

    agent.console.print(Panel(
        'Available Commands:\n'
        '  [cyan]/load <file1> <file2> ...[/cyan]   attach exams (PDF, XLSX, CSV, PNG, JPG)\n'
        '  [cyan]/show[/cyan]                      show attached files\n'
        '  [cyan]/clear[/cyan]                     remove all loaded exams\n'
        '  [cyan]/search on|off[/cyan]             enable/disable web search\n'
        '  [cyan]/disclaimer[/cyan]                re-display the notice\n'
        '  [cyan]/debug[/cyan]                     show structured outputs from the last question\n'
        '  [cyan]exit[/cyan] or [cyan]quit[/cyan]                       shutdown\n\n'
        '[dim]Attach your exams with /load before asking questions about them.[/dim]',
        title="🩺 Commands", border_style="white",
    ))

    use_search = True
    last_state: Optional[Dict[str, Any]] = None  # for /debug

    while True:
        agent.console.print()
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            agent.console.print("\n[yellow]Shutting down...[/yellow]")
            break
        agent.console.print()

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # ----- commands -----
        if q.lower() == "/disclaimer":
            agent.console.print(Panel(DISCLAIMER_EN, title="🩺 DISCLAIMER", border_style="red"))
            continue

        if q.lower().startswith("/search"):
            if "off" in q.lower():
                use_search = False
                agent.console.print("🔴 [red]Web search disabled.[/red]")
            elif "on" in q.lower():
                use_search = True
                agent.console.print("🟢 [green]Web search enabled.[/green]")
            continue

        if q.lower().startswith("/load"):
            paths = parse_load_command(q)
            if not paths:
                agent.console.print("[yellow]Usage: /load file1.pdf file2.xlsx ...[/yellow]")
                continue
            res = agent.exam_loader.add(paths)
            msg = []
            if res["added_text_chars"]:
                msg.append(f"📄 +{res['added_text_chars']} text chars.")
            if res["added_images"]:
                msg.append(f"🖼️  +{res['added_images']} image(s) (base64 cached).")
            if res["missing"]:
                msg.append(f"⚠️ Not loaded: {', '.join(res['missing'])}")
            agent.console.print(Panel("\n".join(msg) or "(nothing loaded)",
                                      title="Loaded Exams", border_style="cyan"))
            continue

        if q.lower() == "/show":
            if not agent.exam_loader.has_any():
                agent.console.print("[dim](no exams loaded)[/dim]")
            else:
                preview = agent.exam_loader.text[:600]
                if len(agent.exam_loader.text) > 600:
                    preview += "\n[...]"
                imgs = ", ".join(Path(p).name for p in agent.exam_loader.image_paths) or "(none)"
                agent.console.print(Panel(
                    f"[bold]Text:[/bold]\n{preview or '(none)'}\n\n[bold]Images:[/bold] {imgs}",
                    title="Current Attachments", border_style="cyan",
                ))
            continue

        if q.lower() == "/clear":
            agent.exam_loader.clear()
            agent.console.print("[green]Attachments removed.[/green]")
            continue

        if q.lower() == "/debug":
            if not last_state:
                agent.console.print("[dim]No questions processed yet.[/dim]")
            else:
                for k, v in last_state.items():
                    if hasattr(v, "model_dump_json"):
                        agent.console.print(Panel(
                            v.model_dump_json(indent=2),
                            title=f"[debug] {k}", border_style="magenta",
                        ))
            continue

        # ----- normal question -----
        final = agent.ask(q, use_web_search=use_search)
        last_state = {"final": final}

        agent.console.print()
        border = "red" if final.must_seek_care else "blue"
        agent.console.print(Panel(
            Markdown(final.answer_markdown),
            title=f"🩺 Answer (confidence: {final.confidence_level})",
            border_style=border,
        ))


if __name__ == "__main__":
    main()
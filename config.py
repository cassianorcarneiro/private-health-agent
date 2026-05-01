# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE HEALTH AI AGENT — V2 (sequential pipeline)
# CASSIANO RIBEIRO CARNEIRO
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:

    # ----- Ollama settings -----
    # MedGemma is multimodal and trained specifically for the medical domain.
    # On hardware with more memory, consider using medgemma:27b (text-only) for
    # clinical reasoning and keep medgemma:4b solely for the vision stage.
    ollama_model: str = "medgemma:4b"
    ollama_vision_model: str = "medgemma:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"

    # Allows using different models per agent (advanced).
    # If empty, defaults to ollama_model.
    ollama_model_clinical: str = ""   # e.g.: "medgemma:27b"
    ollama_model_pharma: str = ""

    # ----- Temperatures -----
    # Health requires a conservative approach; we keep it low at all stages.
    temperature_triage: float = 0.0
    temperature_extractor: float = 0.0
    temperature_planner: float = 0.0
    temperature_clinical: float = 0.2
    temperature_pharma: float = 0.1
    temperature_synthesizer: float = 0.1

    # ----- Robustness -----
    json_max_retries: int = 2
    invoke_timeout_seconds: int = 180

    # ----- Web Search -----
    ddgs_max_results_per_query: int = 5
    max_queries: int = 6
    max_sources_in_prompt: int = 12

    # Trusted medical sources. Re-ranking gives a boost to these domains.
    preferred_medical_sources: List[str] = field(default_factory=lambda: [
        "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "who.int",
        "cdc.gov", "nih.gov", "mayoclinic.org", "medlineplus.gov",
        "uptodate.com", "bvsalud.org", "scielo.br", "cochranelibrary.com",
        "anvisa.gov.br", "gov.br/saude", "sbcardio.org.br", "diabetes.org.br",
        "drugs.com", "rxlist.com", "fda.gov", "ema.europa.eu",
        "merckmanuals.com", "msdmanuals.com",
    ])

    # ----- Health-specific -----
    max_exam_text_chars: int = 20000
    exams_dir: str = "./exams"
    history_max_turns: int = 6

    # PII Sanitization before the planner (prevents leaking patient names in web queries)
    sanitize_pii_in_search: bool = True

    # Directory for versioned prompts
    prompts_dir: str = "./prompts"
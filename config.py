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
    # MedGemma é multimodal e treinado para domínio médico.
    # Em hardware com mais memória, considere medgemma:27b (texto-only) para o
    # raciocínio clínico e mantenha medgemma:4b apenas para a etapa de visão.
    ollama_model: str = "medgemma:4b"
    ollama_vision_model: str = "medgemma:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"

    # Permite usar modelos diferentes por agente (avançado).
    # Se vazio, usa ollama_model.
    ollama_model_clinical: str = ""   # ex.: "medgemma:27b"
    ollama_model_pharma: str = ""

    # ----- Temperaturas -----
    # Saúde exige conservadorismo; mantemos baixo em todas as etapas.
    temperature_triage: float = 0.0
    temperature_extractor: float = 0.0
    temperature_planner: float = 0.0
    temperature_clinical: float = 0.2
    temperature_pharma: float = 0.1
    temperature_synthesizer: float = 0.1

    # ----- Robustez -----
    json_max_retries: int = 2
    invoke_timeout_seconds: int = 180

    # ----- Web Search -----
    ddgs_max_results_per_query: int = 5
    max_queries: int = 6
    max_sources_in_prompt: int = 12

    # Fontes médicas confiáveis. Re-ranking dá boost a essas.
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
    exams_dir: str = "./exames"
    history_max_turns: int = 6

    # Sanitização de PII antes do planner (evita vazar nome do paciente em queries)
    sanitize_pii_in_search: bool = True

    # Diretório de prompts versionados
    prompts_dir: str = "./prompts"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE HEALTH AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:

    # ----- Ollama settings -----
    # MedGemma é multimodal (texto + imagem) e treinado para domínio médico.
    # Opções: medgemma:4b (3.3GB, multimodal), medgemma:27b (17GB, texto),
    # medgemma1.5:4b (versão mais nova, 3D imaging, EHR, lab reports).
    # Modelo TEXTUAL principal (planejamento, raciocínio, agregação):
    ollama_model = "medgemma:4b"
    # Modelo VISÃO (interpretação de imagens médicas e PDFs escaneados).
    # Pode ser o mesmo do ollama_model se ele for multimodal.
    ollama_vision_model = "medgemma:4b"
    ollama_base_url = "http://127.0.0.1:11434"

    # ----- LLM temperatures -----
    # Saúde exige conservadorismo; mantemos temperaturas baixas para reduzir alucinação.
    temperature_planner = 0.0
    temperature_drafters = 0.2
    temperature_aggregator = 0.1

    # ----- Web Search settings -----
    # Para consultas sobre sintomas/medicamentos, priorizamos fontes confiáveis.
    ddgs_max_results_per_query = 5
    max_queries = 6
    max_sources_in_prompt = 12

    # Fontes médicas preferidas (usadas como dica para o planner).
    preferred_medical_sources = [
        "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "who.int",
        "cdc.gov", "nih.gov", "mayoclinic.org", "medlineplus.gov",
        "uptodate.com", "bvsalud.org", "scielo.br",
        "anvisa.gov.br", "gov.br/saude", "bulario.com",
        "drugs.com", "rxlist.com", "fda.gov",
    ]

    # ----- Health-specific settings -----
    # Tamanho máximo (em caracteres) de texto extraído de PDF/planilha por exame.
    # Evita estourar contexto com exames muito longos.
    max_exam_text_chars = 20000

    # Diretório padrão onde os exames podem ser colocados (opcional).
    exams_dir = "./exames"

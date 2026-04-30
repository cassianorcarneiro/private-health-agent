# 🩺 Private Health Agent

> Privacy-first, offline-capable health assistant powered by local medical language models.

A **sequential multi-agent** AI assistant that orchestrates specialized health agents and optional medical web retrieval to analyze lab results, reason about differential diagnoses, and provide medication information — all running locally on your machine via [Google Health's MedGemma](https://deepmind.google/models/gemma/medgemma/) models.

<p align="center">
  <img alt="Stack" src="https://img.shields.io/badge/Stack-LangGraph%20%2B%20Ollama-blue?style=for-the-badge">
  <img alt="Model" src="https://img.shields.io/badge/Model-MedGemma-4285F4?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  <img alt="Status" src="https://img.shields.io/badge/Privacy-Local%20First-success?style=for-the-badge">
  <img alt="Disclaimer" src="https://img.shields.io/badge/⚠️-NOT%20A%20MEDICAL%20DEVICE-critical?style=for-the-badge">
</p>

---

## ⚠️⚠️⚠️ READ THIS FIRST — IMPORTANT DISCLAIMER ⚠️⚠️⚠️

> **THIS PROJECT IS NOT A MEDICAL DEVICE.**
> It is an **experimental, educational** tool built by an individual, with **no clinical validation** and **no regulatory approval** (ANVISA, FDA, EMA, CE, etc.).

### This agent does NOT:

- 🚫 replace a physician, pharmacist, or any licensed healthcare professional;
- 🚫 issue clinical diagnoses;
- 🚫 prescribe medications, doses, or treatments;
- 🚫 carry any regulatory clearance;
- 🚫 guarantee accuracy, completeness, or timeliness of its outputs.

### Known risks

1. **LLM hallucination** — models can fabricate reference ranges, drug interactions, guideline citations, and diagnoses that *sound* correct but are wrong. Google itself describes MedGemma as a **research model**, not for direct clinical use.
2. **Missing clinical context** — proper interpretation depends on history, physical exam, comorbidities, and current medications, which the model has only partially or not at all.
3. **Reference range variability** — normal ranges vary by lab, age, sex, and method; the model may use generic values that don't match yours.
4. **Medical imaging** — multimodal models can miss small lesions and confidently describe regions that don't exist.
5. **Web search quality** — when enabled, results may include outdated, commercial, or non-peer-reviewed content.

### Red flags — seek immediate care, do **not** consult a chatbot

Non-exhaustive examples: chest pain, sudden shortness of breath, syncope, acute neurological deficit (speech, strength, vision), heavy or persistent bleeding, high fever with confusion or stiff neck, severe progressive abdominal pain, symptoms in pregnancy, newborns, or immunocompromised patients.

### Use it responsibly

- Treat answers as **hypotheses to discuss with your doctor**, never as conclusions.
- **Do not start, adjust, or stop medications** based on this agent.
- Do not use third-party patient data without informed consent.
- When web search is on, always verify the cited sources.

> **Use is at your own risk. The author accepts no liability for decisions made based on the agent's output. By running the program you must explicitly type `EU CONCORDO` to confirm you've read this disclaimer.**

---

## 📦 Why this project

Most health-oriented AI tools either ship your medical data to third-party APIs or are full-blown clinical platforms with steep onboarding. This project takes a different stance:

- 🏠 **Local-first** — exam content and images stay on your machine; reasoning runs via [Ollama](https://ollama.com)
- 🩻 **Multimodal** — reads lab PDFs, spreadsheets (`.xlsx` / `.csv` / `.tsv`), and medical images (X-rays, dermatoscopy, fundus photos, etc.) using [MedGemma](https://deepmind.google/models/gemma/medgemma/), Google's open medical model
- 🔗 **Sequential specialist pipeline** — six agents in series, each consuming the *structured output* of the previous one (Pydantic-validated JSON), so each step adds genuinely new information instead of restating prior drafts
- 🚦 **Triage-first short-circuit** — every question is triaged for emergencies before any other processing; red-flag cases skip straight to the response with a clear "seek care" message
- 🎯 **Intent-aware retrieval** — search queries carry intent labels (`clinical` / `drug` / `interaction` / `reference`) so the clinical reasoner doesn't see drug-bula sources and the pharmacology agent doesn't see guideline summaries
- 🛡️ **PII-aware** — Brazilian-format CPF/RG/phone/email/dates are scrubbed from inputs before they reach the search planner
- 📊 **Source re-ranking** — results from trusted medical domains (PubMed, NIH, WHO, Mayo, ANVISA, drugs.com, etc.) are bumped to the top before being summarized into prompts
- 🧩 **Versioned prompts** — every agent's instructions live in `./prompts/*.txt`, separate from code, so changes are diff-able
- 🤖 **Validated outputs end-to-end** — every LLM call that should return structure is constrained with `format=json` and validated by Pydantic, with automatic re-prompting on failure

---

## 🏗️ How it works

```
                                 ┌── short-circuit on red flag ──┐
                                 ▼                                │
    Triage ──> Extractor ──> Planner ──> Web Search ──> Clinical Reasoner ──> Pharma Checker ──> Synthesizer ──> END
     (JSON)     (JSON)        (JSON)       (ranked)         (JSON)              (JSON)             (JSON)
```

| Stage | Input | Output |
|-------|-------|--------|
| **Triage** | Question + exam preview | Category, red flags, mentioned drugs, key topics |
| **Extractor** | Exam text + images | Structured `LabFinding[]` and `ImageFinding[]` |
| **Planner** | Triage + abnormal findings | Search queries tagged with intent |
| **Web Search** | Queries | DDGS results, re-ranked by trusted-domain bonus |
| **Clinical Reasoner** | Structured findings + clinical-intent sources | `Differential[]` with probability and supporting evidence |
| **Pharma Checker** | Mentioned drugs + differentials + drug-intent sources | `DrugInfo[]`, interactions, contraindications |
| **Synthesizer** | All structured outputs | Final markdown answer with sections + confidence level |

The pipeline is implemented as a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with conditional edges (the post-triage router decides between the full pipeline and the short-circuit). State is a `TypedDict` with one slot per agent's structured output, so any intermediate result is inspectable via the `/debug` command.

### Why sequential, not parallel?

Earlier prototypes used a fan-out/fan-in pattern with three "specialist" agents seeing the same context in parallel. With small medical models (4B parameters), this produced near-identical drafts that the aggregator was forced to merge — a lot of compute for a marginal gain. The current pipeline:

- avoids redundancy: each agent works on a strictly different piece of the problem;
- gives the clinical reasoner JSON-typed findings instead of free-text summaries;
- lets the pharma checker react to the *actual* differentials, not to a parallel guess;
- short-circuits emergencies in seconds (no extraction, no search, no drug lookup);
- makes every intermediate state visible and re-runnable.

---

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** running locally — get it at [ollama.com/download](https://ollama.com/download)
- **~3.5 GB free disk** for `medgemma:4b` (multimodal); **~17 GB** for `medgemma:27b` (text-only)
- **8 GB+ RAM** recommended for the 4B model; **24 GB+** for the 27B
- **Internet** for the first run (pulling the model and, optionally, web search)

> **MedGemma terms of use** — MedGemma is distributed under the [Health AI Developer Foundations (HAI-DEF) terms](https://developers.google.com/health-ai-developer-foundations/terms). Please read them before using.

---

## 🚀 Quick start

### 1. Install Ollama and pull a MedGemma model

```bash
# Multimodal, ~3.3 GB — recommended for personal use
ollama pull medgemma:4b

# Or the newer 1.5 release (3D imaging, EHR understanding, lab report parsing)
ollama pull medgemma1.5:4b

# Larger text-only variant, ~17 GB — best for clinical reasoning if you have the RAM
ollama pull medgemma:27b

ollama serve
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the agent

```bash
python agent.py
```

You'll see the disclaimer; type `EU CONCORDO` to proceed. Then interact with the assistant:

```
Você: /load ./exames/hemograma.pdf ./exames/colesterol.xlsx
Você: meu LDL aumentou em relação ao último exame de 6 meses atrás.
      Estou em uso de sinvastatina 20 mg. O que pode estar acontecendo?
```

### Commands

| Command | Effect |
|---------|--------|
| `/load <file1> <file2> ...` | Attach exams (PDF, XLSX, CSV, TSV, PNG, JPG, WEBP, …) — image base64 is cached |
| `/show` | Display currently attached files |
| `/clear` | Remove all attachments |
| `/search on` | Enable DuckDuckGo retrieval over medical sources (default) |
| `/search off` | Disable retrieval — model uses only internal knowledge |
| `/disclaimer` | Reprint the disclaimer |
| `/debug` | Print structured outputs of the last question (Pydantic JSON) |
| `exit` or `quit` | Leave the assistant |

### Session memory

The assistant maintains a rolling conversation history (last 6 turns by default), trimmed to a hard cap to prevent unbounded RAM growth. History and attached exams live **in RAM only** and are discarded on exit — nothing is written to disk.

---

## ⚙️ Configuration

Edit `config.py` to tune behavior:

| Field | Purpose | Default |
|-------|---------|---------|
| `ollama_model` | Default text/reasoning model (exact match preferred, then prefix, then substring) | `medgemma:4b` |
| `ollama_vision_model` | Multimodal model used for medical images | `medgemma:4b` |
| `ollama_model_clinical` | Override for the Clinical Reasoner (e.g. `medgemma:27b`) | `""` (uses default) |
| `ollama_model_pharma` | Override for the Pharma Checker | `""` (uses default) |
| `ollama_base_url` | Ollama server URL | `http://127.0.0.1:11434` |
| `temperature_*` | Per-agent temperature; all conservative (0.0 – 0.2) | see file |
| `json_max_retries` | Retries when an agent's JSON fails Pydantic validation | `2` |
| `max_queries` | Cap on web search queries per question | `6` |
| `ddgs_max_results_per_query` | Results fetched per query | `5` |
| `max_sources_in_prompt` | Sources passed to each agent, post-filtering | `12` |
| `max_exam_text_chars` | Truncation cap on extracted exam text | `20000` |
| `preferred_medical_sources` | Domains the re-ranker boosts | PubMed, NIH, WHO, CDC, Mayo, ANVISA, drugs.com, … |
| `sanitize_pii_in_search` | Mask CPF/RG/phone/email/dates before search | `True` |
| `prompts_dir` | Where versioned prompt files live | `./prompts` |

### Recommended models

| Model | Size | Modality | Best for |
|-------|------|----------|----------|
| `medgemma:4b` | 3.3 GB | Text + Image | General use; reads images of exams |
| `medgemma1.5:4b` | 3.3 GB | Text + Image | Same as above + better lab report parsing and longitudinal CXR comparison |
| `medgemma:27b` | 17 GB | Text only | Best textual reasoning if you don't need vision |

**Hybrid setup:** keep `ollama_vision_model = "medgemma:4b"` for image extraction, set `ollama_model_clinical = "medgemma:27b"` for clinical reasoning. The Pharma Checker can stay on the default — its task is simpler.

### Editing prompts

Prompts live as plain text files in `./prompts/` (`01_triage.txt`, `02_extractor.txt`, etc.). Edit them directly; they're versioned by your VCS like any other source. The agent loads them at startup.

---

## 🔐 Privacy model

- ✅ Exam PDFs, spreadsheets, and medical images are processed **entirely locally** by Ollama
- ✅ When web search is **off**, no data leaves your machine
- ✅ When web search is **on**, only sanitized search **queries** generated by the planner are sent to DuckDuckGo. Before reaching the planner, the question is scrubbed for Brazilian PII patterns (CPF, RG, phone, email, dates, record numbers). This is a **best-effort** mitigation, not a compliance guarantee.
- ✅ No telemetry, no analytics, no API keys required
- ✅ Conversation and attachments are kept in RAM only and wiped on exit
- ✅ Image files are read once and cached as base64 — they aren't re-read from disk on every turn

---

## 📁 Project structure

```
private-health-agent/
├── agent.py            # HealthAgent class, graph nodes, REPL loop, disclaimer
├── config.py           # Config dataclass with models and behavior settings
├── schemas.py          # Pydantic schemas (TriageResult, ExamExtraction, ...)
├── io_utils.py         # Exam loading, PII sanitization, source ranking
├── llm_client.py       # Unified Ollama client with structured-JSON retry
├── prompts/
│   ├── 01_triage.txt
│   ├── 02_extractor.txt
│   ├── 03_planner.txt
│   ├── 04_clinical.txt
│   ├── 05_pharma.txt
│   └── 06_synthesizer.txt
├── requirements.txt
└── README.md
```

---

## 🛣️ Roadmap

- [ ] OCR for scanned PDFs (Tesseract / EasyOCR)
- [ ] DICOM support
- [ ] Longitudinal comparison mode (same exam at different dates)
- [ ] Streaming the synthesizer's output to the terminal
- [ ] Export of the final answer as PDF
- [ ] PT-BR-focused source bundles (national bularies, gov.br/saúde)
- [ ] Per-node checkpoints so `/debug` can show every intermediate Pydantic object
- [ ] Lightweight evaluation harness with sample exams (no real PHI)

---

## 📜 License

MIT — see `LICENSE` file.

> **Note:** the MIT license applies to the source code in this repository. The MedGemma weights are governed by the [HAI-DEF terms](https://developers.google.com/health-ai-developer-foundations/terms) and are not redistributed here.

## 👤 Author

**Cassiano Ribeiro Carneiro** — [@cassianorcarneiro](https://github.com/cassianorcarneiro)

The author is **not a healthcare professional**. This project is provided **"AS IS"**, without warranties of any kind.

---

### 🤖 AI Assistance Disclosure

The codebase architecture, organizational structure, and stylistic formatting of this repository were refactored and optimized leveraging [Claude](https://www.anthropic.com/claude) by Anthropic. All core business logic and intellectual property remain the work of the repository authors and are governed by the project's license.

---

> 💬 *If you're considering using this agent to make a real decision about your health — stop and book an appointment. This tool exists to help you ask better questions, not to replace your doctor.*

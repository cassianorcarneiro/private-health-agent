# 🩺 Private Health Agent

> Privacy-first, offline-capable health assistant powered by local medical language models.

A multi-agent AI assistant that orchestrates specialized health agents and optional medical web retrieval to analyze lab results, reason about differential diagnoses, and provide medication information — all running locally on your machine via [Google Health's MedGemma](https://deepmind.google/models/gemma/medgemma/) models.

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
- 🤖 **Specialist multi-agent design** — three medical agents draft answers in parallel, and a fourth aggregates them, producing more balanced and explicitly-uncertain responses than a single pass
- 🔍 **Optional medical web retrieval** — DuckDuckGo searches biased toward trusted sources (PubMed, WHO, CDC, NIH, Mayo Clinic, ANVISA, etc.); can be toggled per question
- 🧩 **Transparent state** — built on [LangGraph](https://github.com/langchain-ai/langgraph), every step (planning, searching, exam analysis, clinical reasoning, drug info, aggregation) is an explicit node you can inspect or modify

---

## 🏗️ How it works

```
                ┌──> Exam Analyst       (PDFs, spreadsheets, images)  ──┐
plan_search ──> web_search ──> Clinical Reasoner   (differentials, red flags) ──┼──> aggregate ──> END
                └──> Pharma & Conduct   (drugs, interactions, advice)  ──┘
```

| Stage | Role |
|-------|------|
| **Planner** | Generates 3–6 targeted medical queries, biased toward trusted sources |
| **Web Search** | Fetches results via DuckDuckGo (skipped when `/search off`) |
| **Exam Analyst** | Parses attached lab PDFs / spreadsheets and uses MedGemma's vision capability for medical images. Extracts structured findings (parameter, value, reference, status) without inventing reference ranges |
| **Clinical Reasoner** | Lists 3–6 differential diagnoses ordered by probability, with supporting/contradicting findings, confirmatory tests, and red flags. Never says "you have X" — uses "compatible with", "suggestive of" |
| **Pharma & Conduct** | Mechanism, indications, common vs serious adverse effects, interactions, contraindications. Never prescribes specific doses |
| **Aggregator** | Merges drafts into a single structured answer (Findings → Differentials → Red Flags → Drugs → Next Steps), marking uncertainty, ending with a fixed educational disclaimer |

The fan-out / fan-in pattern is implemented with LangGraph's `Annotated[List[str], add]` reducer, so the three specialists run independently and their outputs are concatenated automatically.

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

# Larger text-only variant, ~17 GB
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
| `/load <file1> <file2> ...` | Attach exams (PDF, XLSX, CSV, TSV, PNG, JPG, WEBP, …) |
| `/show` | Display currently attached files |
| `/clear` | Remove all attachments |
| `/search on` | Enable DuckDuckGo retrieval over medical sources (default) |
| `/search off` | Disable retrieval — model uses only internal knowledge |
| `/disclaimer` | Reprint the disclaimer |
| `exit` or `quit` | Leave the assistant |

### Session memory

The assistant maintains a rolling conversation history (last 6 turns by default), so follow-ups stay in context. History and attached exams are held **in memory only** and discarded when the program exits — nothing is written to disk.

---

## ⚙️ Configuration

Edit `config.py` to tune behavior:

| Field | Purpose | Default |
|-------|---------|---------|
| `ollama_model` | Local text/reasoning model (substring match) | `medgemma:4b` |
| `ollama_vision_model` | Multimodal model used for medical images | `medgemma:4b` |
| `ollama_base_url` | Ollama server URL | `http://127.0.0.1:11434` |
| `temperature_planner` | Determinism of the search planner | `0.0` |
| `temperature_drafters` | Creativity of the three specialists | `0.2` |
| `temperature_aggregator` | Determinism of the final merge | `0.1` |
| `max_queries` | Cap on web search queries per question | `6` |
| `ddgs_max_results_per_query` | Results fetched per query | `5` |
| `max_sources_in_prompt` | Sources included in drafter prompts | `12` |
| `max_exam_text_chars` | Truncation cap on extracted exam text | `20000` |
| `preferred_medical_sources` | Domains the planner is biased toward | PubMed, NIH, WHO, CDC, Mayo, ANVISA, … |

### Recommended models

| Model | Size | Modality | Best for |
|-------|------|----------|----------|
| `medgemma:4b` | 3.3 GB | Text + Image | General use; reads images of exams |
| `medgemma1.5:4b` | 3.3 GB | Text + Image | Same as above + better lab report parsing and longitudinal CXR comparison |
| `medgemma:27b` | 17 GB | Text only | Best textual reasoning if you don't need vision |

---

## 🔐 Privacy model

- ✅ Exam PDFs, spreadsheets, and medical images are processed **entirely locally** by Ollama
- ✅ When web search is **off**, no data leaves your machine
- ✅ When web search is **on**, only the planner's generated **queries** are sent to DuckDuckGo — never the exam content or full conversation
- ✅ No telemetry, no analytics, no API keys required
- ✅ Conversation and attachments are kept in RAM only and wiped on exit

---

## 📁 Project structure

```
private-health-agent/
├── agent.py            # HealthMultiAgent class, graph nodes, REPL loop, disclaimer
├── config.py           # Config dataclass with models and behavior settings
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🛣️ Roadmap

- [ ] OCR for scanned PDFs (Tesseract / EasyOCR)
- [ ] DICOM support
- [ ] Longitudinal comparison mode (same exam at different dates)
- [ ] Export of the final answer as PDF
- [ ] PT-BR-focused source bundles (national bularies, gov.br/saúde)
- [ ] Streaming responses

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

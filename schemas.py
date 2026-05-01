# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Pydantic schemas for the structured outputs of the agents.
# Each agent in the pipeline produces a JSON validated by one of these schemas.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------- 1. Triage ----------------------------------------------------------------------------

class TriageResult(BaseModel):
    category: Literal[
        "exam_review", "symptom_inquiry", "drug_question",
        "general_health", "mixed", "out_of_scope"
    ]
    red_flags: List[str] = Field(default_factory=list)
    is_emergency: bool = False
    mentioned_drugs: List[str] = Field(default_factory=list)
    key_topics: List[str] = Field(default_factory=list)


# ---------- 2. Exam Extractor --------------------------------------------------------------------

class LabFinding(BaseModel):
    parameter: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: Literal["high", "low", "normal", "indeterminate"] = "indeterminate"
    clinical_direction: Literal["favorable", "unfavorable", "neutral", "unknown"] = "unknown"
    note: Optional[str] = None


class ImageFinding(BaseModel):
    modality: str
    description: str
    concerning: bool = False


class ExamMetadata(BaseModel):
    exam_dates_present: List[str] = Field(default_factory=list)
    patient_info_present: bool = False
    ocr_quality_concern: bool = False


class ExamExtraction(BaseModel):
    lab_findings: List[LabFinding] = Field(default_factory=list)
    image_findings: List[ImageFinding] = Field(default_factory=list)
    exam_metadata: ExamMetadata = Field(default_factory=ExamMetadata)
    extraction_quality: Literal["high", "medium", "low"] = "medium"
    extraction_notes: Optional[str] = None

    def has_any(self) -> bool:
        return bool(self.lab_findings or self.image_findings)

    def abnormal_findings(self) -> List[LabFinding]:
        return [
            f for f in self.lab_findings
            if f.status in ("high", "low", "indeterminate")
            and f.clinical_direction != "favorable"
        ]


# ---------- 3. Search Plan -----------------------------------------------------------------------

class SearchQueryItem(BaseModel):
    query: str
    intent: Literal["clinical", "drug", "interaction", "reference"]


class SearchPlan(BaseModel):
    queries: List[SearchQueryItem] = Field(default_factory=list)


# ---------- 4. Clinical Reasoner -----------------------------------------------------------------

class Differential(BaseModel):
    name: str
    probability: Literal["high", "moderate", "low", "unclear"]
    supporting_findings: List[str] = Field(default_factory=list)
    contradicting_findings: List[str] = Field(default_factory=list)
    confirmatory_tests: List[str] = Field(default_factory=list)
    rationale: str = ""


class ClinicalReasoning(BaseModel):
    summary: str = ""
    differentials: List[Differential] = Field(default_factory=list)
    data_limitations: List[str] = Field(default_factory=list)
    additional_red_flags: List[str] = Field(default_factory=list)


# ---------- 5. Pharma ----------------------------------------------------------------------------

class DrugInfo(BaseModel):
    name: str
    drug_class: str = ""
    mechanism_short: str = ""
    common_indications: List[str] = Field(default_factory=list)
    common_adverse_effects: List[str] = Field(default_factory=list)
    serious_adverse_effects: List[str] = Field(default_factory=list)
    key_interactions: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    high_risk_populations: List[str] = Field(default_factory=list)
    dose_disclaimer: str = (
        "Individual doses must be defined by a physician — "
        "information provided here is generic package insert info."
    )


class PharmaResult(BaseModel):
    drug_info: List[DrugInfo] = Field(default_factory=list)
    interaction_warnings: List[str] = Field(default_factory=list)
    general_advice: str = ""
    requires_professional_evaluation: bool = True


# ---------- 6. Final synthesis -------------------------------------------------------------------

class FinalAnswer(BaseModel):
    answer_markdown: str
    confidence_level: Literal["high", "medium", "low"] = "low"
    must_seek_care: bool = False
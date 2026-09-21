from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.workflow import analyze_claim, build_workflow


DecisionStatus = Literal[
    "ADMISSIBLE",
    "ADMISSIBLE_WITH_LIMITS",
    "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE",
    "NEEDS_REVIEW",
]

ValidationStatus = Literal[
    "PENDING",
    "PASS",
    "FAIL",
]


class ClaimRequest(BaseModel):
    """
    Accept the synthetic claim as a JSON object while preserving
    additional claim-specific fields used by the workflow.
    """

    model_config = ConfigDict(extra="allow")

    case_id: str = Field(..., min_length=1)


class Citation(BaseModel):
    source: str | None = None
    page: int | str | None = None
    section: str | None = None
    chunk_id: str | None = None


class AnalyzeResponse(BaseModel):
    case_id: str
    decision: DecisionStatus
    confidence: float

    key_findings: list[str] = Field(default_factory=list)

    coverage_findings: list[dict[str, Any]] = Field(default_factory=list)
    exclusion_findings: list[dict[str, Any]] = Field(default_factory=list)
    limit_findings: list[dict[str, Any]] = Field(default_factory=list)

    missing_evidence: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    citations: list[Citation] = Field(default_factory=list)

    validation_status: ValidationStatus
    trace: list[dict[str, Any]] = Field(default_factory=list)

    decision_dimensions: list[str] = Field(default_factory=list)
    investigation_plan: list[str] = Field(default_factory=list)

    expense_calculation: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Policy-Aware Multi-Agent RAG Claim Decision Engine",
    description=(
        "Policy-grounded synthetic health-insurance claim analysis "
        "using a multi-agent LangGraph workflow."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_workflow():
    """
    Build the LangGraph workflow once and reuse it across requests.

    The workflow itself loads the retrieval models lazily when the
    evidence agent is first executed.
    """
    return build_workflow()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "policy-aware-multi-agent-rag",
    }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
def analyze(request: ClaimRequest) -> AnalyzeResponse:
    """
    Analyze one synthetic claim through the complete multi-agent
    workflow.

    The supplied claim is passed unchanged to the workflow. Policy
    conclusions are produced only by the policy-aware agents.
    """

    claim = request.model_dump()

    try:
        result = analyze_claim(
            claim,
            workflow=get_workflow(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Claim analysis failed: {exc}",
        ) from exc

    return AnalyzeResponse(
        case_id=result.get("case_id", request.case_id),
        decision=result.get("decision", "NEEDS_REVIEW"),
        confidence=float(result.get("confidence", 0.0)),
        key_findings=result.get("key_findings", []),
        coverage_findings=result.get("coverage_findings", []),
        exclusion_findings=result.get("exclusion_findings", []),
        limit_findings=result.get("limit_findings", []),
        missing_evidence=result.get("missing_evidence", []),
        unsupported_claims=result.get("unsupported_claims", []),
        citations=result.get("citations", []),
        validation_status=result.get(
            "validation_status",
            "PENDING",
        ),
        trace=result.get("trace", []),
        decision_dimensions=result.get(
            "decision_dimensions",
            [],
        ),
        investigation_plan=result.get(
            "investigation_plan",
            [],
        ),
        expense_calculation=result.get(
            "expense_calculation",
            {},
        ),
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Policy-Aware Multi-Agent RAG Claim Decision Engine",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze",
    }

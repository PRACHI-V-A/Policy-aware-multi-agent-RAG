from typing import TypedDict, Any


class ClaimState(TypedDict, total=False):
    # Input
    case_id: str
    claim_facts: dict[str, Any]

    # Case investigation
    decision_dimensions: list[str]
    investigation_plan: list[str]
    missing_evidence: list[str]

    # RAG
    retrieved_evidence: list[dict[str, Any]]

    # Analysis
    coverage_findings: list[dict[str, Any]]
    exclusion_findings: list[dict[str, Any]]
    limit_findings: list[dict[str, Any]]

    # Decision
    decision: str
    confidence: float
    key_findings: list[str]
    expense_calculation: dict[str, Any]

    # Validation
    citations: list[dict[str, Any]]
    validation_status: str
    unsupported_claims: list[str]

    # Audit trace
    trace: list[dict[str, Any]]
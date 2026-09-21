import pytest

from src.agents.case_agent import _build_decision_dimensions
from src.agents.validation_agent import (
    _build_evidence_locations,
    _validate_decision_safety,
    _validate_finding,
    validation_agent,
)


# ================================================================
# CASE ANALYSIS AGENT TESTS
# ================================================================


def test_case_agent_detects_expense_dimensions():
    claim = {
        "case_id": "TEST-001",
        "hospitalization_type": "inpatient",
        "hospitalization_hours": 72,
        "condition": "acute bacterial infection",
        "procedure": "medical treatment",
        "expenses": {
            "room": 12000,
            "doctor_fees": 25000,
            "medicines_diagnostics": 50000,
            "ambulance": 500,
        },
        "evidence_context": {
            "hospital_registered": True,
            "medical_necessity_confirmed": True,
        },
    }

    dimensions = _build_decision_dimensions(claim)

    assert "expense_limits" in dimensions
    assert "room_limit" in dimensions
    assert "doctor_fee_limit" in dimensions
    assert "medical_expense_limit" in dimensions
    assert "ambulance_limit" in dimensions


def test_case_agent_detects_cosmetic_exclusion():
    claim = {
        "case_id": "TEST-002",
        "hospitalization_type": "inpatient",
        "hospitalization_hours": 72,
        "condition": "cosmetic condition",
        "procedure": "cosmetic surgery",
        "expenses": {},
        "evidence_context": {},
    }

    dimensions = _build_decision_dimensions(claim)

    assert "cosmetic_exclusion" in dimensions


def test_case_agent_detects_experimental_treatment():
    claim = {
        "case_id": "TEST-003",
        "hospitalization_type": "inpatient",
        "hospitalization_hours": 72,
        "condition": "experimental condition",
        "procedure": "experimental therapy",
        "expenses": {},
        "evidence_context": {},
    }

    dimensions = _build_decision_dimensions(claim)

    assert "experimental_treatment" in dimensions


def test_case_agent_detects_pre_post_hospitalization():
    claim = {
        "case_id": "TEST-004",
        "hospitalization_type": "inpatient",
        "hospitalization_hours": 72,
        "condition": "acute infection",
        "procedure": "medical treatment",
        "expenses": {
            "pre_hospitalization": 5000,
            "post_hospitalization": 7000,
        },
        "evidence_context": {},
    }

    dimensions = _build_decision_dimensions(claim)

    assert "expense_limits" in dimensions
    assert "pre_post_hospitalization" in dimensions


def test_case_agent_detects_hospital_and_medical_necessity_evidence():
    claim = {
        "case_id": "TEST-005",
        "hospitalization_type": "inpatient",
        "hospitalization_hours": 72,
        "condition": "acute infection",
        "procedure": "medical treatment",
        "expenses": {},
        "evidence_context": {
            "hospital_registered": None,
            "medical_necessity_confirmed": None,
        },
    }

    dimensions = _build_decision_dimensions(claim)

    assert "hospital_definition" in dimensions
    assert "medical_necessity" in dimensions


# ================================================================
# VALIDATION AGENT TESTS
# ================================================================


def _valid_evidence():
    return [
        {
            "source": "policy",
            "page": 7,
            "section": "Unspecified",
            "chunk_id": "policy_p07_unspecified_021",
            "text": "Room and medical practitioner expense limits apply.",
        }
    ]


def _valid_finding():
    return {
        "dimension": "room_limit",
        "statement": "The policy imposes a room expense limit.",
        "status": "SUPPORTED",
        "citations": [
            {
                "source": "policy",
                "page": 7,
                "section": "Unspecified",
                "chunk_id": "policy_p07_unspecified_021",
            }
        ],
    }


def test_validation_accepts_supported_finding():
    evidence = _valid_evidence()

    locations = _build_evidence_locations(evidence)

    issues = _validate_finding(
        _valid_finding(),
        locations,
    )

    assert issues == []


def test_validation_rejects_missing_citation():
    finding = {
        "dimension": "room_limit",
        "statement": "The policy imposes a room expense limit.",
        "status": "SUPPORTED",
        "citations": [],
    }

    locations = _build_evidence_locations(
        _valid_evidence()
    )

    issues = _validate_finding(
        finding,
        locations,
    )

    assert len(issues) > 0
    assert "no policy citation" in issues[0].lower()


def test_validation_rejects_citation_not_in_evidence():
    finding = {
        "dimension": "room_limit",
        "statement": "The policy imposes a room expense limit.",
        "status": "SUPPORTED",
        "citations": [
            {
                "source": "policy",
                "page": 99,
                "section": "Wrong Section",
                "chunk_id": "fake_chunk",
            }
        ],
    }

    locations = _build_evidence_locations(
        _valid_evidence()
    )

    issues = _validate_finding(
        finding,
        locations,
    )

    assert len(issues) > 0
    assert any(
        "does not map to retrieved policy evidence" in issue
        for issue in issues
    )


def test_validation_rejects_positive_decision_with_missing_evidence():
    state = {
        "decision": "ADMISSIBLE",
        "missing_evidence": [
            "Evidence establishing hospital registration."
        ],
    }

    issues = _validate_decision_safety(state)

    assert len(issues) == 1
    assert "missing evidence" in issues[0].lower()


def test_validation_accepts_needs_review_with_missing_evidence():
    state = {
        "decision": "NEEDS_REVIEW",
        "missing_evidence": [
            "Evidence establishing hospital registration."
        ],
    }

    issues = _validate_decision_safety(state)

    assert issues == []


def test_validation_rejects_invalid_decision():
    state = {
        "decision": "INVALID_DECISION",
        "missing_evidence": [],
    }

    issues = _validate_decision_safety(state)

    assert len(issues) == 1
    assert "invalid decision status" in issues[0].lower()


def test_validation_agent_returns_pass_for_valid_state():
    evidence = _valid_evidence()

    state = {
        "decision": "ADMISSIBLE_WITH_LIMITS",
        "missing_evidence": [],
        "retrieved_evidence": evidence,
        "coverage_findings": [
            _valid_finding()
        ],
        "exclusion_findings": [],
        "limit_findings": [],
        "citations": [
            {
                "source": "policy",
                "page": 7,
                "section": "Unspecified",
                "chunk_id": "policy_p07_unspecified_021",
            }
        ],
        "trace": [],
    }

    result = validation_agent(state)

    assert result["validation_status"] == "PASS"
    assert result["unsupported_claims"] == []
    assert result["decision"] == "ADMISSIBLE_WITH_LIMITS"


def test_validation_agent_forces_review_on_unsupported_positive_decision():
    state = {
        "decision": "ADMISSIBLE",
        "missing_evidence": [],
        "retrieved_evidence": [],
        "coverage_findings": [
            {
                "dimension": "room_limit",
                "statement": "The policy imposes a room limit.",
                "status": "SUPPORTED",
                "citations": [],
            }
        ],
        "exclusion_findings": [],
        "limit_findings": [],
        "citations": [],
        "trace": [],
    }

    result = validation_agent(state)

    assert result["validation_status"] == "FAIL"
    assert result["decision"] == "NEEDS_REVIEW"
    assert len(result["unsupported_claims"]) > 0
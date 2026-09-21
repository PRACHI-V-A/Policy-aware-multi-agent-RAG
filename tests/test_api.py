from fastapi.testclient import TestClient

import src.api as api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_returns_structured_response(monkeypatch):
    fake_result = {
        "case_id": "TEST-001",
        "decision": "NEEDS_REVIEW",
        "confidence": 0.45,
        "key_findings": ["Hospital eligibility evidence is incomplete."],
        "coverage_findings": [],
        "exclusion_findings": [],
        "limit_findings": [],
        "missing_evidence": ["hospital_registered"],
        "unsupported_claims": [],
        "citations": [
            {
                "source": "policy.pdf",
                "page": 3,
                "section": "Hospital Definition",
                "chunk_id": "policy_p03_unspecified_008",
            }
        ],
        "validation_status": "PASS",
        "trace": [
            {"agent": "Case Analysis Agent", "action": "Identified hospital eligibility."},
            {"agent": "Policy Evidence Agent", "action": "Retrieved policy evidence."},
        ],
        "decision_dimensions": ["hospital_definition"],
        "investigation_plan": ["Verify hospital eligibility."],
        "expense_calculation": {},
    }

    monkeypatch.setattr(api, "analyze_claim", lambda claim, workflow=None: fake_result)
    api.get_workflow.cache_clear()

    response = client.post(
        "/analyze",
        json={
            "case_id": "TEST-001",
            "policy_id": "USGIC-CSC-2017-2018",
            "hospital": {"name": "Test Hospital"},
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["case_id"] == "TEST-001"
    assert body["decision"] == "NEEDS_REVIEW"
    assert body["validation_status"] == "PASS"
    assert body["citations"][0]["chunk_id"] == "policy_p03_unspecified_008"
    assert body["missing_evidence"] == ["hospital_registered"]

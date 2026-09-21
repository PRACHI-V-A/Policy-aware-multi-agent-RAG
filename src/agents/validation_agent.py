from __future__ import annotations

from typing import Any


VALID_DECISIONS = {
    "ADMISSIBLE",
    "ADMISSIBLE_WITH_LIMITS",
    "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE",
    "NEEDS_REVIEW",
}


# ================================================================
# CITATION HELPERS
# ================================================================

def _citation_location(
    citation: dict[str, Any]
) -> tuple:

    section = str(
        citation.get("section", "")
    ).strip().lower()

    chunk_id = str(
        citation.get("chunk_id", "")
    ).strip().lower()

    return (
        citation.get("page"),
        section,
        chunk_id,
    )


def _citation_is_structurally_valid(
    citation: dict[str, Any]
) -> bool:

    required_fields = [
        "source",
        "page",
        "section",
        "chunk_id",
    ]

    return all(
        citation.get(field) not in (None, "")
        for field in required_fields
    )


def _build_evidence_locations(
    evidence: list[dict[str, Any]]
) -> set[tuple]:

    locations = set()

    for item in evidence:

        # Use the SAME normalization as citations.
        location = _citation_location(item)

        if all(
            value not in (None, "")
            for value in location
        ):
            locations.add(location)

    return locations


def _citation_matches_evidence(
    citation: dict[str, Any],
    evidence_locations: set[tuple],
) -> bool:

    return (
        _citation_location(citation)
        in evidence_locations
    )


# ================================================================
# FINDING VALIDATION
# ================================================================

def _validate_finding(
    finding: dict[str, Any],
    evidence_locations: set[tuple],
) -> list[str]:

    issues = []

    statement = finding.get("statement")

    if not statement:

        issues.append(
            "A policy finding does not contain a statement."
        )

        return issues

    citations = finding.get(
        "citations",
        []
    )

    if not citations:

        issues.append(
            f"Finding has no policy citation: {statement}"
        )

        return issues

    for citation in citations:

        if not _citation_is_structurally_valid(
            citation
        ):

            issues.append(
                "A finding contains an incomplete policy citation."
            )

            continue

        if not _citation_matches_evidence(
            citation,
            evidence_locations,
        ):

            issues.append(
                "A finding citation does not map to "
                "retrieved policy evidence."
            )

    return issues


# ================================================================
# FINAL CITATION VALIDATION
# ================================================================

def _validate_final_citations(
    citations: list[dict[str, Any]],
    evidence_locations: set[tuple],
) -> list[str]:

    issues = []

    for citation in citations:

        if not _citation_is_structurally_valid(
            citation
        ):

            issues.append(
                "A final citation is missing required metadata."
            )

            continue

        if not _citation_matches_evidence(
            citation,
            evidence_locations,
        ):

            issues.append(
                "A final citation does not match "
                "any retrieved policy evidence."
            )

    return issues


# ================================================================
# DECISION SAFETY
# ================================================================

def _validate_decision_safety(
    state: dict[str, Any]
) -> list[str]:

    issues = []

    decision = state.get(
        "decision"
    )

    missing_evidence = state.get(
        "missing_evidence",
        []
    )

    if decision not in VALID_DECISIONS:

        issues.append(
            f"Invalid decision status: {decision}"
        )

        return issues

    positive_decisions = {
        "ADMISSIBLE",
        "ADMISSIBLE_WITH_LIMITS",
        "PARTIALLY_ADMISSIBLE",
    }

    if (
        decision in positive_decisions
        and missing_evidence
    ):

        issues.append(
            "A positive decision was produced despite "
            "explicitly identified missing evidence."
        )

    return issues


# ================================================================
# EXCLUSION SAFETY
# ================================================================

def _validate_exclusion_safety(
    state: dict[str, Any],
    evidence_locations: set[tuple],
) -> list[str]:

    issues = []

    exclusions = state.get(
        "exclusion_findings",
        []
    )

    for finding in exclusions:

        if finding.get("status") != "SUPPORTED":
            continue

        issues.extend(
            _validate_finding(
                finding,
                evidence_locations,
            )
        )

    return issues


# ================================================================
# MAIN VALIDATION AGENT
# ================================================================

def validation_agent(
    state: dict[str, Any]
) -> dict[str, Any]:

    retrieved_evidence = state.get(
        "retrieved_evidence",
        []
    )

    evidence_locations = (
        _build_evidence_locations(
            retrieved_evidence
        )
    )

    unsupported_claims = []

    # ------------------------------------------------------------
    # Coverage findings
    # ------------------------------------------------------------

    for finding in state.get(
        "coverage_findings",
        []
    ):

        unsupported_claims.extend(
            _validate_finding(
                finding,
                evidence_locations,
            )
        )

    # ------------------------------------------------------------
    # Exclusion findings
    # ------------------------------------------------------------

    unsupported_claims.extend(
        _validate_exclusion_safety(
            state,
            evidence_locations,
        )
    )

    # ------------------------------------------------------------
    # Limit findings
    # ------------------------------------------------------------

    for finding in state.get(
        "limit_findings",
        []
    ):

        unsupported_claims.extend(
            _validate_finding(
                finding,
                evidence_locations,
            )
        )

    # ------------------------------------------------------------
    # Final citations
    # ------------------------------------------------------------

    unsupported_claims.extend(
        _validate_final_citations(
            state.get(
                "citations",
                []
            ),
            evidence_locations,
        )
    )

    # ------------------------------------------------------------
    # Decision safety
    # ------------------------------------------------------------

    unsupported_claims.extend(
        _validate_decision_safety(
            state
        )
    )

    # Remove duplicates while preserving order
    unsupported_claims = list(
        dict.fromkeys(
            unsupported_claims
        )
    )

    validation_status = (
        "FAIL"
        if unsupported_claims
        else "PASS"
    )

    final_decision = state.get(
        "decision",
        "NEEDS_REVIEW"
    )

    # ------------------------------------------------------------
    # Only material validation failures force abstention.
    # ------------------------------------------------------------

    material_failure = False

    for issue in unsupported_claims:

        issue_lower = issue.lower()

        if any(
            phrase in issue_lower
            for phrase in [
                "positive decision was produced",
                "invalid decision status",
                "no policy citation",
                "does not map to retrieved policy evidence",
            ]
        ):

            material_failure = True
            break

    if material_failure:

        final_decision = "NEEDS_REVIEW"

    # ------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    trace.append(
        {
            "agent": "ValidationAgent",
            "action": (
                "Validated evidence-to-finding citations "
                "and decision safety."
            ),
            "validation_status": validation_status,
            "unsupported_claims": len(
                unsupported_claims
            ),
        }
    )

    return {
        "decision": final_decision,

        "validation_status":
            validation_status,

        "unsupported_claims":
            unsupported_claims,

        "trace":
            trace,
    }
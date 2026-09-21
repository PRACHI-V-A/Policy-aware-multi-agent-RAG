from __future__ import annotations

from typing import Any


# ================================================================
# BASIC HELPERS
# ================================================================

def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _find_evidence_for_dimension(
    evidence: list[dict[str, Any]],
    dimension: str,
) -> list[dict[str, Any]]:
    """
    Return all retrieved policy chunks associated with a decision dimension.
    """

    target = _normalize(dimension)

    matches = []

    for item in evidence:
        dimensions = item.get("dimensions") or []

        normalized_dimensions = {
            _normalize(d)
            for d in dimensions
        }

        if target in normalized_dimensions:
            matches.append(item)

    return matches


def _make_citation(
    item: dict[str, Any]
) -> dict[str, Any]:

    return {
        "source": item.get("source"),
        "page": item.get("page"),
        "section": item.get("section"),
        "chunk_id": item.get("chunk_id"),
    }


def _make_finding(
    *,
    dimension: str,
    finding_type: str,
    statement: str,
    item: dict[str, Any],
    status: str = "SUPPORTED",
) -> dict[str, Any]:

    citation = _make_citation(item)

    return {
        "dimension": dimension,
        "finding_type": finding_type,
        "status": status,
        "statement": statement,

        # Keep both forms for compatibility with the
        # existing validation layer.
        "citation": citation,
        "citations": [citation],
    }


def _evidence_text(
    item: dict[str, Any]
) -> str:

    return " ".join(
        [
            str(item.get("text", "")),
            str(item.get("content", "")),
            str(item.get("section", "")),
            str(item.get("title", "")),
            str(item.get("chunk_id", "")),
        ]
    ).lower()


def _sort_evidence(
    evidence: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    return sorted(
        evidence,
        key=lambda x: (
            x.get("rank", 999),
            -(x.get("rerank_score") or -999),
        ),
    )


# ================================================================
# FINDING GENERATION
# ================================================================

def _append_findings_from_dimension(
    *,
    dimension: str,
    evidence: list[dict[str, Any]],
    coverage_findings: list[dict[str, Any]],
    exclusion_findings: list[dict[str, Any]],
    limit_findings: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> None:

    matching_evidence = _find_evidence_for_dimension(
        evidence,
        dimension,
    )

    if not matching_evidence:
        return

    matching_evidence = _sort_evidence(
        matching_evidence
    )

    # ============================================================
    # COSMETIC EXCLUSION
    # ============================================================

    if dimension == "cosmetic_exclusion":

        # IMPORTANT:
        # Do NOT inspect only the top-ranked chunk.
        #
        # The relevant exclusion may appear in a lower-ranked
        # retrieved chunk. Search all evidence associated with
        # this dimension.

        cosmetic_keywords = [
            "cosmetic",
            "aesthetic",
            "plastic surgery",
            "beautification",
            "appearance",
        ]

        cosmetic_evidence = [
            item
            for item in matching_evidence
            if any(
                keyword in _evidence_text(item)
                for keyword in cosmetic_keywords
            )
        ]

        if cosmetic_evidence:

            primary = cosmetic_evidence[0]

            finding = _make_finding(
                dimension=dimension,
                finding_type="EXCLUSION",
                statement=(
                    "The retrieved policy evidence contains an exclusion "
                    "for cosmetic or aesthetic treatment and plastic surgery, "
                    "subject to the exceptions stated in the policy."
                ),
                item=primary,
            )

            exclusion_findings.append(
                finding
            )

            citations.append(
                _make_citation(primary)
            )

        return

    # ============================================================
    # EXPERIMENTAL / UNPROVEN TREATMENT EXCLUSION
    # ============================================================

    if dimension == "experimental_treatment":

        experimental_keywords = [
            "experimental",
            "unproven",
            "investigational",
            "experimental therapy",
            "unproven treatment",
        ]

        experimental_evidence = [
            item
            for item in matching_evidence
            if any(
                keyword in _evidence_text(item)
                for keyword in experimental_keywords
            )
        ]

        if experimental_evidence:

            primary = experimental_evidence[0]

            finding = _make_finding(
                dimension=dimension,
                finding_type="EXCLUSION",
                statement=(
                    "The retrieved policy evidence addresses an exclusion "
                    "for experimental or unproven treatment."
                ),
                item=primary,
            )

            exclusion_findings.append(
                finding
            )

            citations.append(
                _make_citation(primary)
            )

        return

    # ============================================================
    # PRE-EXISTING DISEASE
    # ============================================================

    if dimension == "pre_existing_disease":

        finding = _make_finding(
            dimension=dimension,
            finding_type="WAITING_PERIOD",
            statement=(
                "The retrieved policy evidence establishes a waiting-period "
                "provision for pre-existing diseases."
            ),
            item=matching_evidence[0],
        )

        exclusion_findings.append(
            finding
        )

        citations.append(
            _make_citation(matching_evidence[0])
        )

        return

    # ============================================================
    # HOSPITALIZATION ELIGIBILITY
    # ============================================================

    if dimension == "hospitalization_eligibility":

        finding = _make_finding(
            dimension=dimension,
            finding_type="COVERAGE",
            statement=(
                "The retrieved policy evidence defines the conditions under "
                "which hospitalization expenses may be admissible."
            ),
            item=matching_evidence[0],
        )

        coverage_findings.append(
            finding
        )

        citations.append(
            _make_citation(matching_evidence[0])
        )

        return

    # ============================================================
    # HOSPITAL DEFINITION
    # ============================================================

    if dimension == "hospital_definition":

        finding = _make_finding(
            dimension=dimension,
            finding_type="ELIGIBILITY",
            statement=(
                "The retrieved policy evidence specifies the requirements "
                "for a facility to qualify as a hospital."
            ),
            item=matching_evidence[0],
        )

        coverage_findings.append(
            finding
        )

        citations.append(
            _make_citation(matching_evidence[0])
        )

        return

    # ============================================================
    # DAY CARE
    # ============================================================

    if dimension == "day_care":

        finding = _make_finding(
            dimension=dimension,
            finding_type="COVERAGE",
            statement=(
                "The retrieved policy evidence defines eligibility for "
                "specified day-care treatment performed in less than 24 hours."
            ),
            item=matching_evidence[0],
        )

        coverage_findings.append(
            finding
        )

        citations.append(
            _make_citation(matching_evidence[0])
        )

        return

    # ============================================================
    # DOMICILIARY TREATMENT
    # ============================================================

    if dimension in {
        "domiciliary_treatment",
        "domiciliary",
    }:

        finding = _make_finding(
            dimension=dimension,
            finding_type="COVERAGE",
            statement=(
                "The retrieved policy evidence specifies conditions and "
                "limits applicable to domiciliary hospitalization."
            ),
            item=matching_evidence[0],
        )

        coverage_findings.append(
            finding
        )

        citations.append(
            _make_citation(matching_evidence[0])
        )

        return

    # ============================================================
    # EXPENSE LIMITS
    # ============================================================

    if dimension in {
        "expense_limits",
        "room_limit",
        "doctor_fee_limit",
        "medical_expense_limit",
        "ambulance_limit",
        "pre_post_hospitalization",
    }:

        finding = _make_finding(
            dimension=dimension,
            finding_type="LIMIT",
            statement=(
                "The retrieved policy evidence contains an applicable "
                "expense limitation or condition for this category."
            ),
            item=matching_evidence[0],
        )

        limit_findings.append(
            finding
        )

        citations.append(
            _make_citation(matching_evidence[0])
        )

        return


# ================================================================
# COVERAGE & EXCLUSION AGENT
# ================================================================

def coverage_exclusion_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    claim = state.get(
        "claim_facts"
    ) or {}

    evidence = state.get(
        "retrieved_evidence"
    ) or []

    dimensions = state.get(
        "decision_dimensions"
    ) or []

    coverage_findings: list[dict[str, Any]] = []

    exclusion_findings: list[dict[str, Any]] = []

    limit_findings: list[dict[str, Any]] = []

    citations: list[dict[str, Any]] = []

    trace = list(
        state.get("trace")
        or []
    )

    # ============================================================
    # ANALYZE EVERY DECISION DIMENSION
    # ============================================================

    for dimension in dimensions:

        _append_findings_from_dimension(
            dimension=dimension,
            evidence=evidence,
            coverage_findings=coverage_findings,
            exclusion_findings=exclusion_findings,
            limit_findings=limit_findings,
            citations=citations,
        )

    # ============================================================
    # DEDUPLICATE CITATIONS
    # ============================================================

    unique_citations = []

    seen = set()

    for citation in citations:

        key = (
            citation.get("source"),
            citation.get("page"),
            _normalize(
                citation.get("section")
            ),
            _normalize(
                citation.get("chunk_id")
            ),
        )

        if key not in seen:

            seen.add(key)

            unique_citations.append(
                citation
            )

    # ============================================================
    # TRACE
    # ============================================================

    trace.append(
        {
            "agent": "CoverageExclusionAgent",
            "action": (
                "Analyzed retrieved policy evidence against claim "
                "conditions and produced structured coverage, "
                "exclusion and limit findings."
            ),
            "coverage_findings_count":
                len(coverage_findings),
            "exclusion_findings_count":
                len(exclusion_findings),
            "limit_findings_count":
                len(limit_findings),
        }
    )

    return {
        "coverage_findings":
            coverage_findings,

        "exclusion_findings":
            exclusion_findings,

        "limit_findings":
            limit_findings,

        "citations":
            unique_citations,

        "trace":
            trace,
    }


# ================================================================
# COMPATIBILITY WRAPPER
# ================================================================

def coverage_analysis_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    return coverage_exclusion_agent(
        state
    )
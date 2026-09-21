from __future__ import annotations

from datetime import datetime
from typing import Any


VALID_DECISIONS = {
    "ADMISSIBLE",
    "ADMISSIBLE_WITH_LIMITS",
    "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE",
    "NEEDS_REVIEW",
}


# ================================================================
# BASIC HELPERS
# ================================================================

def _parse_date(value: Any):
    if not value:
        return None

    try:
        return datetime.strptime(
            str(value),
            "%Y-%m-%d"
        ).date()
    except (ValueError, TypeError):
        return None


def calculate_days_between(
    start_date: Any,
    end_date: Any,
) -> int | None:

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    if start is None or end is None:
        return None

    return (end - start).days


def _citation_key(
    citation: dict[str, Any]
) -> tuple:

    return (
        citation.get("source"),
        citation.get("page"),
        citation.get("section"),
        citation.get("chunk_id"),
    )


def _deduplicate_citations(
    citations: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    result = []
    seen = set()

    for citation in citations:

        key = _citation_key(citation)

        if key not in seen:
            result.append(citation)
            seen.add(key)

    return result


# ================================================================
# FIND POLICY EVIDENCE
# ================================================================

def _get_retrieved_evidence(
    state: dict[str, Any]
) -> list[dict[str, Any]]:

    return state.get(
        "retrieved_evidence",
        []
    )


def _evidence_text(
    evidence: dict[str, Any]
) -> str:

    parts = [
        str(evidence.get("text", "")),
        str(evidence.get("content", "")),
        str(evidence.get("section", "")),
        str(evidence.get("title", "")),
        str(evidence.get("chunk_id", "")),
    ]

    return " ".join(parts).lower()


def _citation_from_evidence(
    evidence: dict[str, Any]
) -> dict[str, Any]:

    return {
        "source": evidence.get(
            "source",
            "supplied_policy"
        ),
        "page": evidence.get("page"),
        "section": evidence.get(
            "section",
            "Unspecified"
        ),
        "chunk_id": evidence.get(
            "chunk_id"
        ),
    }


def get_waiting_period_evidence(
    state: dict[str, Any]
) -> list[dict[str, Any]]:

    evidence = _get_retrieved_evidence(state)

    results = []

    for item in evidence:

        chunk_id = str(
            item.get("chunk_id", "")
        ).lower()

        text = _evidence_text(item)

        # Strongest match: known policy waiting-period chunk.
        if "waiting" in chunk_id:

            results.append(
                _citation_from_evidence(item)
            )
            continue

        # Generic lexical fallback.
        waiting_keywords = [
            "waiting period",
            "initial waiting",
            "30 days",
            "thirty days",
        ]

        if any(
            keyword in text
            for keyword in waiting_keywords
        ):

            results.append(
                _citation_from_evidence(item)
            )

    # If retrieved evidence didn't expose a useful text field,
    # use existing state citations as fallback.
    if not results:

        for citation in state.get(
            "citations",
            []
        ):

            text = str(citation).lower()

            if (
                "waiting" in text
                or citation.get("chunk_id")
                == "policy_p09_what_we_exclude_029"
            ):
                results.append(citation)

    return _deduplicate_citations(results)


# ================================================================
# DIMENSION CITATIONS
# ================================================================

def get_dimension_citations(
    state: dict[str, Any],
    dimensions: list[str],
) -> list[dict[str, Any]]:

    citations = []

    for finding_group in (
        state.get("coverage_findings", []),
        state.get("exclusion_findings", []),
        state.get("limit_findings", []),
    ):

        for finding in finding_group:

            dimension = str(
                finding.get(
                    "dimension",
                    ""
                )
            ).lower()

            statement = str(
                finding.get(
                    "statement",
                    ""
                )
            ).lower()

            combined = (
                dimension
                + " "
                + statement
            )

            if any(
                d.lower() in combined
                for d in dimensions
            ):

                citations.extend(
                    finding.get(
                        "citations",
                        []
                    )
                )

    return _deduplicate_citations(
        citations
    )


# ================================================================
# WAITING PERIOD EVALUATION
# ================================================================

def evaluate_waiting_period(
    state: dict[str, Any]
) -> dict[str, Any]:

    claim = state.get(
        "claim_facts",
        {}
    )

    policy_start = claim.get(
        "policy_start_date"
    )

    claim_date = claim.get(
        "claim_date"
    )

    elapsed_days = calculate_days_between(
        policy_start,
        claim_date
    )

    waiting_citations = get_waiting_period_evidence(
        state
    )

    # ------------------------------------------------------------
    # Missing policy evidence
    # ------------------------------------------------------------

    if not waiting_citations:

        return {
            "status": "NEEDS_REVIEW",
            "reason": (
                "Applicable waiting-period policy evidence "
                "was not retrieved."
            ),
            "citations": [],
            "elapsed_days": elapsed_days,
        }

    # ------------------------------------------------------------
    # Missing dates
    # ------------------------------------------------------------

    if elapsed_days is None:

        return {
            "status": "NEEDS_REVIEW",
            "reason": (
                "Policy start date or claim date is "
                "missing or invalid."
            ),
            "citations": waiting_citations,
            "elapsed_days": None,
        }

    # ------------------------------------------------------------
    # Initial 30-day waiting period
    # ------------------------------------------------------------

    if elapsed_days < 30:

        evidence_context = claim.get(
            "evidence_context",
            {}
        )

        prior_years = claim.get(
            "prior_insurer_continuous_years",
            0
        )

        # --------------------------------------------------------
        # Prior-insurance exception
        # --------------------------------------------------------

        if prior_years and prior_years >= 1:

            continuity = evidence_context.get(
                "previous_policy_continuity"
            )

            claim_history = evidence_context.get(
                "previous_claim_history_received"
            )

            # Explicit negative evidence
            if (
                continuity is False
                or claim_history is False
            ):

                return {
                    "status": "NOT_ADMISSIBLE",
                    "reason": (
                        "The claim falls within the initial "
                        "waiting period and the available evidence "
                        "does not establish the required prior "
                        "continuous insurance exception."
                    ),
                    "citations": waiting_citations,
                    "elapsed_days": elapsed_days,
                }

            # Exception potentially applicable but not proven
            return {
                "status": "NEEDS_REVIEW",
                "reason": (
                    "The claim falls within the initial waiting "
                    "period. A prior-insurance exception may apply, "
                    "but the required continuity evidence is not "
                    "fully established."
                ),
                "citations": waiting_citations,
                "elapsed_days": elapsed_days,
            }

        # --------------------------------------------------------
        # Normal initial waiting-period rejection
        # --------------------------------------------------------

        return {
            "status": "NOT_ADMISSIBLE",
            "reason": (
                "The claim occurred within the policy's "
                "initial 30-day waiting period."
            ),
            "citations": waiting_citations,
            "elapsed_days": elapsed_days,
        }

    # ------------------------------------------------------------
    # Waiting period satisfied
    # ------------------------------------------------------------

    return {
        "status": "WAITING_PERIOD_SATISFIED",
        "reason": (
            "The claim occurred after the initial "
            "30-day waiting period."
        ),
        "citations": waiting_citations,
        "elapsed_days": elapsed_days,
    }


# ================================================================
# MAIN DECISION AGENT
# ================================================================

def decision_agent(
    state: dict[str, Any]
) -> dict[str, Any]:

    claim = state.get(
        "claim_facts",
        {}
    )

    coverage_findings = state.get(
        "coverage_findings",
        []
    )

    exclusion_findings = state.get(
        "exclusion_findings",
        []
    )

    limit_findings = state.get(
        "limit_findings",
        []
    )

    missing_evidence = state.get(
        "missing_evidence",
        []
    )

    citations = list(
        state.get(
            "citations",
            []
        )
    )

    trace = list(
        state.get(
            "trace",
            []
        )
    )

    key_findings = []

    # ============================================================
    # 1. WAITING PERIOD
    # ============================================================

    waiting_result = evaluate_waiting_period(
        state
    )

    waiting_status = waiting_result.get(
        "status"
    )

    waiting_citations = waiting_result.get(
        "citations",
        []
    )

    citations.extend(
        waiting_citations
    )

    waiting_reason = waiting_result.get(
        "reason"
    )

    if waiting_reason:
        key_findings.append(
            waiting_reason
        )

    # ------------------------------------------------------------
    # Waiting period rejection
    # ------------------------------------------------------------

    if waiting_status == "NOT_ADMISSIBLE":

        trace.append({
            "agent": "DecisionAgent",
            "action": "decision",
            "decision": "NOT_ADMISSIBLE",
            "reason": waiting_reason,
            "waiting_period_days":
                waiting_result.get(
                    "elapsed_days"
                ),
        })

        return {
            "decision": "NOT_ADMISSIBLE",
            "confidence": 0.90,
            "key_findings": key_findings,
            "citations":
                _deduplicate_citations(
                    citations
                ),
            "expense_calculation": {},
            "trace": trace,
        }

    # ------------------------------------------------------------
    # Waiting exception not established
    # ------------------------------------------------------------

    if waiting_status == "NEEDS_REVIEW":

        trace.append({
            "agent": "DecisionAgent",
            "action": "decision",
            "decision": "NEEDS_REVIEW",
            "reason": waiting_reason,
        })

        return {
            "decision": "NEEDS_REVIEW",
            "confidence": 0.45,
            "key_findings": key_findings,
            "citations":
                _deduplicate_citations(
                    citations
                ),
            "expense_calculation": {},
            "trace": trace,
        }

    # ============================================================
    # 2. HARD POLICY EXCLUSIONS
    #
    # These MUST happen before expense limits.
    # ============================================================

    supported_exclusions = []

    for finding in exclusion_findings:

        finding_citations = finding.get(
            "citations",
            []
        )

        if not finding_citations:
            continue

        dimension = str(
            finding.get(
                "dimension",
                ""
            )
        ).lower()

        statement = str(
            finding.get(
                "statement",
                ""
            )
        )

        supported_exclusions.append({
            "dimension": dimension,
            "statement": statement,
            "citations": finding_citations,
        })

    # ============================================================
    # 2A. COSMETIC EXCLUSION
    # ============================================================

    cosmetic_items = [
        item
        for item in supported_exclusions
        if (
            "cosmetic" in item["dimension"]
            or "cosmetic" in
            item["statement"].lower()
        )
    ]

    if cosmetic_items:

        cosmetic_citations = []

        for item in cosmetic_items:
            cosmetic_citations.extend(
                item["citations"]
            )

        citations.extend(
            cosmetic_citations
        )

        key_findings.append(
            "The retrieved policy contains an applicable "
            "cosmetic-treatment exclusion."
        )

        trace.append({
            "agent": "DecisionAgent",
            "action": "decision",
            "decision": "NOT_ADMISSIBLE",
            "reason":
                "Supported cosmetic-treatment exclusion.",
        })

        return {
            "decision": "NOT_ADMISSIBLE",
            "confidence": 0.90,
            "key_findings": key_findings,
            "citations":
                _deduplicate_citations(
                    citations
                ),
            "expense_calculation": {},
            "trace": trace,
        }

    # ============================================================
    # 2B. EXPERIMENTAL / UNPROVEN EXCLUSION
    # ============================================================

    experimental_items = [
        item
        for item in supported_exclusions
        if (
            "experimental"
            in item["dimension"]
            or "experimental"
            in item["statement"].lower()
            or "unproven"
            in item["statement"].lower()
        )
    ]

    if experimental_items:

        experimental_citations = []

        for item in experimental_items:
            experimental_citations.extend(
                item["citations"]
            )

        citations.extend(
            experimental_citations
        )

        key_findings.append(
            "The retrieved policy contains an applicable "
            "experimental or unproven treatment exclusion."
        )

        trace.append({
            "agent": "DecisionAgent",
            "action": "decision",
            "decision": "NOT_ADMISSIBLE",
            "reason":
                "Supported experimental/unproven treatment exclusion.",
        })

        return {
            "decision": "NOT_ADMISSIBLE",
            "confidence": 0.90,
            "key_findings": key_findings,
            "citations":
                _deduplicate_citations(
                    citations
                ),
            "expense_calculation": {},
            "trace": trace,
        }

    # ============================================================
    # 2C. PRE-EXISTING DISEASE
    # ============================================================

    pre_existing_items = [
        item
        for item in supported_exclusions
        if (
            "pre_existing"
            in item["dimension"]
            or "pre-existing"
            in item["statement"].lower()
            or "pre existing"
            in item["statement"].lower()
        )
    ]

    if pre_existing_items:

        pre_existing_citations = []

        for item in pre_existing_items:
            pre_existing_citations.extend(
                item["citations"]
            )

        citations.extend(
            pre_existing_citations
        )

        key_findings.append(
            "The retrieved policy contains an applicable "
            "pre-existing-disease exclusion or waiting provision."
        )

        trace.append({
            "agent": "DecisionAgent",
            "action": "decision",
            "decision": "NOT_ADMISSIBLE",
            "reason":
                "Supported pre-existing-disease exclusion.",
        })

        return {
            "decision": "NOT_ADMISSIBLE",
            "confidence": 0.90,
            "key_findings": key_findings,
            "citations":
                _deduplicate_citations(
                    citations
                ),
            "expense_calculation": {},
            "trace": trace,
        }

    # ============================================================
    # 3. MISSING EVIDENCE
    # ============================================================

    if missing_evidence:

        key_findings.append(
            "Required evidence is missing for one or more "
            "material policy conditions."
        )

        trace.append({
            "agent": "DecisionAgent",
            "action": "decision",
            "decision": "NEEDS_REVIEW",
            "reason":
                "Required evidence is missing.",
            "missing_evidence":
                missing_evidence,
        })

        return {
            "decision": "NEEDS_REVIEW",
            "confidence": 0.45,
            "key_findings": key_findings,
            "citations":
                _deduplicate_citations(
                    citations
                ),
            "expense_calculation": {},
            "trace": trace,
        }

    # ============================================================
    # 4. EXPENSE LIMITS
    # ============================================================

    expense_calculation = calculate_expense_limits(
        claim
    )

    if expense_calculation:

        citations.extend(
            expense_calculation.get(
                "citations",
                []
            )
        )

        total_deduction = expense_calculation.get(
            "total_deduction_inr",
            0
        )

        # If policy limit findings exist, the decision should explicitly
        # reflect that limits apply, even when the current claimed amounts
        # do not exceed those limits.
        if limit_findings:

            final_decision = "ADMISSIBLE_WITH_LIMITS"
            confidence = 0.82

            key_findings.append(
                "Applicable policy provisions impose limits "
                "on one or more expense categories."
            )

        else:

            final_decision = "ADMISSIBLE"
            confidence = 0.85

    else:

        # ========================================================
        # 5. GENERAL COVERAGE
        # ========================================================

        if coverage_findings:

            final_decision = "ADMISSIBLE"
            confidence = 0.85

            key_findings.append(
                "The retrieved policy evidence supports "
                "the applicable coverage conditions."
            )

        else:

            final_decision = "NEEDS_REVIEW"
            confidence = 0.45

            key_findings.append(
                "The available policy evidence is insufficient "
                "to establish a final claim decision."
            )

    # ============================================================
    # 6. LIMIT FINDINGS
    # ============================================================

    for finding in limit_findings:

        if finding.get("citations"):
            citations.extend(
                finding["citations"]
            )

    # ============================================================
    # 7. FINAL TRACE
    # ============================================================

    trace.append({
        "agent": "DecisionAgent",
        "action": "decision",
        "decision": final_decision,
        "confidence": confidence,
        "reason": (
            "Decision derived from waiting-period evaluation, "
            "supported policy exclusions, evidence sufficiency, "
            "coverage findings, and applicable expense limits."
        ),
    })

    return {
        "decision": final_decision,
        "confidence": confidence,
        "key_findings": key_findings,
        "citations":
            _deduplicate_citations(
                citations
            ),
        "expense_calculation":
            expense_calculation,
        "trace": trace,
    }


# ================================================================
# EXPENSE CALCULATIONS
# ================================================================

def calculate_expense_limits(
    claim: dict[str, Any]
) -> dict[str, Any]:

    expenses = claim.get(
        "expenses",
        {}
    )

    sum_insured = claim.get(
        "sum_insured_inr"
    )

    if not sum_insured:
        return {}

    # ------------------------------------------------------------
    # Hospital days
    # ------------------------------------------------------------

    hospitalization_hours = claim.get(
        "hospitalization_hours"
    )

    if hospitalization_hours:

        hospital_days = max(
            1,
            int(
                (
                    float(
                        hospitalization_hours
                    ) + 23
                )
                // 24
            )
        )

    else:

        hospital_days = 1

    calculations = []

    total_deduction = 0.0

    # ============================================================
    # ROOM
    # ============================================================

    room_claimed = expenses.get(
        "room"
    )

    if room_claimed is not None:

        room_limit_per_day = (
            0.01 * sum_insured
        )

        room_allowed = (
            room_limit_per_day
            * hospital_days
        )

        room_allowed = min(
            float(room_claimed),
            room_allowed
        )

        room_deduction = (
            float(room_claimed)
            - room_allowed
        )

        total_deduction += (
            room_deduction
        )

        calculations.append({
            "category": "room",
            "claimed_inr":
                float(room_claimed),
            "allowed_inr":
                round(
                    room_allowed,
                    2
                ),
            "deduction_inr":
                round(
                    room_deduction,
                    2
                ),
            "calculation": (
                f"1.0% of Basic Sum Insured "
                f"per day × {hospital_days} days"
            ),
            "policy_page": 7,
            "policy_chunk":
                "policy_p07_unspecified_021",
        })

    # ============================================================
    # DOCTOR FEES
    # ============================================================

    doctor_claimed = expenses.get(
        "doctor_fees"
    )

    if doctor_claimed is not None:

        doctor_limit = (
            0.25 * sum_insured
        )

        doctor_allowed = min(
            float(doctor_claimed),
            doctor_limit
        )

        doctor_deduction = (
            float(doctor_claimed)
            - doctor_allowed
        )

        total_deduction += (
            doctor_deduction
        )

        calculations.append({
            "category":
                "doctor_fees",
            "claimed_inr":
                float(doctor_claimed),
            "allowed_inr":
                round(
                    doctor_allowed,
                    2
                ),
            "deduction_inr":
                round(
                    doctor_deduction,
                    2
                ),
            "calculation":
                "25% of Sum Assured",
            "policy_page": 7,
            "policy_chunk":
                "policy_p07_unspecified_021",
        })

    # ============================================================
    # MEDICINES / DIAGNOSTICS
    # ============================================================

    medical_claimed = expenses.get(
        "medicines_diagnostics"
    )

    if medical_claimed is not None:

        medical_limit = (
            0.40 * sum_insured
        )

        medical_allowed = min(
            float(medical_claimed),
            medical_limit
        )

        medical_deduction = (
            float(medical_claimed)
            - medical_allowed
        )

        total_deduction += (
            medical_deduction
        )

        calculations.append({
            "category":
                "medicines_diagnostics",
            "claimed_inr":
                float(medical_claimed),
            "allowed_inr":
                round(
                    medical_allowed,
                    2
                ),
            "deduction_inr":
                round(
                    medical_deduction,
                    2
                ),
            "calculation":
                "40% of Sum Insured",
            "policy_page": 7,
            "policy_chunk":
                "policy_p07_unspecified_022",
        })

    # ============================================================
    # AMBULANCE
    # ============================================================

    ambulance_claimed = expenses.get(
        "ambulance"
    )

    if ambulance_claimed is not None:

        ambulance_limit = min(
            0.01 * sum_insured,
            1000
        )

        ambulance_allowed = min(
            float(ambulance_claimed),
            ambulance_limit
        )

        ambulance_deduction = (
            float(ambulance_claimed)
            - ambulance_allowed
        )

        total_deduction += (
            ambulance_deduction
        )

        calculations.append({
            "category":
                "ambulance",
            "claimed_inr":
                float(ambulance_claimed),
            "allowed_inr":
                round(
                    ambulance_allowed,
                    2
                ),
            "deduction_inr":
                round(
                    ambulance_deduction,
                    2
                ),
            "calculation": (
                "1% of Basic Sum Insured or "
                "₹1,000, whichever is less"
            ),
            "policy_page": 8,
            "policy_chunk":
                "policy_p08_what_we_exclude_027",
        })

    # ============================================================
    # PRE / POST HOSPITALIZATION
    # ============================================================

    pre_hospitalization = expenses.get(
        "pre_hospitalization"
    )

    post_hospitalization = expenses.get(
        "post_hospitalization"
    )

    pre_post_status = {
        "pre_claimed_inr":
            pre_hospitalization,

        "post_claimed_inr":
            post_hospitalization,

        "status":
            "REQUIRES_VERIFICATION",

        "reason": (
            "Pre- and post-hospitalization expenses "
            "require verification of the applicable "
            "time windows and policy conditions before "
            "being included in the payable amount."
        ),
    }

    if (
        pre_hospitalization is not None
        or post_hospitalization is not None
    ):

        calculations.append({
            "category":
                "pre_post_hospitalization",

            "claimed_inr": (
                float(
                    pre_hospitalization or 0
                )
                +
                float(
                    post_hospitalization or 0
                )
            ),

            "allowed_inr":
                None,

            "deduction_inr":
                None,

            "calculation": (
                "Pre-hospitalization up to 30 days "
                "and post-hospitalization up to 60 days, "
                "subject to policy conditions"
            ),

            "policy_page": 8,

            "policy_chunk":
                "policy_p08_what_we_exclude_027",
        })

    return {
        "calculations":
            calculations,

        "total_deduction_inr":
            round(
                total_deduction,
                2
            ),

        "pre_post_hospitalization":
            pre_post_status,
        "citations": [
            {
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 7,
                "section": "Unspecified",
                "chunk_id": "policy_p07_unspecified_021",
            },
            {
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 7,
                "section": "Unspecified",
                "chunk_id": "policy_p07_unspecified_022",
            },
            {
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 8,
                "section": "What We Exclude",
                "chunk_id": "policy_p08_what_we_exclude_027",
            },
        ],
    }
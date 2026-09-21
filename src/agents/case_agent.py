from __future__ import annotations

from datetime import date
from typing import Any


def _parse_date(value: Any) -> date | None:
    if not value:
        return None

    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _days_between(start_value: Any, end_value: Any) -> int | None:
    start = _parse_date(start_value)
    end = _parse_date(end_value)

    if start is None or end is None:
        return None

    return (end - start).days


def _add_dimension(dimensions: list[str], dimension: str) -> None:
    if dimension not in dimensions:
        dimensions.append(dimension)


def _build_decision_dimensions(claim: dict[str, Any]) -> list[str]:
    """
    Identify the policy questions that must be investigated.

    Supports both:
    1. Nested claim schema used by the public cases:
       treatment.type, treatment.diagnosis, treatment.procedure,
       treatment.admission_hours, expenses_inr
    2. Flat claim schema used by candidate cases:
       hospitalization_type, condition, procedure,
       hospitalization_hours, expenses

    Dimensions are derived only from explicit claim facts.
    The Case Analysis Agent does not make the final claim decision.
    """

    dimensions: list[str] = []

    # ---------------------------------------------------------
    # Read both supported claim schemas
    # ---------------------------------------------------------

    treatment = claim.get("treatment") or {}
    evidence_context = claim.get("evidence_context") or {}

    # Nested public-case fields
    diagnosis = str(
        treatment.get("diagnosis", "")
    ).strip().lower()

    procedure = str(
        treatment.get("procedure", "")
    ).strip().lower()

    treatment_type = str(
        treatment.get("type", "")
    ).strip().lower()

    admission_hours = treatment.get("admission_hours")

    # Flat candidate-case fields
    flat_condition = str(
        claim.get("condition", "")
    ).strip().lower()

    flat_procedure = str(
        claim.get("procedure", "")
    ).strip().lower()

    flat_treatment_type = str(
        claim.get("hospitalization_type", "")
    ).strip().lower()

    flat_admission_hours = claim.get("hospitalization_hours")

    # Combine equivalent fields from both schemas.
    effective_diagnosis = diagnosis or flat_condition
    effective_procedure = procedure or flat_procedure
    effective_treatment_type = treatment_type or flat_treatment_type

    if admission_hours is None:
        admission_hours = flat_admission_hours

    # Expenses:
    # Public cases use expenses_inr.
    # Candidate cases use expenses.
    expenses = (
        claim.get("expenses_inr")
        or claim.get("expenses")
        or {}
    )

    # ---------------------------------------------------------
    # 1. Waiting period
    # ---------------------------------------------------------

    _add_dimension(dimensions, "waiting_periods")

    # ---------------------------------------------------------
    # 2. Portability
    # ---------------------------------------------------------

    prior_years = claim.get("prior_insurer_continuous_years")

    if prior_years is not None:
        try:
            if float(prior_years) > 0:
                _add_dimension(dimensions, "portability")
        except (TypeError, ValueError):
            _add_dimension(dimensions, "portability")

    portability_keys = (
        "previous_policy_continuity",
        "previous_claim_history_received",
        "previous_insurer",
        "previous_sum_insured_inr",
    )

    if any(
        key in claim or key in evidence_context
        for key in portability_keys
    ):
        _add_dimension(dimensions, "portability")

    # ---------------------------------------------------------
    # 3. Hospitalization eligibility
    # ---------------------------------------------------------

    hospitalization_types = {
        "inpatient",
        "hospitalization",
        "hospitalisation",
        "daycare",
        "day_care",
        "day care",
        "domiciliary",
    }

    if effective_treatment_type in hospitalization_types:
        _add_dimension(
            dimensions,
            "hospitalization_eligibility",
        )
        _add_dimension(
            dimensions,
            "hospital_definition",
        )

    # Also investigate hospitalization eligibility when the
    # claim explicitly provides hospitalization hours.
    if admission_hours is not None:
        _add_dimension(
            dimensions,
            "hospitalization_eligibility",
        )
        _add_dimension(
            dimensions,
            "hospital_definition",
        )

    # ---------------------------------------------------------
    # 4. Day-care treatment
    # ---------------------------------------------------------

    if effective_treatment_type in {
        "daycare",
        "day_care",
        "day care",
    }:
        _add_dimension(dimensions, "day_care")

    elif admission_hours is not None:
        try:
            if float(admission_hours) < 24:
                _add_dimension(dimensions, "day_care")
        except (TypeError, ValueError):
            pass

    # ---------------------------------------------------------
    # 5. Domiciliary treatment
    # ---------------------------------------------------------

    if effective_treatment_type == "domiciliary":
        _add_dimension(
            dimensions,
            "domiciliary_treatment",
        )
        _add_dimension(
            dimensions,
            "domiciliary",
        )

    if evidence_context.get(
        "hospital_room_unavailable"
    ) is not None:
        _add_dimension(
            dimensions,
            "domiciliary_treatment",
        )

    if evidence_context.get(
        "patient_cannot_be_moved"
    ) is not None:
        _add_dimension(
            dimensions,
            "domiciliary_treatment",
        )

    # ---------------------------------------------------------
    # 6. Pre-existing disease
    # ---------------------------------------------------------

    pre_existing = treatment.get("pre_existing")

    if pre_existing is True:
        _add_dimension(
            dimensions,
            "pre_existing_disease",
        )

    if "pre-existing" in effective_diagnosis:
        _add_dimension(
            dimensions,
            "pre_existing_disease",
        )

    if "pre existing" in effective_diagnosis:
        _add_dimension(
            dimensions,
            "pre_existing_disease",
        )

    if "pre-existing" in effective_procedure:
        _add_dimension(
            dimensions,
            "pre_existing_disease",
        )

    if "pre existing" in effective_procedure:
        _add_dimension(
            dimensions,
            "pre_existing_disease",
        )

    # ---------------------------------------------------------
    # 7. Cosmetic / aesthetic exclusion
    # ---------------------------------------------------------

    cosmetic_terms = (
        "cosmetic",
        "aesthetic",
        "plastic surgery",
        "appearance",
        "beautification",
    )

    if any(
        term in effective_diagnosis
        or term in effective_procedure
        for term in cosmetic_terms
    ):
        _add_dimension(
            dimensions,
            "cosmetic_exclusion",
        )

    # ---------------------------------------------------------
    # 8. Experimental / unproven treatment exclusion
    # ---------------------------------------------------------

    experimental_flag = treatment.get("experimental")

    # Also support flat candidate schema if it contains
    # an explicit experimental field.
    if "experimental" in claim:
        experimental_flag = claim.get("experimental")

    experimental_terms = (
        "experimental",
        "unproven",
        "investigational",
        "gene therapy",
        "experimental therapy",
    )

    if experimental_flag is True:
        _add_dimension(
            dimensions,
            "experimental_treatment",
        )

    if any(
        term in effective_diagnosis
        or term in effective_procedure
        for term in experimental_terms
    ):
        _add_dimension(
            dimensions,
            "experimental_treatment",
        )

    # ---------------------------------------------------------
    # 9. Expense limits
    # ---------------------------------------------------------

    expense_dimension_map = {
        "room": "room_limit",
        "doctor_fees": "doctor_fee_limit",
        "medicines_diagnostics": "medical_expense_limit",
        "ambulance": "ambulance_limit",
        "pre_hospitalization": "pre_post_hospitalization",
        "post_hospitalization": "pre_post_hospitalization",
    }

    for field, dimension in expense_dimension_map.items():
        if field in expenses and expenses[field] is not None:
            _add_dimension(
                dimensions,
                "expense_limits",
            )
            _add_dimension(
                dimensions,
                dimension,
            )

    # ---------------------------------------------------------
    # 10. Pre/post hospitalization
    # ---------------------------------------------------------

    if (
        expenses.get("pre_hospitalization") is not None
        or expenses.get("post_hospitalization") is not None
    ):
        _add_dimension(
            dimensions,
            "pre_post_hospitalization",
        )

    # ---------------------------------------------------------
    # 11. Hospital evidence requirements
    # ---------------------------------------------------------

    if "hospital_registered" in evidence_context:
        _add_dimension(
            dimensions,
            "hospital_definition",
        )

    if "hospital_minimum_criteria_documented" in evidence_context:
        _add_dimension(
            dimensions,
            "hospital_definition",
        )

    # ---------------------------------------------------------
    # 12. Medical necessity
    # ---------------------------------------------------------

    if "medical_necessity_confirmed" in evidence_context:
        _add_dimension(
            dimensions,
            "medical_necessity",
        )

    return dimensions

def _identify_missing_evidence(
    claim: dict[str, Any],
    dimensions: list[str],
) -> list[str]:
    """
    Identify only explicitly missing evidence required by the claim.

    Do NOT mark evidence as missing merely because an optional field
    is absent from the synthetic case.
    """

    missing: list[str] = []

    treatment = claim.get("treatment") or {}
    evidence_context = claim.get("evidence_context") or {}

    # Hospital eligibility evidence.
    if "hospital_definition" in dimensions:

        if (
            "hospital_registered" in evidence_context
            and evidence_context.get("hospital_registered") is None
        ):
            missing.append(
                "Evidence establishing hospital registration or eligibility."
            )

        if (
            evidence_context.get(
                "hospital_minimum_criteria_documented"
            ) is False
        ):
            missing.append(
                "Evidence establishing the hospital's minimum policy criteria."
            )

    # Medical necessity only when explicitly supplied as an unknown.
    if (
        "medical_necessity" in dimensions
        and "medical_necessity_confirmed" in evidence_context
        and evidence_context.get("medical_necessity_confirmed") is None
    ):
        missing.append(
            "Evidence establishing medical necessity for the treatment."
        )

    # Portability evidence.
    if "portability" in dimensions:

        prior_years = claim.get("prior_insurer_continuous_years")

        if prior_years is not None:
            try:
                if float(prior_years) > 0:
                    if (
                        evidence_context.get(
                            "previous_policy_continuity"
                        ) is False
                    ):
                        missing.append(
                            "Evidence establishing qualifying continuous "
                            "coverage under the previous insurer."
                        )

                    if (
                        evidence_context.get(
                            "previous_claim_history_received"
                        ) is False
                    ):
                        missing.append(
                            "Evidence that the previous insurer's database "
                            "and claim history were received."
                        )
            except (TypeError, ValueError):
                pass

    # Pre/post condition evidence.
 

    # Explicit experimental uncertainty.
    if "experimental_treatment" in dimensions:
        if treatment.get("experimental") is None:
            missing.append(
                "Evidence establishing whether the treatment is "
                "experimental or unproven."
            )

    return missing


def _build_investigation_plan(
    dimensions: list[str],
) -> list[str]:
    plan_map = {
        "waiting_periods":
            "Verify applicable initial and condition-specific waiting periods.",

        "portability":
            "Verify continuity credit and portability-related waiting-period reductions.",

        "hospitalization_eligibility":
            "Verify whether the treatment satisfies the policy hospitalization definition.",

        "hospital_definition":
            "Verify whether the facility satisfies the policy hospital definition.",

        "day_care":
            "Verify whether the treatment qualifies as a covered day-care procedure.",

        "domiciliary_treatment":
            "Verify the policy conditions for domiciliary hospitalization.",

        "domiciliary":
            "Verify applicable domiciliary hospitalization limits and conditions.",

        "pre_existing_disease":
            "Verify the policy waiting period and conditions applicable to pre-existing disease.",

        "cosmetic_exclusion":
            "Verify whether cosmetic or aesthetic treatment is excluded by the policy.",

        "experimental_treatment":
            "Verify whether experimental or unproven treatment is excluded by the policy.",

        "expense_limits":
            "Verify category-specific expense limits and calculate applicable deductions.",

        "room_limit":
            "Verify the policy room/board/nursing expense limit.",

        "doctor_fee_limit":
            "Verify the policy medical practitioner and consultant fee limit.",

        "medical_expense_limit":
            "Verify applicable limits for medicines, diagnostics and related medical expenses.",

        "ambulance_limit":
            "Verify the policy ambulance expense limit.",

        "pre_post_hospitalization":
            "Verify applicable pre- and post-hospitalization expense conditions and time windows.",

        "medical_necessity":
            "Verify evidence establishing medical necessity.",
    }

    plan: list[str] = []

    for dimension in dimensions:
        instruction = plan_map.get(dimension)

        if instruction and instruction not in plan:
            plan.append(instruction)

    return plan


def case_analysis_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Case Analysis Agent.

    Converts raw claim facts into explicit policy investigation dimensions,
    missing evidence requirements, and an investigation plan.

    The agent does not make the final claim decision.
    """

    claim = state.get("claim_facts") or {}

    dimensions = _build_decision_dimensions(claim)

    missing_evidence = _identify_missing_evidence(
        claim,
        dimensions,
    )

    investigation_plan = _build_investigation_plan(
        dimensions,
    )

    trace = list(state.get("trace") or [])

    trace.append(
        {
            "agent": "CaseAnalysisAgent",
            "action": (
                "Analyzed claim facts and identified the policy dimensions "
                "requiring evidence retrieval and evaluation."
            ),
            "decision_dimensions": dimensions,
            "missing_evidence_count": len(missing_evidence),
        }
    )

    return {
        "decision_dimensions": dimensions,
        "missing_evidence": missing_evidence,
        "investigation_plan": investigation_plan,
        "trace": trace,
    }
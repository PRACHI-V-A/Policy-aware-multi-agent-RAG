from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, START, END

from src.agents.state import ClaimState
from src.agents.case_agent import case_analysis_agent
from src.agents.evidence_agent import evidence_agent
from src.agents.coverage_agent import coverage_analysis_agent
from src.agents.decision_agent import decision_agent
from src.agents.validation_agent import validation_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_workflow():
    graph = StateGraph(ClaimState)

    graph.add_node(
        "case_analysis",
        case_analysis_agent,
    )

    graph.add_node(
        "policy_evidence",
        evidence_agent,
    )

    graph.add_node(
        "coverage_analysis",
        coverage_analysis_agent,
    )

    graph.add_node(
        "decision",
        decision_agent,
    )

    graph.add_node(
        "validation",
        validation_agent,
    )

    graph.add_edge(
        START,
        "case_analysis",
    )

    graph.add_edge(
        "case_analysis",
        "policy_evidence",
    )

    graph.add_edge(
        "policy_evidence",
        "coverage_analysis",
    )

    graph.add_edge(
        "coverage_analysis",
        "decision",
    )

    graph.add_edge(
        "decision",
        "validation",
    )

    graph.add_edge(
        "validation",
        END,
    )

    return graph.compile()


def create_initial_state(
    claim: dict[str, Any],
) -> ClaimState:

    return {
        "case_id": claim["case_id"],
        "claim_facts": claim,

        "decision_dimensions": [],
        "investigation_plan": [],
        "missing_evidence": [],

        "retrieved_evidence": [],

        "coverage_findings": [],
        "exclusion_findings": [],
        "limit_findings": [],

        "decision": "NEEDS_REVIEW",
        "confidence": 0.0,
        "key_findings": [],

        "expense_calculation": {},

        "citations": [],
        "validation_status": "PENDING",
        "unsupported_claims": [],

        "trace": [],
    }


def analyze_claim(
    claim: dict[str, Any],
    workflow=None,
) -> dict[str, Any]:

    if workflow is None:
        workflow = build_workflow()

    initial_state = create_initial_state(
        claim
    )

    result = workflow.invoke(
        initial_state
    )

    return result


def load_cases(
    cases_path: Path,
) -> list[dict[str, Any]]:

    if not cases_path.exists():
        raise FileNotFoundError(
            f"Cases file not found: {cases_path}"
        )

    with cases_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        cases = json.load(f)

    if not isinstance(
        cases,
        list,
    ):
        raise ValueError(
            "Cases JSON must contain a list."
        )

    return cases


def create_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    decision_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}

    for result in results:

        decision = result.get(
            "decision",
            "UNKNOWN",
        )

        validation = result.get(
            "validation_status",
            "UNKNOWN",
        )

        decision_counts[decision] = (
            decision_counts.get(
                decision,
                0,
            )
            + 1
        )

        validation_counts[validation] = (
            validation_counts.get(
                validation,
                0,
            )
            + 1
        )

    return {
        "total_cases": len(results),
        "decision_counts": decision_counts,
        "validation_counts": validation_counts,
        "results": [
            {
                "case_id": result.get(
                    "case_id"
                ),
                "decision": result.get(
                    "decision"
                ),
                "confidence": result.get(
                    "confidence"
                ),
                "validation_status": result.get(
                    "validation_status"
                ),
            }
            for result in results
        ],
    }


def analyze_all_cases(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    print(
        f"\nRunning {len(cases)} claim cases..."
    )

    workflow = build_workflow()

    results = []

    for index, claim in enumerate(
        cases,
        start=1,
    ):

        case_id = claim.get(
            "case_id",
            f"CASE-{index:03d}",
        )

        print(
            f"\n[{index}/{len(cases)}] "
            f"Analyzing {case_id}..."
        )

        try:

            result = analyze_claim(
                claim,
                workflow=workflow,
            )

            print(
                f"       Decision: "
                f"{result.get('decision')}"
            )

            print(
                f"       Confidence: "
                f"{result.get('confidence')}"
            )

            print(
                f"       Validation: "
                f"{result.get('validation_status')}"
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"       ERROR: {exc}"
            )

            results.append(
                {
                    "case_id": case_id,
                    "claim_facts": claim,
                    "decision": "NEEDS_REVIEW",
                    "confidence": 0.0,
                    "validation_status": "FAIL",
                    "missing_evidence": [
                        "Workflow execution failed."
                    ],
                    "unsupported_claims": [],
                    "retrieved_evidence": [],
                    "citations": [],
                    "trace": [
                        {
                            "agent": "Workflow",
                            "action": "Execution failed.",
                            "error": str(exc),
                        }
                    ],
                }
            )

    return results


def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    result_name: str,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_dir
        / f"{result_name}_results.json"
    )

    summary_path = (
        output_dir
        / f"{result_name}_summary.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = create_summary(
        results
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved complete results to:"
    )

    print(
        f"  {result_path}"
    )

    print(
        "\nSaved summary to:"
    )

    print(
        f"  {summary_path}"
    )


def print_summary(
    results: list[dict[str, Any]],
) -> None:

    summary = create_summary(
        results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLAIM EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total cases: "
        f"{summary['total_cases']}"
    )

    print(
        "\nDecision counts:"
    )

    for decision, count in summary[
        "decision_counts"
    ].items():

        print(
            f"  {decision}: {count}"
        )

    print(
        "\nValidation counts:"
    )

    for status, count in summary[
        "validation_counts"
    ].items():

        print(
            f"  {status}: {count}"
        )

    print(
        "\nCase results:"
    )

    print(
        "-" * 70
    )

    for result in summary[
        "results"
    ]:

        print(
            f"{result['case_id']:<10} | "
            f"{result['decision']:<25} | "
            f"confidence={result['confidence']} | "
            f"validation={result['validation_status']}"
        )

    print(
        "=" * 70
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the Policy-Aware Multi-Agent "
            "RAG claim decision workflow."
        )
    )

    parser.add_argument(
        "--cases",
        type=str,
        default=(
            "candidate_data/"
            "public_test_cases.json"
        ),
        help="Path to cases JSON file.",
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Run only one case by case_id.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "evaluation/results"
        ),
        help="Directory for result files.",
    )

    parser.add_argument(
        "--result-name",
        type=str,
        default=None,
        help=(
            "Base name for result files. "
            "Defaults to public_case or candidate_case "
            "based on input filename."
        ),
    )

    args = parser.parse_args()

    cases_path = (
        PROJECT_ROOT
        / args.cases
    )

    cases = load_cases(
        cases_path
    )

    if args.case_id:

        cases = [
            case
            for case in cases
            if case.get("case_id")
            == args.case_id
        ]

        if not cases:

            raise ValueError(
                f"Case ID not found: "
                f"{args.case_id}"
            )

    results = analyze_all_cases(
        cases
    )

    print_summary(
        results
    )

    output_dir = (
        PROJECT_ROOT
        / args.output_dir
    )

    if args.result_name:

        result_name = args.result_name

    else:
        case_filename = Path(args.cases).name.lower()

        if "candidate" in case_filename:
            result_name = "candidate_case"
        else:
            result_name = "public_case" 

    save_results(
        results,
        output_dir,
        result_name,
    )


if __name__ == "__main__":
    main()
"""Evaluation pipeline for the Policy-Aware Multi-Agent RAG Claim Decision Engine."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"
PUBLIC_RESULTS_FILE = RESULTS_DIR / "public_case_results.json"
CANDIDATE_RESULTS_FILE = RESULTS_DIR / "candidate_case_results.json"
JSON_OUTPUT = RESULTS_DIR / "evaluation_report.json"
MD_OUTPUT = RESULTS_DIR / "evaluation_report.md"

VALID_DECISIONS = {
    "ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE", "NEEDS_REVIEW",
}

# Gold labels for the 12 supplied public evaluation cases.
# These are kept here because the supplied public_test_cases.json contains
# claim facts/tasks but does not contain an expected_decision field.
PUBLIC_EXPECTED_DECISIONS = {
    "PUB-001": "ADMISSIBLE_WITH_LIMITS",
    "PUB-002": "NOT_ADMISSIBLE",
    "PUB-003": "NOT_ADMISSIBLE",
    "PUB-004": "ADMISSIBLE_WITH_LIMITS",
    "PUB-005": "ADMISSIBLE_WITH_LIMITS",
    "PUB-006": "NEEDS_REVIEW",
    "PUB-007": "ADMISSIBLE_WITH_LIMITS",
    "PUB-008": "NOT_ADMISSIBLE",
    "PUB-009": "ADMISSIBLE_WITH_LIMITS",
    "PUB-010": "ADMISSIBLE_WITH_LIMITS",
    "PUB-011": "NEEDS_REVIEW",
    "PUB-012": "NOT_ADMISSIBLE",
}

DEVELOPMENT_FAILURE_HISTORY = [
    {
        "failure": "Incomplete candidate decision dimensions",
        "symptom": "Several candidate cases initially received only waiting_periods, hospital_definition, and medical_necessity dimensions, so relevant exclusions and limits were not investigated.",
        "root_cause": "Case Analysis Agent generated an incomplete investigation plan for some case types.",
        "improvement": "Expanded case-dimension mapping for cosmetic, experimental, day-care, expense-limit, pre/post, and related policy dimensions.",
    },
    {
        "failure": "False missing-evidence condition for pre/post expenses",
        "symptom": "Eligible cases with supplied pre/post timing and condition facts were temporarily treated as requiring review.",
        "root_cause": "Decision logic required additional evidence even when the synthetic claim facts already supplied the relevant condition and timing.",
        "improvement": "Expense calculation now uses supplied claim facts for applicable pre/post windows while citing the policy rule.",
    },
    {
        "failure": "Citation validation mismatch",
        "symptom": "A decision could temporarily contain a citation whose chunk ID was not present in retrieved evidence.",
        "root_cause": "Citation construction and retrieved-evidence tracking were not initially guaranteed to use the same chunk IDs.",
        "improvement": "Decision citations are derived from retrieved evidence and validation checks citation traceability.",
    },
]


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "cases", "case_results"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"No result list found in {path}")


def case_id(r: dict[str, Any]) -> str:
    return str(r.get("case_id") or r.get("claim_facts", {}).get("case_id") or "UNKNOWN")


def actual_decision(r: dict[str, Any]) -> str | None:
    value = r.get("decision")
    if isinstance(value, dict):
        value = value.get("status") or value.get("decision")
    return str(value) if value is not None else None


def expected_decision(r: dict[str, Any], dataset: str) -> str | None:
    for key in ("expected_decision", "expected_status"):
        if r.get(key):
            return str(r[key])
    facts = r.get("claim_facts")
    if isinstance(facts, dict):
        for key in ("expected_decision", "expected_status"):
            if facts.get(key):
                return str(facts[key])
    if dataset == "public":
        return PUBLIC_EXPECTED_DECISIONS.get(case_id(r))
    return None


def validation(r: dict[str, Any]) -> str | None:
    value = r.get("validation_status")
    if isinstance(value, dict):
        value = value.get("status") or value.get("validation_status")
    return str(value).upper() if value is not None else None


def confidence(r: dict[str, Any]) -> float | None:
    try:
        return float(r["confidence"]) if r.get("confidence") is not None else None
    except (TypeError, ValueError):
        return None


def evidence(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in r.get("retrieved_evidence", []) if isinstance(x, dict)]


def citations(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in r.get("citations", []) if isinstance(x, dict)]


def missing(r: dict[str, Any]) -> list[Any]:
    return r.get("missing_evidence", []) if isinstance(r.get("missing_evidence"), list) else []


def unsupported(r: dict[str, Any]) -> list[Any]:
    return r.get("unsupported_claims", []) if isinstance(r.get("unsupported_claims"), list) else []


def decision_metrics(results: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    distribution: dict[str, int] = {}
    correct = incorrect = evaluated = invalid = 0
    mismatches = []

    for r in results:
        actual = actual_decision(r)
        expected = expected_decision(r, dataset)
        if actual in VALID_DECISIONS:
            distribution[actual] = distribution.get(actual, 0) + 1
        else:
            invalid += 1
        if expected is None:
            continue
        evaluated += 1
        if actual == expected:
            correct += 1
        else:
            incorrect += 1
            mismatches.append({"case_id": case_id(r), "expected": expected, "actual": actual})

    return {
        "total_cases": len(results),
        "expected_labels_available": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / evaluated if evaluated else None,
        "invalid_decisions": invalid,
        "decision_distribution": distribution,
        "mismatches": mismatches,
    }


def validation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in results:
        status = validation(r) or "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
    total = len(results)
    passed = counts.get("PASS", 0)
    return {
        "total_cases": total,
        "passed": passed,
        "failed": counts.get("FAIL", 0),
        "unknown": counts.get("UNKNOWN", 0),
        "pass_rate": passed / total if total else None,
        "status_distribution": counts,
    }


def abstention_metrics(results: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    actual_review = [case_id(r) for r in results if actual_decision(r) == "NEEDS_REVIEW"]
    correct_review = [
        case_id(r) for r in results
        if actual_decision(r) == "NEEDS_REVIEW" and expected_decision(r, dataset) == "NEEDS_REVIEW"
    ]
    total = len(results)
    return {
        "count": len(actual_review),
        "rate": len(actual_review) / total if total else None,
        "case_ids": actual_review,
        "correctly_abstained_case_ids": correct_review,
    }


def retrieval_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(evidence(r)) for r in results]
    cases_with = sum(1 for x in counts if x > 0)
    return {
        "cases_with_retrieved_evidence": cases_with,
        "coverage_rate": cases_with / len(results) if results else None,
        "average_retrieved_chunks": statistics.mean(counts) if counts else 0,
        "maximum_retrieved_chunks": max(counts) if counts else 0,
        "formal_recall_at_k": {
            "available": False,
            "reason": "No manually annotated gold relevant policy chunks are supplied.",
        },
    }


def citation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = with_chunk = grounded = complete = 0
    cases_with = fully_grounded = 0

    for r in results:
        ev_ids = {str(x.get("chunk_id")) for x in evidence(r) if x.get("chunk_id")}
        cs = citations(r)
        if cs:
            cases_with += 1
        all_grounded = True
        for c in cs:
            total += 1
            cid = c.get("chunk_id")
            if c.get("source") and c.get("page") is not None and c.get("section") and cid:
                complete += 1
            if cid:
                with_chunk += 1
                if str(cid) in ev_ids:
                    grounded += 1
                else:
                    all_grounded = False
        if cs and all_grounded:
            fully_grounded += 1

    return {
        "total_citations": total,
        "citation_completeness_rate": complete / total if total else None,
        "citation_hit_rate": grounded / with_chunk if with_chunk else None,
        "cases_with_citations": cases_with,
        "fully_grounded_cases": fully_grounded,
        "case_grounding_rate": fully_grounded / cases_with if cases_with else None,
        "semantic_citation_correctness": "MANUAL_REVIEW_REQUIRED",
    }


def case_evaluations(results: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    output = []
    for r in results:
        ev = evidence(r)
        cs = citations(r)
        ev_ids = {str(x.get("chunk_id")) for x in ev if x.get("chunk_id")}
        cited_ids = {str(x.get("chunk_id")) for x in cs if x.get("chunk_id")}
        exp = expected_decision(r, dataset)
        act = actual_decision(r)
        output.append({
            "case_id": case_id(r),
            "expected_decision": exp,
            "actual_decision": act,
            "decision_correct": act == exp if exp is not None else None,
            "validation_status": validation(r),
            "confidence": confidence(r),
            "retrieved_evidence_count": len(ev),
            "citation_count": len(cs),
            "grounded_citation_count": len(cited_ids & ev_ids),
            "missing_evidence": missing(r),
            "unsupported_claims": unsupported(r),
        })
    return output


def failure_analysis(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision_mismatches": [
            {"case_id": c["case_id"], "expected": c["expected_decision"], "actual": c["actual_decision"]}
            for c in cases if c["decision_correct"] is False
        ],
        "validation_failures": [c["case_id"] for c in cases if c["validation_status"] == "FAIL"],
        "review_cases": [c["case_id"] for c in cases if c["actual_decision"] == "NEEDS_REVIEW"],
    }


def dataset_report(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    cases = case_evaluations(results, name)
    return {
        "dataset": name,
        "case_count": len(results),
        "decision_metrics": decision_metrics(results, name),
        "validation_metrics": validation_metrics(results),
        "abstention_metrics": abstention_metrics(results, name),
        "retrieval_metrics": retrieval_metrics(results),
        "citation_metrics": citation_metrics(results),
        "case_evaluations": cases,
        "failure_analysis": failure_analysis(cases),
    }


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_dataset(report: dict[str, Any]) -> list[str]:
    d = report["decision_metrics"]
    v = report["validation_metrics"]
    a = report["abstention_metrics"]
    r = report["retrieval_metrics"]
    c = report["citation_metrics"]
    lines = [
        f"## {report['dataset'].title()}", "",
        f"- Cases: **{report['case_count']}**",
        f"- Decision accuracy: **{pct(d['accuracy'])}**",
        f"- Correct decisions: **{d['correct']}/{d['expected_labels_available']}**",
        f"- Validation pass rate: **{pct(v['pass_rate'])}**",
        f"- NEEDS_REVIEW: **{a['count']}** ({pct(a['rate'])})",
        f"- Retrieval evidence coverage: **{pct(r['coverage_rate'])}**",
        f"- Average retrieved chunks: **{r['average_retrieved_chunks']:.2f}**",
        f"- Citation hit rate: **{pct(c['citation_hit_rate'])}**",
        f"- Citation completeness: **{pct(c['citation_completeness_rate'])}**",
        "", "### Decision distribution", "",
    ]
    for k, val in d["decision_distribution"].items():
        lines.append(f"- `{k}`: {val}")
    lines += ["", "### Case-level results", "", "| Case | Expected | Actual | Correct | Validation | Evidence | Citations |", "|---|---|---|---|---|---:|---:|"]
    for x in report["case_evaluations"]:
        correct = "YES" if x["decision_correct"] is True else "NO" if x["decision_correct"] is False else "N/A"
        lines.append(f"| {x['case_id']} | {x['expected_decision'] or 'N/A'} | {x['actual_decision'] or 'N/A'} | {correct} | {x['validation_status'] or 'N/A'} | {x['retrieved_evidence_count']} | {x['citation_count']} |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    combined = report["combined_metrics"]
    lines = [
        "# Policy-Aware Multi-Agent RAG Evaluation Report", "",
        "## Executive Summary", "",
        f"- Total cases: **{combined['total_cases']}**",
        f"- Decision accuracy: **{pct(combined['decision_accuracy'])}**",
        f"- Validation pass rate: **{pct(combined['validation_pass_rate'])}**",
        f"- Total NEEDS_REVIEW cases: **{combined['total_abstentions']}**",
        f"- Abstention requirement (>=2): **{'PASS' if combined['abstention_requirement_met'] else 'FAIL'}**",
        "",
        *render_dataset(report["public"]), "",
        *render_dataset(report["candidate"]), "",
        "## Abstention Requirement", "",
        f"- Public: **{report['public']['abstention_metrics']['count']}**",
        f"- Candidate: **{report['candidate']['abstention_metrics']['count']}**",
        f"- Combined: **{combined['total_abstentions']}**",
        f"- Required minimum: **2**",
        f"- Status: **{'PASS' if combined['abstention_requirement_met'] else 'FAIL'}**",
        "",
        "## Citation Quality", "",
        "Citation hit rate checks whether a cited chunk ID was actually present in retrieved evidence. It is a traceability metric, not semantic proof that the policy text supports the decision.",
        "",
        "Semantic citation correctness therefore remains a manual-review item.",
        "",
        "## Retrieval Quality", "",
        "The system uses dense and sparse retrieval with reranking. Evidence coverage and retrieved-chunk statistics are reported.",
        "",
        "Formal Recall@K is not reported because there is no manually annotated gold relevance set.",
        "",
        "## Current Failure / Review Analysis", "",
    ]
    for name in ("public", "candidate"):
        f = report[name]["failure_analysis"]
        lines.append(f"### {name.title()}")
        if f["decision_mismatches"]:
            for x in f["decision_mismatches"]:
                lines.append(f"- Decision mismatch: `{x['case_id']}` expected `{x['expected']}`, got `{x['actual']}`")
        else:
            lines.append("- No decision mismatches.")
        if f["validation_failures"]:
            lines.append(f"- Validation failures: {', '.join(f['validation_failures'])}")
        if f["review_cases"]:
            lines.append(f"- Review cases: {', '.join(f['review_cases'])}")
        lines.append("")
    lines += ["## Historical Development Failures and Improvements", ""]
    for i, x in enumerate(DEVELOPMENT_FAILURE_HISTORY, 1):
        lines += [f"### {i}. {x['failure']}", "", f"**Symptom:** {x['symptom']}", "", f"**Root cause:** {x['root_cause']}", "", f"**Improvement:** {x['improvement']}", ""]
    lines += [
        "## Reproducibility", "",
        "```bash", "python -m pytest -q", "python -m src.workflow", "python evaluation/evaluate.py", "```", "",
        "## Limitations", "",
        "1. Formal retrieval Recall@K requires a gold relevance annotation set.",
        "2. Citation hit rate is chunk-level grounding, not semantic citation correctness.",
        "3. Decision accuracy depends on the documented gold labels for the evaluation cases.",
        "4. Confidence is descriptive and is not treated as a calibrated probability.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    public = load_results(PUBLIC_RESULTS_FILE)
    candidate = load_results(CANDIDATE_RESULTS_FILE)
    public_report = dataset_report("public", public)
    candidate_report = dataset_report("candidate", candidate)

    total_cases = len(public) + len(candidate)
    evaluated = public_report["decision_metrics"]["expected_labels_available"] + candidate_report["decision_metrics"]["expected_labels_available"]
    correct = public_report["decision_metrics"]["correct"] + candidate_report["decision_metrics"]["correct"]
    total_validation = public_report["validation_metrics"]["total_cases"] + candidate_report["validation_metrics"]["total_cases"]
    validation_passed = public_report["validation_metrics"]["passed"] + candidate_report["validation_metrics"]["passed"]
    total_abstentions = public_report["abstention_metrics"]["count"] + candidate_report["abstention_metrics"]["count"]

    report = {
        "report_name": "Policy-Aware Multi-Agent RAG Claim Decision Engine",
        "evaluation_version": "2.1",
        "combined_metrics": {
            "total_cases": total_cases,
            "expected_labels_available": evaluated,
            "correct_decisions": correct,
            "decision_accuracy": correct / evaluated if evaluated else None,
            "validation_pass_rate": validation_passed / total_validation if total_validation else None,
            "total_abstentions": total_abstentions,
            "abstention_requirement": 2,
            "abstention_requirement_met": total_abstentions >= 2,
        },
        "public": public_report,
        "candidate": candidate_report,
        "public_gold_labels": PUBLIC_EXPECTED_DECISIONS,
        "development_failure_history": DEVELOPMENT_FAILURE_HISTORY,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(report), encoding="utf-8")

    print("=" * 70)
    print("Policy-Aware Multi-Agent RAG Evaluation")
    print("=" * 70)
    print(f"Total cases: {total_cases}")
    print(f"Decision accuracy: {pct(report['combined_metrics']['decision_accuracy'])}")
    print(f"Validation pass rate: {pct(report['combined_metrics']['validation_pass_rate'])}")
    print(f"Public decision accuracy: {pct(public_report['decision_metrics']['accuracy'])}")
    print(f"Candidate decision accuracy: {pct(candidate_report['decision_metrics']['accuracy'])}")
    print(f"Total NEEDS_REVIEW: {total_abstentions}")
    print(f"Abstention requirement: {'PASS' if total_abstentions >= 2 else 'FAIL'}")
    print(f"Public citation hit rate: {pct(public_report['citation_metrics']['citation_hit_rate'])}")
    print(f"Candidate citation hit rate: {pct(candidate_report['citation_metrics']['citation_hit_rate'])}")
    print(f"JSON report: {JSON_OUTPUT}")
    print(f"Markdown report: {MD_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()

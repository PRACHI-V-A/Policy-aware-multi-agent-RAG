from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever


QUERY_MAP = {

    "hospitalization_eligibility": [
        "Hospitalization means admission to Hospital for minimum 24 consecutive hours",
        "Hospitalization definition minimum 24 hours",
        "hospitalisation 24 hours specified procedures",
    ],

    "hospital_definition": [
        "definition of Hospital registration minimum criteria",
        "Hospital means institution registered minimum criteria",
        "qualified medical practitioner inpatient beds operation theatre records",
    ],

    "waiting_periods": [
        "30 days waiting period illness policy",
        "waiting period first 30 days policy",
        "one year waiting period specified diseases",
    ],

    "pre_existing_disease": [
        "pre-existing diseases 48 months continuous coverage",
        "pre existing disease waiting period",
        "48 months continuous coverage inception first policy",
    ],

    "portability": [
        "portability waiting period credit previous insurer",
        "continuous coverage Indian insurer waiting period reduction",
        "completed years coverage waiver waiting periods",
    ],

    "domiciliary_treatment": [
        "domiciliary hospitalization hospital room unavailable",
        "domiciliary treatment patient cannot be moved",
        "domiciliary hospitalization sub limit 20 percent",
    ],

    "expense_limits": [
        "hospitalization expenses sub limits room doctor medical expenses",
        "room boarding nursing expense 1 percent basic sum insured",
        "medical expenses 40 percent sum insured",
    ],

    "room_limit": [
        "normal room expenses 1.0 percent Basic Sum Insured",
        "room boarding nursing expense sub limits",
    ],

    "doctor_fee_limit": [
        "medical practitioner consultant fees 25 percent Sum Insured",
        "surgeon anesthetist consultant fees limit",
    ],

    "medical_expense_limit": [
        "medicines drugs diagnostic materials expenses 40 percent Sum Insured",
        "medical expenses limit 40 percent sum insured",
    ],

    "ambulance_limit": [
        "ambulance charges 1 percent Basic Sum Insured 1000",
        "ambulance admissible claim limit",
    ],

    "pre_post_hospitalization": [
        "Pre-Hospitalisation maximum 30 days",
        "Post Hospitalisation maximum 60 days",
        "pre hospitalization post hospitalization same condition",
    ],

    "cosmetic_exclusion": [
        "cosmetic treatment exclusion",
        "cosmetic aesthetic treatment excluded",
        "plastic surgery cosmetic aesthetic treatment exclusion",
    ],

    "experimental_treatment": [
        "Unproven Experimental Treatment established medical practice India",
        "experimental therapy unproven treatment",
        "treatment not based on established medical practice India",
    ],

    "domiciliary": [
        "domiciliary treatment hospital room unavailable",
        "domiciliary hospitalization patient cannot be moved",
        "domiciliary hospitalization 20 percent Basic Sum Insured",
    ],
}


def evidence_agent(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Retrieve policy evidence for every decision dimension.

    Multiple focused queries are used instead of one query
    for the complete claim.

    The HybridRetriever is shared across cases so that the
    embedding model, FAISS index, BM25 index, and reranker
    are initialized only once per process.
    """

    # ---------------------------------------------------------
    # Reuse shared retriever
    # ---------------------------------------------------------

    retriever = HybridRetriever.get_instance()

    dimensions = state.get(
        "decision_dimensions",
        [],
    )

    all_results: dict[str, dict[str, Any]] = {}

    trace = state.get(
        "trace",
        [],
    )

    # ---------------------------------------------------------
    # Multi-query retrieval
    # ---------------------------------------------------------

    for dimension in dimensions:

        queries = QUERY_MAP.get(
            dimension,
            [dimension.replace("_", " ")],
        )

        for query in queries:

            results = retriever.retrieve(
                query,
                top_k=3,
                retrieval_k=12,
            )

            for result in results:

                chunk_id = result["chunk_id"]

                if chunk_id not in all_results:

                    all_results[chunk_id] = {
                        **result,
                        "retrieval_queries": [query],
                        "dimensions": [dimension],
                    }

                else:

                    all_results[chunk_id][
                        "retrieval_queries"
                    ].append(query)

                    if dimension not in all_results[
                        chunk_id
                    ]["dimensions"]:

                        all_results[chunk_id][
                            "dimensions"
                        ].append(dimension)

    # ---------------------------------------------------------
    # Sort strongest evidence first
    # ---------------------------------------------------------

    evidence = list(
        all_results.values()
    )

    evidence.sort(
        key=lambda x: x.get(
            "rerank_score",
            0,
        ),
        reverse=True,
    )

    # Keep a manageable evidence set.
    evidence = evidence[:40]

    # ---------------------------------------------------------
    # Trace
    # ---------------------------------------------------------

    trace.append(
        {
            "agent": "PolicyEvidenceAgent",
            "action": (
                "Executed multi-query hybrid retrieval "
                "using shared policy retriever."
            ),
            "dimensions_investigated": dimensions,
            "evidence_chunks_retrieved": len(evidence),
        }
    )

    return {
        "retrieved_evidence": evidence,
        "trace": trace,
    }
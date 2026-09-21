# Policy-Aware Multi-Agent RAG Evaluation Report

## Executive Summary

- Total cases: **19**
- Decision accuracy: **100.00%**
- Validation pass rate: **100.00%**
- Total NEEDS_REVIEW cases: **3**
- Abstention requirement (>=2): **PASS**

## Public

- Cases: **12**
- Decision accuracy: **100.00%**
- Correct decisions: **12/12**
- Validation pass rate: **100.00%**
- NEEDS_REVIEW: **2** (16.67%)
- Retrieval evidence coverage: **100.00%**
- Average retrieved chunks: **19.42**
- Citation hit rate: **100.00%**
- Citation completeness: **100.00%**

### Decision distribution

- `ADMISSIBLE_WITH_LIMITS`: 6
- `NOT_ADMISSIBLE`: 4
- `NEEDS_REVIEW`: 2

### Case-level results

| Case | Expected | Actual | Correct | Validation | Evidence | Citations |
|---|---|---|---|---|---:|---:|
| PUB-001 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 18 | 13 |
| PUB-002 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 18 | 10 |
| PUB-003 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 19 | 10 |
| PUB-004 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 20 | 14 |
| PUB-005 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 19 | 14 |
| PUB-006 | NEEDS_REVIEW | NEEDS_REVIEW | YES | PASS | 21 | 10 |
| PUB-007 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 18 | 13 |
| PUB-008 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 21 | 11 |
| PUB-009 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 18 | 13 |
| PUB-010 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 21 | 16 |
| PUB-011 | NEEDS_REVIEW | NEEDS_REVIEW | YES | PASS | 18 | 10 |
| PUB-012 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 22 | 11 |

## Candidate

- Cases: **7**
- Decision accuracy: **100.00%**
- Correct decisions: **7/7**
- Validation pass rate: **100.00%**
- NEEDS_REVIEW: **1** (14.29%)
- Retrieval evidence coverage: **100.00%**
- Average retrieved chunks: **20.29**
- Citation hit rate: **100.00%**
- Citation completeness: **100.00%**

### Decision distribution

- `ADMISSIBLE_WITH_LIMITS`: 3
- `NEEDS_REVIEW`: 1
- `NOT_ADMISSIBLE`: 3

### Case-level results

| Case | Expected | Actual | Correct | Validation | Evidence | Citations |
|---|---|---|---|---|---:|---:|
| CAND-001 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 20 | 13 |
| CAND-002 | NEEDS_REVIEW | NEEDS_REVIEW | YES | PASS | 19 | 10 |
| CAND-003 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 19 | 10 |
| CAND-004 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 22 | 11 |
| CAND-005 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 21 | 13 |
| CAND-006 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | YES | PASS | 22 | 11 |
| CAND-007 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | YES | PASS | 19 | 13 |

## Abstention Requirement

- Public: **2**
- Candidate: **1**
- Combined: **3**
- Required minimum: **2**
- Status: **PASS**

## Citation Quality

Citation hit rate checks whether a cited chunk ID was actually present in retrieved evidence. It is a traceability metric, not semantic proof that the policy text supports the decision.

Semantic citation correctness therefore remains a manual-review item.

## Retrieval Quality

The system uses dense and sparse retrieval with reranking. Evidence coverage and retrieved-chunk statistics are reported.

Formal Recall@K is not reported because there is no manually annotated gold relevance set.

## Current Failure / Review Analysis

### Public
- No decision mismatches.
- Review cases: PUB-006, PUB-011

### Candidate
- No decision mismatches.
- Review cases: CAND-002

## Historical Development Failures and Improvements

### 1. Incomplete candidate decision dimensions

**Symptom:** Several candidate cases initially received only waiting_periods, hospital_definition, and medical_necessity dimensions, so relevant exclusions and limits were not investigated.

**Root cause:** Case Analysis Agent generated an incomplete investigation plan for some case types.

**Improvement:** Expanded case-dimension mapping for cosmetic, experimental, day-care, expense-limit, pre/post, and related policy dimensions.

### 2. False missing-evidence condition for pre/post expenses

**Symptom:** Eligible cases with supplied pre/post timing and condition facts were temporarily treated as requiring review.

**Root cause:** Decision logic required additional evidence even when the synthetic claim facts already supplied the relevant condition and timing.

**Improvement:** Expense calculation now uses supplied claim facts for applicable pre/post windows while citing the policy rule.

### 3. Citation validation mismatch

**Symptom:** A decision could temporarily contain a citation whose chunk ID was not present in retrieved evidence.

**Root cause:** Citation construction and retrieved-evidence tracking were not initially guaranteed to use the same chunk IDs.

**Improvement:** Decision citations are derived from retrieved evidence and validation checks citation traceability.

## Reproducibility

```bash
python -m pytest -q
python -m src.workflow
python evaluation/evaluate.py
```

## Limitations

1. Formal retrieval Recall@K requires a gold relevance annotation set.
2. Citation hit rate is chunk-level grounding, not semantic citation correctness.
3. Decision accuracy depends on the documented gold labels for the evaluation cases.
4. Confidence is descriptive and is not treated as a calibrated probability.

# Architecture and Design Note

## 1. Objective

The system is designed to make policy-grounded decisions over synthetic health-insurance claim cases. The supplied policy is treated as the authoritative source. The architecture prioritizes evidence traceability and safe abstention over unconstrained language-model answers.

## 2. State flow

A claim enters the workflow as structured data and is represented by a shared typed state.

```text
Claim Input
   |
   v
Case Analysis Agent
   |
   +--> facts
   +--> decision dimensions
   +--> missing evidence
   +--> investigation plan
   |
   v
Policy Evidence Agent
   |
   +--> dense retrieval
   +--> BM25 retrieval
   +--> fusion
   +--> reranking
   |
   v
Coverage & Exclusion Agent
   |
   +--> coverage
   +--> definitions
   +--> waiting periods
   +--> exclusions
   +--> limits
   |
   v
Decision Agent
   |
   +--> decision
   +--> confidence
   +--> findings
   +--> missing evidence
   +--> expense/limit calculations
   |
   v
Validation Agent
   |
   +--> citation grounding
   +--> unsupported claims
   +--> validation status
   |
   v
Structured API response
```

The agents exchange structured state. The application does not expose hidden chain-of-thought.

## 3. Agent boundaries

### Case Analysis Agent

Responsible for determining what must be investigated. It does not make the final coverage decision.

### Policy Evidence Agent

Responsible for finding policy evidence. It does not decide whether a claim is payable.

### Coverage & Exclusion Agent

Responsible for translating retrieved policy clauses into structured findings about coverage, exclusions, waiting periods, definitions, and limits.

### Decision Agent

Responsible for combining the specialist findings and claim facts into the final allowed status.

### Validation Agent

Responsible for checking that material claims in the final result are grounded in retrieved evidence and for flagging unsupported claims.

This separation prevents the workflow from degenerating into repeated versions of one generic prompt.

## 4. Retrieval design

The policy is indexed into meaningful chunks rather than arbitrary fixed-size slices. Each chunk retains source, page, section, and chunk identifier.

Two retrieval channels are used:

1. Dense semantic retrieval for conceptual similarity.
2. BM25 sparse retrieval for exact policy terminology and lexical matches.

The candidate sets are fused and reranked before being passed to the specialist reasoning agents.

This combination is useful because policy language contains both semantic concepts and exact terms such as waiting-period names, exclusions, percentages, and defined phrases.

## 5. Evidence contract

A material finding should point to policy evidence. Citation records include:

- source
- page
- section
- chunk ID

The validation stage checks citation grounding against retrieved evidence.

When evidence is missing or policy support is insufficient, the system uses `NEEDS_REVIEW` rather than inventing a conclusion.

## 6. Decision statuses

The workflow supports:

- `ADMISSIBLE`
- `ADMISSIBLE_WITH_LIMITS`
- `PARTIALLY_ADMISSIBLE`
- `NOT_ADMISSIBLE`
- `NEEDS_REVIEW`

The distinction between admissibility and limits is important because a claim may satisfy coverage conditions while still being affected by category-specific caps or deductions.

## 7. API and frontend

FastAPI provides:

- `GET /health`
- `POST /analyze`

Streamlit provides reviewer-facing claim entry and supports both structured manual entry and JSON paste/upload.

The frontend exposes concise audit information:

- final decision
- confidence
- findings
- limits
- missing evidence
- citations
- validation
- agent trace

## 8. Important trade-offs

### FAISS versus managed vector infrastructure

FAISS keeps the assignment reproducible and inexpensive locally. The trade-off is that it does not provide the operational features of a managed vector database.

### Local/low-cost embeddings

Using a sentence-transformer model avoids requiring a paid embedding API. The trade-off is model-download size and local compute requirements.

### Hybrid retrieval complexity

BM25 plus dense retrieval adds implementation complexity compared with vector-only search, but improves robustness for exact policy terms.

### Rule-assisted decision logic

Some deterministic policy checks are implemented in structured Python logic rather than relying entirely on generative reasoning. This reduces unsupported conclusions for high-value conditions such as waiting periods and expense limits.

### Abstention over forced completion

The system intentionally sacrifices some apparent answer coverage when required evidence is unavailable. This is a deliberate reliability trade-off for the assignment's evidence-grounding requirement.

## 9. Reliability strategy

The evaluation suite covers waiting periods, category-specific limits, exclusions, insufficient evidence, irrelevant claimant attributes, uncertain required conditions, and citation grounding.

Three development failures were specifically corrected:

1. incomplete investigation dimensions;
2. incorrect pre/post hospitalization evidence handling;
3. citation/retrieval mismatch during validation.

## 10. Current limitations

The evaluation does not contain a manually annotated retrieval relevance set, so formal Recall@K is not claimed. Citation hit rate is used as a reproducible grounding metric.

The application is an assignment demonstration over synthetic cases and should not be treated as a production insurance adjudication service without additional controls, testing, monitoring, security review, and domain validation.

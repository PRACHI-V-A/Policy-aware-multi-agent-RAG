# Policy-Aware Multi-Agent RAG Claim Decision Engine

A production-style AI system for analyzing synthetic health-insurance claims against an authoritative policy document. The system combines hybrid retrieval, specialized agents, structured evidence, citation validation, and abstention when required evidence is missing.

## Assignment scope

The supplied policy is the authoritative source for policy decisions. Claimant, hospital, date, amount, and claim facts are synthetic test inputs. The system does not use external insurance or medical knowledge to invent policy conclusions.

If the available evidence is insufficient to make a safe policy-grounded decision, the system returns `NEEDS_REVIEW`.

## Architecture

```text
                         ┌─────────────────────┐
                         │  Claim Input        │
                         │  Form / JSON         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Case Analysis Agent │
                         │ facts + dimensions  │
                         │ + investigation plan│
                         └──────────┬──────────┘
                                    │ structured state
                                    ▼
                    ┌───────────────────────────────┐
                    │      Policy Evidence Agent    │
                    │                               │
                    │ Dense semantic retrieval      │
                    │ BM25 sparse retrieval         │
                    │ Fusion                         │
                    │ Reranking                      │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │ Coverage & Exclusion Agent    │
                    │ coverage / exclusions /       │
                    │ waiting periods / limits      │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   Decision Agent    │
                         │ decision + limits + │
                         │ missing evidence    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Validation Agent   │
                         │ citation grounding  │
                         │ unsupported claims  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │ FastAPI structured response      │
                  │ + Streamlit reviewer interface   │
                  └──────────────────────────────────┘
```

## Multi-agent design

The workflow uses five specialized agents:

1. **Case Analysis Agent** — extracts case facts, identifies decision dimensions, detects missing fields, and creates the investigation plan.
2. **Policy Evidence Agent** — retrieves relevant policy clauses using dense + BM25 retrieval, fusion, and reranking.
3. **Coverage & Exclusion Agent** — evaluates coverage, definitions, waiting periods, exclusions, and applicable limits.
4. **Decision Agent** — combines specialist findings into one structured decision.
5. **Validation Agent** — checks that material decision statements are supported by retrieved policy evidence and reports unsupported claims.

Agents exchange a typed structured claim state rather than only free-form text. Hidden chain-of-thought is not exposed.

## Policy ingestion and retrieval

The supplied policy PDF is indexed into meaningful chunks with traceability metadata including:

- source policy
- page
- section/heading
- chunk ID

Retrieval is hybrid:

- dense semantic retrieval using sentence-transformers / FAISS
- sparse lexical retrieval using BM25
- fusion of dense and sparse candidates
- reranking before evidence reaches downstream reasoning agents

Citation records retain source/page/section/chunk information so decisions can be traced back to the policy.

## Decision contract

Supported decisions:

- `ADMISSIBLE`
- `ADMISSIBLE_WITH_LIMITS`
- `PARTIALLY_ADMISSIBLE`
- `NOT_ADMISSIBLE`
- `NEEDS_REVIEW`

The API response includes:

- `case_id`
- `decision`
- `confidence`
- `key_findings`
- `coverage_findings`
- `exclusion_findings`
- `limit_findings`
- `missing_evidence`
- `citations`
- `validation_status`
- `unsupported_claims`
- `trace`
- `investigation_plan`
- `expense_calculation`

When required evidence is missing or policy support is insufficient, the system abstains with `NEEDS_REVIEW` instead of guessing.

## API

### Start locally

```bash
uvicorn src.api:app --reload
```

### Health

```text
GET /health
```

### Analyze

```text
POST /analyze
```

Example:

```json
{
  "case_id": "PUB-006",
  "policy_id": "POLICY-001",
  "task": "Evaluate hospitalization admissibility",
  "policy_start_date": "2024-01-01",
  "claim_date": "2026-07-03",
  "coverage_months": 30,
  "sum_insured": 1000000,
  "prior_insurer_continuous_years": 0,
  "claimant": {"age": 35},
  "hospital": {
    "name": "Test Hospital",
    "network": false,
    "registered": null
  },
  "treatment": {
    "type": "Inpatient",
    "admission_hours": 96,
    "diagnosis": "Acute infection",
    "procedure": "Inpatient treatment",
    "pre_existing": false,
    "experimental": false,
    "medical_necessity_confirmed": null
  },
  "expenses": {
    "room_inr": 0,
    "doctor_inr": 0,
    "medicines_diagnostics_inr": 0,
    "pre_claimed_inr": 0,
    "post_claimed_inr": 0,
    "ambulance_inr": 0
  },
  "evidence_context": {
    "hospital_registered": null,
    "medical_necessity_confirmed": null
  },
  "documents": ["claim_form", "discharge_summary"]
}
```

The above case intentionally leaves hospital eligibility and medical necessity unresolved and should abstain with `NEEDS_REVIEW`.

## Streamlit frontend

Start locally:

```bash
streamlit run src/streamlit_app.py
```

The frontend supports:

- manual claim entry
- paste JSON claim
- upload JSON claim
- final decision and confidence
- findings
- limits/deductions
- missing evidence
- policy citations
- validation result
- multi-agent trace
- investigation plan
- raw API response

## Evaluation

Run:

```bash
python evaluation/evaluate.py
```

The current evaluation suite contains:

- 12 supplied public cases
- 7 candidate-created cases
- 19 total cases
- 3 `NEEDS_REVIEW` cases

Current recorded evaluation:

| Metric | Result |
|---|---:|
| Total cases | 19 |
| Overall decision accuracy | 100% (19/19) |
| Public decision accuracy | 100% (12/12) |
| Candidate decision accuracy | 100% (7/7) |
| Validation pass rate | 100% |
| NEEDS_REVIEW cases | 3 |
| Abstention requirement | PASS |
| Public citation hit rate | 100% |
| Candidate citation hit rate | 100% |

The citation hit-rate metric checks that cited chunk IDs are grounded in retrieved evidence. It is not a manually annotated semantic Recall@K benchmark. No formal gold relevance set is currently maintained, so semantic Recall@K is not reported.

The evaluation produces:

```text
evaluation/results/evaluation_report.json
evaluation/results/evaluation_report.md
```

Run the automated tests with:

```bash
python -m pytest -q
```

## Development failure analysis

Three important development failures were identified and corrected:

### 1. Incomplete investigation dimensions

The first Case Analysis mapping omitted some dimensions such as limits and exclusions.

**Root cause:** incomplete case-to-policy dimension mapping.

**Fix:** expanded the decision-dimension mapping so the investigation plan includes waiting periods, hospital eligibility, medical necessity, limits, exclusions, and related conditions.

### 2. False missing-evidence results for valid pre/post cases

Some eligible cases initially abstained because same-condition and timing facts were not being interpreted consistently.

**Root cause:** incorrect handling of supplied case facts relative to policy pre/post hospitalization windows.

**Fix:** aligned the calculation with the supplied claim facts and policy-defined timing windows.

### 3. Citation validation mismatch

A citation could initially refer to a chunk that was not present in the retrieved evidence set.

**Root cause:** citation generation and retrieved-evidence validation were not tightly coupled.

**Fix:** citations are derived/validated against retrieved evidence before the final result is accepted.

## Reproducibility

Recommended local sequence:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python evaluation/evaluate.py
uvicorn src.api:app --reload
streamlit run src/streamlit_app.py
```

For a clean environment, use Python 3.10+ and a virtual environment.

## Deployment

The recommended deployment uses two web services from the same repository:

- FastAPI backend
- Streamlit frontend

The frontend receives the deployed backend URL through the `API_URL` environment variable. Do not hardcode credentials or API keys.

See `render.yaml` for the Render Blueprint configuration.

## Security and configuration

- Do not commit `.env` files, credentials, or API keys.
- Use environment variables for deployment configuration.
- Candidate data and supplied policy materials are kept separate from source code according to the repository's `.gitignore` policy.
- The application exposes concise auditable traces rather than hidden chain-of-thought.

## Known limitations

1. The supplied assignment data is synthetic and does not establish real-world claims handling behavior.
2. Citation hit rate measures chunk-ID grounding, not independent semantic correctness.
3. A manually annotated retrieval relevance set is not currently available, so formal Recall@K is not reported.
4. The current deployment is intended for assignment demonstration and is not a production insurance claims-processing service.
5. Model downloads may be slower or rate-limited when using unauthenticated Hugging Face access.

## Repository structure

```text
candidate_data/       Candidate-created evaluation cases
docs/                 Architecture and failure-analysis documentation
evaluation/           Evaluation script and generated reports
schema/               Supplied/request schema material
src/                  Agents, retrieval, workflow, API, frontend
tests/                Automated tests
policy/               Supplied policy material (ignored/private)
```

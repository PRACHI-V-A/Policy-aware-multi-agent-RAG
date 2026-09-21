# Failure Analysis

This document records important development failures encountered while building the policy-aware claim decision engine.

## Failure 1 — Incomplete investigation dimensions

### Symptom

Early candidate cases did not consistently investigate limits and exclusions. The Case Analysis Agent produced an incomplete investigation plan.

### Root cause

The initial mapping from claim facts to policy decision dimensions was too narrow. It emphasized waiting periods, hospital definition, and medical necessity while omitting some downstream dimensions.

### Correction

The mapping was expanded to include:

- waiting periods
- hospital eligibility
- medical necessity
- exclusions
- category-specific limits
- pre/post hospitalization
- related policy conditions

### Result

Candidate cases covering limits and exclusions were subsequently evaluated correctly.

---

## Failure 2 — False abstention for valid pre/post hospitalization cases

### Symptom

Some otherwise eligible cases were incorrectly treated as lacking evidence for pre/post hospitalization treatment.

### Root cause

The implementation did not initially reconcile the supplied claim facts with the policy-defined timing windows and same-condition requirement.

### Correction

The calculation was updated to use the supplied case facts together with the policy windows and to distinguish a genuinely missing condition from a condition already established by the synthetic case.

### Result

The affected public cases no longer falsely abstained.

---

## Failure 3 — Citation validation mismatch

### Symptom

A generated finding could cite a chunk that was not present in the final retrieved evidence set.

### Root cause

Citation construction and retrieval validation were initially separate.

### Correction

Final citations are now derived and checked against the retrieved evidence before validation accepts the result.

### Result

The current evaluation reports 100% citation hit rate for both public and candidate cases.

---

## Evaluation interpretation

The three failures above are development failures, not current evaluation failures. The final evaluation should be run with:

```bash
python evaluation/evaluate.py
```

and the automated test suite with:

```bash
python -m pytest -q
```

Current recorded results are:

- 19 total cases
- 19/19 decision accuracy
- 100% validation pass rate
- 3 `NEEDS_REVIEW` cases
- 100% citation hit rate for public cases
- 100% citation hit rate for candidate cases

The citation hit-rate metric checks chunk-ID grounding against retrieved evidence. It should not be interpreted as an independently human-annotated semantic citation-correctness score.

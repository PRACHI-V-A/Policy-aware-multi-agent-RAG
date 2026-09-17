# Policy-Aware Multi-Agent RAG Claim Decision Engine

An AI-powered insurance claim decision system built using Retrieval-Augmented Generation (RAG) and a multi-agent workflow.

## Project Overview

This project analyzes synthetic health-insurance claim cases against a supplied insurance policy and produces an evidence-backed claim decision.

The system is designed to:

- Retrieve relevant policy evidence
- Analyze claim coverage and exclusions
- Identify applicable limits and waiting periods
- Abstain when available evidence is insufficient
- Provide traceable policy citations
- Validate that decision statements are supported by retrieved evidence

## Planned Architecture

Claim Input
→ Case Analysis Agent
→ Hybrid Policy Retrieval
→ Coverage & Exclusion Analysis
→ Decision Agent
→ Validation Agent
→ Final Decision

## Technology Stack

- Python
- LangGraph
- FastAPI
- Streamlit
- FAISS
- BM25
- Sentence Transformers
- Reranker
- LLM

## Project Status

🚧 Development in progress
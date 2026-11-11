# RAG LangChain & DeepAgents

A Retrieval-Augmented Generation (RAG) system with multi-agent chunk analysis built using **LangChain**, **DeepAgents**, and **OpenRouter**.

---

## Overview

This project implements an intelligent document Q&A workflow. Instead of overwhelming the main language model with large context windows or raw chunks, it uses an orchestrator-subagent architecture:

1. **Document Ingestion & Indexing**: Reads PDF documents, splits them into semantic chunks, and indexes them into an `InMemoryVectorStore` using custom OpenRouter embeddings (`nvidia/nemotron-3-embed-1b:free`).
2. **Agentic Retrieval**: The main agent queries the vector store via a custom tool (`search_documentation`) and stores retrieved chunks in an isolated virtual filesystem (`/retrieved/`).
3. **Parallel Subagent Analysis**: Up to multiple `chunk-analyst` subagents run concurrently to analyze individual chunk files and summarize key facts and sources.
4. **Synthesis & Citation**: The coordinator synthesizes subagent summaries into a coherent, source-referenced final response.

---

## Project Structure

```text
├── agent.py                 # Main coordinator agent & subagent definition
├── indexing.py              # PDF chunking and vector store indexing
├── tools.py                 # Retrieval tool & StateBackend virtual filesystem
├── prompts.py               # Prompt templates and subagent delegation instructions
├── openrouter_embeddings.py # Custom LangChain Embeddings class for OpenRouter
├── file_reader.py           # Multi-page PDF text extraction utility
├── pdfs/                    # Source PDF documents
└── pyproject.toml           # Project dependencies and configuration
```

---

## Getting Started

### 1. Prerequisites
- Python >= 3.14
- Package manager: [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 3. Installation
```bash
uv sync
```

### 4. Running the Agent
Run the indexing pipeline and test the deep agent:
```bash
python indexing.py
python agent.py
```

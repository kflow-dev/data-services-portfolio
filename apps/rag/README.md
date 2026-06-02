# Multi-media RAG

Retrieval Augmented Generation with pgvector and LLM-based Q&A.

## Overview

Multi-media RAG ingests documents, embeds them into pgvector, and provides grounded Q&A with citations from retrieved documents.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-media RAG                          │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py ingest document.txt --source my-docs    │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import RAG                          │
├─────────────────────────────────────────────────────────────┤
│  Data: documents, embeddings in pgvector                    │
│  Model: Embeddings + LLM Retrieval + Generation             │
│  Features: text_chunks, embeddings, metadata               │
│  Output: Answers with document citations                   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer psycopg2
```

## Usage

### CLI

```bash
# Ingest document
python cli.py ingest document.txt --source my-docs

# Query RAG system
python cli.py ask "What does the document say about X?"

# List available documents
python cli.py list-documents
```

### Streamlit UI

```bash
streamlit run apps/rag/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/rag/notebooks/rag_pipeline.ipynb
```

## Project Structure

```
rag/
├── cli.py
├── streamlit_app.py
├── README.md
├── notebooks/
│   └── rag_pipeline.ipynb
└── supabase/
    └── migrations/
        └── pgvector_schema.sql
```

## License

See parent directory for license information.

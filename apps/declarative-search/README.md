# Declarative Search

Natural language search interface with query understanding.

## Overview

Declarative Search enables natural language queries with intent understanding and result ranking.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Declarative Search                          │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py search --query "best restaurants nearby"│
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import DeclarativeSearch            │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/documents.csv                         │
│  Model: NLP Query Understanding + Semantic Search           │
│  Features: query_text, document_content, embeddings        │
│  Output: Ranked search results with relevance scores       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer sentence-transformers
```

## Usage

### CLI

```bash
# Search with natural language
python cli.py search --query "best restaurants nearby"

# Get query intent analysis
python cli.py analyze --query "find flights to paris"

# Search with filters
python cli.py search --query "python tutorials" --type video
```

### Streamlit UI

```bash
streamlit run apps/declarative-search/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/declarative-search/notebooks/nlp_search.ipynb
```

## Project Structure

```
declarative-search/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── documents.csv
└── notebooks/
    └── nlp_search.ipynb
```

## License

See parent directory for license information.

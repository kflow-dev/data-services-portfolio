# SciTubbies

Scientific paper recommender with content analysis.

## Overview

SciTubbies recommends research papers based on reading interests, field of study, and citation analysis.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SciTubbies                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --field "machine-learning"    │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import SciTubbies                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/papers.csv                            │
│  Model: Paper Embedding + Citation Network                  │
│  Features: abstract, keywords, citations, field            │
│  Output: Research paper recommendations with relevance     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Get paper recommendations
python cli.py recommend --field "machine-learning" --level "advanced"

# Find related papers
python cli.py related --paper-id P001 --count 5

# Analyze research trends
python cli.py trends --field "ai" --years 5
```

### Streamlit UI

```bash
streamlit run apps/scitubbies/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/scitubbies/notebooks/paper_recommendation.ipynb
```

## Project Structure

```
scitubbies/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── papers.csv
└── notebooks/
    └── paper_recommendation.ipynb
```

## License

See parent directory for license information.

# EBooks

Personalized book recommender with content analysis.

## Overview

EBooks recommends books based on reading preferences, genre preferences, and content similarity analysis.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      EBooks                                 │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --genre "sci-fi"              │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import EBooks                       │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/books.csv                             │
│  Model: Content-based + Collaborative Filtering             │
│  Features: genre, author, rating, content_similarity       │
│  Output: Personalized book recommendations                 │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Get book recommendations
python cli.py recommend --genre "sci-fi" --rating 4+

# Find similar books
python cli.py similar --book-id BK001 --count 5

# Analyze reading preferences
python cli.py preferences --history "BK001,BK002,BK003"
```

### Streamlit UI

```bash
streamlit run apps/ebooks/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/ebooks/notebooks/book_recommendation.ipynb
```

## Project Structure

```
ebooks/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── books.csv
└── notebooks/
    └── book_recommendation.ipynb
```

## License

See parent directory for license information.

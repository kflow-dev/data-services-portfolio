# My Wardrobe

Outfit recommender with seasonal and occasion-based suggestions.

## Overview

My Wardrobe recommends outfits based on weather, occasion, personal style, and available clothing items.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    My Wardrobe                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --occasion "business meeting" │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import MyWardrobe                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/clothing_items.csv                    │
│  Model: Outfit Combinator + Style Recommender               │
│  Features: item_type, season, color, occasion              │
│  Output: Coordinated outfit recommendations                │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Get outfit recommendation
python cli.py recommend --occasion "business meeting" --season "spring"

# Plan wardrobe for trip
python cli.py pack --destination "London" --days 5 --season "summer"

# Analyze outfit combinations
python cli.py combinations --items C001,C002,C003
```

### Streamlit UI

```bash
streamlit run apps/mywardrobe/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/mywardrobe/notebooks/outfit_recommendation.ipynb
```

## Project Structure

```
mywardrobe/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── clothing_items.csv
└── notebooks/
    └── outfit_recommendation.ipynb
```

## License

See parent directory for license information.

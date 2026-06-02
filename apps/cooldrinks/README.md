# Cool Drinks

Personalized beverage recommender based on preferences and context.

## Overview

Cool Drinks recommends beverages based on user taste preferences, weather, time of day, and occasion.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cool Drinks                             │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --context "summer,outdoor"    │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import CoolDrinks                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/beverages.csv                         │
│  Model: Context-aware Recommender                           │
│  Features: taste_profile, weather, occasion, time_of_day   │
│  Output: Personalized beverage recommendations             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Get beverage recommendations
python cli.py recommend --context "summer,outdoor" --taste "sweet"

# Find drinks for occasion
python cli.py occasion --event "party" --guests 20

# Compare beverage options
python cli.py compare --types "coffee,tea"
```

### Streamlit UI

```bash
streamlit run apps/cooldrinks/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/cooldrinks/notebooks/beverage_recommendation.ipynb
```

## Project Structure

```
cooldrinks/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── beverages.csv
└── notebooks/
    └── beverage_recommendation.ipynb
```

## License

See parent directory for license information.

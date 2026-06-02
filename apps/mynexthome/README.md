# My Next Home

Real estate recommender with property valuation.

## Overview

My Next Home recommends properties based on preferences, budget, and location with automated valuation models.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   My Next Home                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --budget 500000 --location "NYC"│
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import MyNextHome                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/properties.csv                        │
│  Model: HREvaluation + Recommendation                       │
│  Features: location, size, bedrooms, price, amenities      │
│  Output: Property recommendations with valuation estimates │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Find properties
python cli.py recommend --budget 500000 --location "NYC" --bedrooms 2

# Get property valuation
python cli.py valuate --property-id P001

# Compare neighborhoods
python cli.py compare --areas "Brooklyn,Queens" --criteria "schools,transit"
```

### Streamlit UI

```bash
streamlit run apps/mynexthome/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/mynexthome/notebooks/real_estate_recommender.ipynb
```

## Project Structure

```
mynexthome/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── properties.csv
└── notebooks/
    └── real_estate_recommender.ipynb
```

## License

See parent directory for license information.

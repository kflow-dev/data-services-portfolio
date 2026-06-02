# My Local Radar

Local events and services discovery platform.

## Overview

My Local Radar discovers local events, services, and points of interest based on location and preferences.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   My Local Radar                            │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py discover --location "New York"          │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import MyLocalRadar                 │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/local_events.csv                      │
│  Model: Location-based Recommender                          │
│  Features: location, category, distance, rating            │
│  Output: Local discoveries with relevance ranking          │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Discover local events
python cli.py discover --location "New York" --category events

# Find nearby services
python cli.py nearby --type restaurants --radius 5km

# Get recommendations
python cli.py recommend --location "SF" --preferences "outdoor,culture"
```

### Streamlit UI

```bash
streamlit run apps/mylocalradar/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/mylocalradar/notebooks/local_discovery.ipynb
```

## Project Structure

```
mylocalradar/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── local_events.csv
└── notebooks/
    └── local_discovery.ipynb
```

## License

See parent directory for license information.

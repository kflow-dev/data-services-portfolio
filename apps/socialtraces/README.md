# Social Traces

Social network analysis and contact relationship insights.

## Overview

Social Traces analyzes social connections, interaction patterns, and provides insights on network dynamics.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Social Traces                             │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py analyze --network contacts.csv          │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import SocialTraces                 │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/social_network.csv                    │
│  Model: Network Analysis + Community Detection              │
│  Features: connections, interaction_frequency, node_type   │
│  Output: Network metrics with community insights           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer networkx
```

## Usage

### CLI

```bash
# Analyze social network
python cli.py analyze --network contacts.csv

# Find influential contacts
python cli.py influence --network contacts.csv --top 10

# Detect communities
python cli.py communities --network contacts.csv
```

### Streamlit UI

```bash
streamlit run apps/socialtraces/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/socialtraces/notebooks/network_analysis.ipynb
```

## Project Structure

```
socialtraces/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── social_network.csv
└── notebooks/
    └── network_analysis.ipynb
```

## License

See parent directory for license information.

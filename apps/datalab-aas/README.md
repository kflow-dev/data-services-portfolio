# Data Lab as a Service

Managed cloud data science lab with collaborative features.

## Overview

Data Lab AAS provides a managed cloud environment for data science experimentation with collaborative tools, version control, and reproducible workflows.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Lab AAS                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py create-project --name myproject         │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import DataLabAAS                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/projects.csv                          │
│  Model: Resource Allocation + Collaboration Analysis        │
│  Features: project_type, team_size, resource_usage         │
│  Output: Lab configuration with resource recommendations   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Create new data science project
python cli.py create-project --name myproject --type ml

# Get resource recommendations
python cli.py resources --project-type deep-learning --team 5

# Collaborate on project
python cli.py collaborate --project proj_001 --users user1,user2
```

### Streamlit UI

```bash
streamlit run apps/datalab-aas/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/datalab-aas/notebooks/lab_management.ipynb
```

## Project Structure

```
datalab-aas/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── projects.csv
└── notebooks/
    └── lab_management.ipynb
```

## License

See parent directory for license information.

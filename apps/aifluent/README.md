# AI Fluent Career Path

Skill graph-based career development recommender.

## Overview

AI Fluent provides personalized career development recommendations using skill graphs and reinforcement learning. It maps current skills to target roles and suggests learning paths.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Fluent                                │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --role "Data Scientist"       │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import AIFluent                     │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/skills_graph.csv                      │
│  Model: Skill Graph + RL recommendation                     │
│  Features: current_skills, target_role, learning_history   │
│  Output: Personalized learning paths with skill gaps       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer networkx
```

## Usage

### CLI

```bash
# Get career recommendations
python cli.py recommend --role "Data Scientist" --skills "python,sql"

# Analyze skill gaps
python cli.py analyze-skills --current "python,excel" --target "ML Engineer"

# Generate learning path
python cli.py learning-path --role "Product Manager"
```

### Streamlit UI

```bash
streamlit run apps/aifluent/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/aifluent/notebooks/skill_graph_analysis.ipynb
```

## Project Structure

```
aifluent/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── skills_graph.csv
└── notebooks/
    └── skill_graph_analysis.ipynb
```

## License

See parent directory for license information.

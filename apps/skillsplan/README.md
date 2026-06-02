# Skills Plan

Personalized skills development planner with learning path optimization.

## Overview

Skills Plan creates personalized learning paths based on current skills, target roles, and available resources.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Skills Plan                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py plan --target "ML Engineer"             │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import SkillsPlan                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/skills_database.csv                   │
│  Model: Path Optimization + Skill Graph                     │
│  Features: current_skills, target_skills, learning_rate    │
│  Output: Optimized learning path with timeline             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Generate learning plan
python cli.py plan --target "ML Engineer" --current "python,basic"

# Get skill recommendations
python cli.py recommend --role "Data Scientist" --gap "statistics"

# Estimate learning time
python cli.py estimate --skills "deep-learning,nlp" --hours-week 10
```

### Streamlit UI

```bash
streamlit run apps/skillsplan/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/skillsplan/notebooks/skills_planning.ipynb
```

## Project Structure

```
skillsplan/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── skills_database.csv
└── notebooks/
    └── skills_planning.ipynb
```

## License

See parent directory for license information.

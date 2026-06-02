# JobPromis

AI career advisor for job matching and career path recommendations.

## Overview

JobPromis analyzes job descriptions and user profiles to provide job matching scores and career path recommendations.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     JobPromis                               │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py match --job-id JD001                    │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import JobPromis                    │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/job_descriptions.csv                  │
│  Model: NLP Job Matching + Career Path RL                   │
│  Features: job_skills, experience, salary, location        │
│  Output: Match scores with career development paths        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Get job match score
python cli.py match --job-id JD001 --profile my_profile.json

# Get career recommendations
python cli.py recommend --current-role "Junior Dev" --target "Tech Lead"

# Analyze job requirements
python cli.py analyze --job-id JD001
```

### Streamlit UI

```bash
streamlit run apps/jobpromis/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/jobpromis/notebooks/job_matching.ipynb
```

## Project Structure

```
jobpromis/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── job_descriptions.csv
└── notebooks/
    └── job_matching.ipynb
```

## License

See parent directory for license information.

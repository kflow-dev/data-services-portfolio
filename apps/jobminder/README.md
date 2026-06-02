# JobMinder

Job application tracker with interview preparation assistant.

## Overview

JobMinder tracks job applications, schedules interviews, and provides interview preparation with mock questions and feedback.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      JobMinder                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py add-application --company "Google"      │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import JobMinder                    │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/jobs.csv                              │
│  Model: NLP Job Matching + Interview Prep                   │
│  Features: job_description, skills, interview_type         │
│  Output: Application tracking with prep resources          │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Add job application
python cli.py add-application --company "Google" --role "Software Engineer"

# Get interview preparation
python cli.py prep --role "Data Scientist" --experience "mid-level"

# Track application status
python cli.py status --application-id APP001
```

### Streamlit UI

```bash
streamlit run apps/jobminder/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/jobminder/notebooks/job_tracking.ipynb
```

## Project Structure

```
jobminder/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── jobs.csv
└── notebooks/
    └── job_tracking.ipynb
```

## License

See parent directory for license information.

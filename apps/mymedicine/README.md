# MyMedicine

Travel medicine assistant with drug interaction checker.

## Overview

MyMedicine provides travel medicine recommendations, drug interaction checking, and healthcare location services for travelers.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MyMedicine                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py check-interaction --drugs "aspirin,warfarin"│
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import MyMedicine                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/medications.csv                       │
│  Model: Drug Interaction Detection + Travel Recommendations │
│  Features: drug_names, condition, destination              │
│  Output: Interaction warnings with travel health advice    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Check drug interactions
python cli.py check-interaction --drugs "aspirin,warfarin"

# Get travel health recommendations
python cli.py travel --destination "Thailand" --duration 2weeks

# Find nearby pharmacies
python cli.py pharmacy --location "Paris" --type 24hour
```

### Streamlit UI

```bash
streamlit run apps/mymedicine/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/mymedicine/notebooks/travel_medicine.ipynb
```

## Project Structure

```
mymedicine/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── medications.csv
└── notebooks/
    └── travel_medicine.ipynb
```

## License

See parent directory for license information.

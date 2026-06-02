# My Smart Diet

Personalized meal planning with nutritional optimization.

## Overview

My Smart Diet creates personalized meal plans with nutritional goals, dietary restrictions, and calorie targets.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   My Smart Diet                             │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py plan --calories 2000 --diet "keto"      │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import MySmartDiet                  │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/recipes.csv                           │
│  Model: Nutritional Optimization + Recipe Recommendation    │
│  Features: calories, macros, allergens, cuisine_type       │
│  Output: Weekly meal plans with nutritional breakdown      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Create meal plan
python cli.py plan --calories 2000 --diet "keto" --duration 7

# Get recipe recommendations
python cli.py recipes --macros "high-protein,low-carb"

# Analyze nutritional balance
python cli.py analyze --plan-id PLAN001
```

### Streamlit UI

```bash
streamlit run apps/mysmartdiet/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/mysmartdiet/notebooks/meal_planning.ipynb
```

## Project Structure

```
mysmartdiet/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── recipes.csv
└── notebooks/
    └── meal_planning.ipynb
```

## License

See parent directory for license information.

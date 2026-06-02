# EMagazzine

E-commerce inventory and order management optimizer.

## Overview

EMagazzine optimizes e-commerce inventory levels and order fulfillment with demand prediction and stock optimization.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EMagazzine                              │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py optimize-inventory --category electronics│
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import EMagazzine                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/inventory.csv                         │
│  Model: Demand Forecasting + Optimization                   │
│  Features: sales_history, stock_level, seasonality         │
│  Output: Optimal inventory levels with reorder points      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Optimize inventory for category
python cli.py optimize-inventory --category electronics --horizon 30

# Get reorder recommendations
python cli.py reorder --warehouse W001

# Forecast demand
python cli.py forecast --product SKU123 --weeks 12
```

### Streamlit UI

```bash
streamlit run apps/emagazzine/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/emagazzine/notebooks/inventory_optimization.ipynb
```

## Project Structure

```
emagazzine/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── inventory.csv
└── notebooks/
    └── inventory_optimization.ipynb
```

## License

See parent directory for license information.

# Cloud ML Estimator

Cloud ML infrastructure cost estimator and optimizer.

## Overview

Cloud ML Estimator calculates and optimizes cloud infrastructure costs for machine learning workloads, including compute, storage, and training costs.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Cloud ML Estimator                         │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py estimate --model-size 1B --epochs 10    │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import CloudMLEstimator             │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/pricing_data.csv                      │
│  Model: Cost Estimation + Optimization                      │
│  Features: model_size, epochs, compute_type, storage        │
│  Output: Cost breakdown with optimization suggestions      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Estimate cloud ML costs
python cli.py estimate --model-size 1B --epochs 10 --compute-type GPU

# Get cost optimization recommendations
python cli.py optimize --current-cost 5000 --model-size 10B

# Compare cloud providers
python cli.py compare --workload training --duration 24h
```

### Streamlit UI

```bash
streamlit run apps/cloud-ml-estimator/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/cloud-ml-estimator/notebooks/cost_analysis.ipynb
```

## Project Structure

```
cloud-ml-estimator/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── pricing_data.csv
└── notebooks/
    └── cost_analysis.ipynb
```

## License

See parent directory for license information.

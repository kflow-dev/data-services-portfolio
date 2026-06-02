# Customer Segmentation

Personalized customer persona creation with KMeans clustering.

## Overview

Customer Segmentation identifies distinct customer groups (personas) based on behavioral and demographic features using KMeans clustering. It provides:

- **Persona creation**: Automatically discovers customer segments
- **Silhouette validation**: Validates cluster quality
- **Representative customers**: Identifies typical customers per persona
- **Cluster profiling**: Describes each persona's characteristics

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Customer Segmentation                    │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py create-personas --count 4               │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import CustomerSegmentation         │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/customer_segmentation.csv            │
│  Model: KMeans clustering (sklearn)                        │
│  Features: age, income, spending_score, visit_frequency    │
│  Output: Personas with descriptions and size percentages   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit scikit-learn pandas numpy typer
```

## Usage

### CLI

```bash
# Create personas (default: 4 clusters)
python cli.py create-personas --data-dir data/synthetic --count 4

# Create personas with custom random seed
python cli.py create-personas --count 5 --seed 123

# Get representative customers for persona 0 (0-indexed)
python cli.py representative-customers 0 --count 5

# Evaluate clustering quality
python cli.py evaluate
```

### Streamlit UI

```bash
# From project root
streamlit run apps/segmentation/streamlit_app.py

# With custom port
streamlit run apps/segmentation/streamlit_app.py --server.port 8503
```

### Jupyter Notebook

```bash
# From project root
jupyter notebook apps/segmentation/notebooks/customer_segmentation_example.ipynb

# Or start jupyter and navigate to:
# apps/segmentation/notebooks/customer_segmentation_example.ipynb
```

### As a Library

```python
from catalog import CustomerSegmentation
import pandas as pd

# Load data
df = pd.read_csv("data/synthetic/customer_segmentation.csv")

# Initialize segmenter
segmenter = CustomerSegmentation(n_clusters=4, random_state=42)

# Create personas
result = segmenter.fit(df)
print(f"Silhouette Score: {result['silhouette_score']:.3f}")

# Get persona descriptions
for cluster_id, profile in result['clusters'].items():
    print(f"\nPersona {cluster_id + 1}: {profile['persona_name']}")
    print(f"  Size: {profile['size']} customers ({profile['percentage']}%)")

# Get representative customers for persona 0
rep_customers = segmenter.get_representative_customers(0, n_samples=5)
print(rep_customers)
```

## Model Details

### Algorithm

**KMeans Clustering**:
- Partitions data into k clusters based on feature similarity
- Minimizes within-cluster variance (inertia)
- Uses K-means++ initialization for better convergence
- Validates with silhouette score

**Silhouette Score Interpretation**:
- **> 0.7**: Strong cluster structure
- **> 0.5**: Reasonable structure
- **> 0.25**: Weak structure
- **< 0**: Overlapping clusters

### Features Used

| Feature | Description | Scaling |
|---------|-------------|---------|
| `age` | Customer age | StandardScaler |
| `income` | Annual income | StandardScaler |
| `spending_score` | Spending propensity (0-100) | StandardScaler |
| `visit_frequency` | Monthly visits | StandardScaler |
| `avg_order_value` | Average order value | StandardScaler |
| `tenure_months` | Customer tenure | StandardScaler |
| `engagement_score` | Engagement metric (0-100) | StandardScaler |

### Persona Naming Logic

| Criteria | Persona Name |
|----------|--------------|
| High spending + High income | Affluent Professionals |
| High spending + Young age | Young Enthusiasts |
| High tenure + High income | Loyal High-Value |
| Low spending | Budget Conscious |
| High visit frequency | Frequent Shoppers |
| Otherwise | Average Customers |

## Data Format

Input CSV format:

```csv
customer_id,age,income,spending_score,visit_frequency,avg_order_value,tenure_months,location_type,engagement_score
C0001,35,95000,72,12,85.50,36,urban,68
C0002,42,120000,45,6,120.00,48,suburban,52
```

**Required columns:**
- `customer_id`: Unique customer identifier
- `age`: Customer age (18-70)
- `income`: Annual income
- `spending_score`: Spending propensity (0-100)
- `visit_frequency`: Monthly visit count
- `avg_order_value`: Average order value
- `tenure_months`: Customer tenure in months
- `location_type`: Urban, suburban, or rural
- `engagement_score`: Engagement metric (0-100)

## Project Structure

```
segmentation/
├── cli.py              # Command-line interface
├── streamlit_app.py    # Web UI
├── catalog.py          # Public API library (to be created)
├── README.md           # This file
├── data/
│   └── synthetic/
│       └── customer_segmentation.csv
└── notebooks/
    └── customer_segmentation_example.ipynb
```

## CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `create-personas` | Create customer personas | `python cli.py create-personas --count 4` |
| `representative-customers` | Get typical customers per persona | `python cli.py representative-customers 0 --count 5` |
| `evaluate` | Evaluate clustering quality | `python cli.py evaluate` |

## License

See parent directory for license information.

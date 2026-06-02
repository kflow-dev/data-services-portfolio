# CHAP (Common Hybrid Agent Architecture)

Multi-agent socio-physical system simulation platform.

## Overview

CHAP simulates multi-agent systems with complex interactions between agents in a shared environment, useful for studying emergent behaviors and coordination.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CHAP                                   │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py simulate traffic --agents 100 --steps 60│
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import CHAP                         │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/agent_states.csv                      │
│  Model: Agent-based Simulation + Emergent Behavior          │
│  Features: agent_type, position, state, interactions        │
│  Output: Simulation logs with agent trajectories           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer matplotlib
```

## Usage

### CLI

```bash
# Simulate traffic flow
python cli.py simulate traffic --agents 100 --steps 60

# Run social interaction simulation
python cli.py simulate social --agents 50 --duration 100

# Analyze emergent patterns
python cli.py analyze --run-id run_001
```

### Streamlit UI

```bash
streamlit run apps/chap/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/chap/notebooks/agent_simulation.ipynb
```

## Project Structure

```
chap/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── agent_states.csv
└── notebooks/
    └── agent_simulation.ipynb
```

## License

See parent directory for license information.

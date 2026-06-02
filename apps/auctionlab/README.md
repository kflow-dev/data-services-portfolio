# Auction Lab

Real-time auction simulation and bidding optimization.

## Overview

Auction Lab simulates bidding auctions with optimization strategies for maximizing bid outcomes.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Auction Lab                             │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py simulate --items 100 --bidders 20      │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import AuctionLab                   │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/auction_history.csv                   │
│  Model: Bid Optimization + Game Theory                      │
│  Features: bid_history, item_value, bidder_strategy        │
│  Output: Optimal bidding strategies                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Run auction simulation
python cli.py simulate --items 100 --bidders 20 --rounds 50

# Analyze bidding patterns
python cli.py analyze --auction-id A001

# Generate optimal strategy
python cli.py strategy --item-type electronics
```

### Streamlit UI

```bash
streamlit run apps/auctionlab/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/auctionlab/notebooks/auction_simulation.ipynb
```

## Project Structure

```
auctionlab/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── auction_history.csv
└── notebooks/
    └── auction_simulation.ipynb
```

## License

See parent directory for license information.

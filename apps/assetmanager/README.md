# Asset Manager

Asset manager is a chatbot which maintains a dashboard of a set of assets (described in data/assets.csv) and estimates its value and impact of news and lifecycle events on its valuation over time, notifying the user when peaks and valleys are detected in its valuation; it takes a description of items of interest to manage (eg, financial, IT, industry/organizational stock assets, domestic hardware, etc) from data/assets.csv and scrapes a specified set of data sources from internet data sources specified about that specific asset category and asset, and monitors assets lifecycle events over time; from time to time, with specific frequencies for each class of assets, an event occurs (or is read from data/events.csv) and the chatbot maintains a database of asset lifecycle events/activities for that asset, and the valuation of that assets over time, including some value deprecation using a decay function, value increase due to positive news and/or specific events (repair, renew, maintenance, buy_new); its main function is a notification of periodic lifecycle events that need to occur and value depreciation based on outliers from a set of managed assets.

Asset manager takes a description of items of interest to manage (eg, financial, IT, industry/organizational stock assets, domestic hardware, etc) and scrapes a specified set of data sources from internet about that asset's meta-data discovery and lifecycle tracking, and retrieves from specific news and task/activities logs information about asset's lifecycle, including sentiment extracted from reviews, website comments, investor call analysis, and projects estimated value of the asset and impact of news on its projected value over time.

## Overview

Asset Manager discovers and tracks a large range of socio-physical and informational/virtual assets across infrastructure with lifecycle monitoring and compliance tracking.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Asset Manager                            │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py discover --target 192.168.1.0/24       │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import AssetManager                 │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/assets.csv                            │
│  Model: Discovery + Anomaly Detection                       │
│  Features: asset_type, age, compliance_status              │
│  Output: Asset inventory with lifecycle insights           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer
```

## Usage

### CLI

```bash
# Discover assets in network range
python cli.py discover --target 192.168.1.0/24

# Get asset lifecycle status
python cli.py lifecycle --asset-type server

# Check compliance
python cli.py compliance --asset-type workstation
```

### Streamlit UI

```bash
streamlit run apps/assetmanager/streamlit_app.py
```

### Jupyter Notebook

```bash
jupyter notebook apps/assetmanager/notebooks/asset_analysis.ipynb
```

## Project Structure

```
assetmanager/
├── cli.py
├── streamlit_app.py
├── README.md
├── data/synthetic/
│   └── assets.csv
└── notebooks/
    └── asset_analysis.ipynb
```

## License

See parent directory for license information.

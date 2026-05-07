# Data Science Services Projects

Version: v01: 2026-05-08 

List of AI/ML/causal data science projects implemented.

## 01: INSE

- (PORTALS) INSE/MyCallTechPhonebook - a personalized tech companies phonebook for scraping, discovery and arranging maintenance calls for hardware/IoT devices and domestic services

- (+) (PORTALS) INSE/RobotServicesCatalog - Robotic product catalogue - webshop of services for sales of robotics products (drones, cleaning robots) for small offices and individual homes

-  (++) (RECSYS) INSE/DieteRecommender - Medical regime, RL, graph

## 02: PAP

- (+++) FBML - PAP/Personal Assistant for Privacy - personalized search agent, NLP, RAG, smart proxies, rules learning, Tryllian SDK, threat prediction, multi-modal BDI logic, multi-modal recommendation, privacy


## 03: BASEWEB:

- (+) (PORTALS/Services/SecurityServices) BASEWEB/Web mining + User Profiling + Persona Builder + Stress/load testing for load times optimization and for IT 2001 certification

## 04: SCINLP:

- (++) SCINLP/ProtegeSemanticEngine - Ontology, reasoning, SPARQL, knowledge graphs/LLMs, Text2Model, Text2Onto, Karma.

## 05: SCIKRR:

- (++) SCIKRR/FluentLeadsToAdaptiveSimulation - Simulation, trace analysis, temporal logic (causal/temporal modeling)

## 06: EASOA:

- (+) EASOA/KnowledgeHub/RedBooks live books and application development frameworks

## 07: NETSYM: 

- (++) NETSYM/BitcoinDetective - Blockchain analysis, anomaly detection, price shock, ML/stats/causal

## 08: SCISIM:

- (+++) CHAP - Common Hybrid Agent Architecture, multi-agent socio-physical system simulation

## 09: DSS:

- (++) DSS/VideoAnalyticsSummarizer - RL, federated learning, video analytics, CNN/DL/RL/MLOps

- (+++) AECR - Advisor for Energy Consumption Reduction, RecSys

## 10: LPLAN:

- (++) LPLAN/Mobile_AB_Test - A/B testing of features and stress testing platform across devices

## 11: ADMARCOM:

- (++) ADMARCOM/AdRetargeterBloomFilter - Ad targeting, bloom filters, real-time return customer classification

## 12: SYSTEMA:

- (+) Wherescape dashboard: Self-service BI + connectors Facebook API for visualization of social media contacts in LinkedIn and Facebook
 
## 13: ACN:

- (+++) Agrotech Customer Segmentation & Persona Creation reusable asset, incl. stratified sampling, representative customer selection for interviews

- (+++) Personalized beer product SKU recommender for B2B, including geo-location shop scraping in Google Place API

- (++) Citizen Data Lab: managed cloud data science lab setup for Gemeente Bold

- (++) MyMedicine - mobile app for travelling abroad with medicine (IBM Watson + BeInformed + IBM cloud APIs + H100 GPU training)

## 14: HOTELINNO:
    
- (++) HOTELINNO/AirbnbPricePredictor - HMM, hierarchical forecasting, causal relations, outlier detection, stats/causal/DL

## 15: FASHIONTECH:

- (++) FASHIONTECH/CLTVOptimizer - CLTV prediction, survival analysis, mixed-integer programming, Bayesian A/B testing, causal bandits, stats/causal/ML

- (++) multi-step hierarchical department-product group-product-color-size demand forecasting (LSTM, DeepAR)

- (+) Sweet spot detection for frequency & timing of newsletters - experiment with multiple fashion organizations



## 16: KFLOW:

- (+++) KFLOW/AIFluentCareerPath - Skill graphs, RL recommendation, causal path analysis, ML/architecture Skills platform - knowledge acquisition plan (a la Netflix) + gamification + dynamic rewards monetization

- (+) KFLOW/WorkforcePulse - HR Analytics Hub transitioning from Reactive Reporting to Predictive Talent Management


## Project Structure Overview:

```
ai-portfolio-project/
├── .github/                       # CI/CD workflows, templates, issue configs
│   └── workflows/
├── .vscode/                       # Dev container + Jupyter config
├── scripts/                       # Utility scripts: data generation, visualization, model deployment
├── docs/                          # Architecture diagrams, project overviews, diagrams (Mermaid, PlantUML)
│   ├── flowcharts/
│   └── architecture/
├── projects/                      # << MAIN FOCUS: Portfolio Projects >>
│   ├── [ID]_[Name]/
│   │   ├── README.md              # Project description, roadmap, milestones, tasks, deliverables
│   │   ├── data/                  # Dummy/raw/synthetic data (per dataset)
│   │   │   └── raw/               # e.g., `call_logs.csv`, `rewards.csv`, `prices.tsv`
│   │   ├── notebooks/             # Jupyter/Quarto notebooks for EDA, modeling, experiments
│   │   ├── src/                   # Python/Scala/SQL/Java modules (modularized)
│   │   │   ├── data/              # ingestion, validation, transforms
│   │   │   ├── models/            # ML/DL/causal models
│   │   │   ├── mlops/             # MLflow, Optuna, FastAPI, CI/CD hooks
│   │   │   └── utils/
│   │   ├── sql/                   # Pre/post-processing queries (Redshift/BigQuery/Postgres)
│   │   ├── diagrams/              # Architecture sketches (SVG/PNG/Mermaid)
│   │   ├── reports/               # executive summaries, dashboard links (e.g., Superset/Tableau)
│   │   └── requirements.txt       # Conda env + pinned packages
│   │
│   ├── 00_INDEX.md               # Master README of all featured projects
│   └── roadmap_template.md       # Reusable template for future projects
│
├── templates/
│   ├── project_template/
│   │   └── README.md
│   └── notebook_template.ipynb
│
├── architecture/
│   ├── data_lakehouse.txt        # Logical architecture (Medallion, Delta, etc.)
│   └── mlops_stack.yaml          # Standard stack: DVC, MLflow, Prefect, Kubeflow, etc.
│
├── CONTRIBUTING.md
├── LICENSE
└── README.md                     # This is the main entry point for recruiters & collaborators
```

# END

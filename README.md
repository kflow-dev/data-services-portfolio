# Data Science Services Projects

Version: v02: 2026-05-28 / v01: 2026-05-08 

List of AI/ML/GenAI/IR/RL/causal/bayesian data science projects implemented.

## 01: INSE

-  (++) (RECSYS) INSE/HealthyDietRecommender - Personalized medical regime expert system and healthy diet recommender, calory counter, health objectives monitoring, RL, graph

- (+) (PORTALS) INSE/RobotServicesCatalog - Robotic product catalogue - webshop of services for sales of robotics products (drones, cleaning robots) for small offices and individual homes

- (+) (PORTALS) INSE/MyCallTechPhonebook - a personalized tech companies phonebook for scraping, discovery and arranging maintenance calls for hardware/IoT devices and domestic services


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
└── README.md                     # This is the main entry po.
```

## DETAILS

A full RAG + ML product portfolio: ~25 data-science apps spanning recommenders, search&scrape,
chatbots, forecasters, agents and optimizers — wrapped in a mobile-friendly portfolio
dashboard with a self-managed VPS CI/CD pipeline.

## Architecture

```text
                       Internet (mobile / desktop)
                                  |
                                  v
                  +------------------------------+
                  |  nginx  (TLS, gzip, routing) |
                  +--------------+---------------+
         +------------------------+--------------------------+
         v                        v                          v
   /  (dashboard)           /app/<slug>/             /api/  (edge)
   Next.js    		   Streamlit per-app          Cloud
   :7000 (frontend)        :3000 (backend)          Edge Functions
                                                  + Postgres + pgvector
```

- Frontend  : React (TanStack Start) on **:7000**
- Backend   : Streamlit Python apps on **:3000**
- Reverse proxy: nginx on **:80 / :443** (see `nginx/portfolio.conf`)
- Data      : Postgres + pgvector (running in Cloud) for RAG
- AI        : AI Gateway (`google/gemini-2.5-flash`, `openai/text-embedding-3-small`)

## Live demos in the dashboard

| App | What it does | Stack |
| --- | --- | --- |
| HotNews4U       | LLM-ranked news recommender              | Gemini Flash JSON-mode |
| Multi-media RAG | Ingest text -> embed -> grounded Q&A     | pgvector + embeddings + LLM |
| PAPIE           | Streaming personal-assistant chatbot     | SSE chat completions |

The other apps are listed in the dashboard as scaffolds — each has a folder under
`apps/` with a CLI entry point, a Streamlit UI, and a Dockerfile.

## Repo layout

```text
.
├── src/                       # TanStack Start frontend (dashboard + demos)
├── supabase/functions/        # Edge functions
├── supabase/migrations/       # pgvector schema + match function
├── apps/                      # ~25 Streamlit/CLI apps
├── nginx/portfolio.conf       # reverse proxy
├── docker-compose.yml         # local + VPS stack
├── .github/workflows/         # ci.yml + deploy.yml
└── README.md
```

## Run locally

```bash
bun install && bun run dev -- --port 7000
python apps/hotnews4u/cli.py recommend --interests "AI, startups"
streamlit run apps/hotnews4u/streamlit_app.py --server.port 3000
docker compose up --build   # full stack behind nginx
```

## Deploy to a self-managed VPS (Ubuntu 22.04)

1. Provision the VPS, install Docker + Compose, open 80/443.
2. Add GitHub secrets: `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`, `LOVABLE_API_KEY`,
   `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_PROJECT_ID`.
3. Push to `main` → `.github/workflows/deploy.yml` rsyncs the repo and runs
   `docker compose up -d --build` on the VPS.
4. Point DNS at the VPS, then `certbot --nginx` for TLS.

# END

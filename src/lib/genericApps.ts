export type GenericSpec = {
  slug: string;
  name: string;
  tagline: string;
  inputLabel: string;
  defaultInput: string;
  rows?: number;
};

export const GENERIC_APPS: Record<string, GenericSpec> = {
  "drift-monitor": {
    slug: "drift-monitor",
    name: "Drift Monitor",
    tagline: "Model & data drift monitoring asset.",
    inputLabel: "Describe your model & recent input/output stats",
    defaultInput:
      "Credit-risk XGBoost model, trained 2024-01. Last 7d: avg score 0.31 vs baseline 0.22; feature 'income' mean shifted +18%; missing-rate on 'employer' jumped from 2% to 11%.",
    rows: 5,
  },
  "datalab-aas": {
    slug: "datalab-aas",
    name: "DataLab-as-a-Service",
    tagline: "Jupyter workbench planner for DS teams.",
    inputLabel: "Describe your team & workloads",
    defaultInput:
      "12 data scientists, mix of LLM fine-tuning and tabular ML. Need shared GPU pool, secure access to S3 + Snowflake, reproducible envs.",
    rows: 4,
  },
  segmentation: {
    slug: "segmentation",
    name: "Customer Segmentation",
    tagline: "Representative customers & persona creation.",
    inputLabel: "Describe your customer base",
    defaultInput:
      "DTC sneaker brand, 80k monthly active users, age 16-45, mix of hype buyers, casual replenishers, and resellers. Avg basket 120 EUR.",
    rows: 4,
  },
  jobminder: {
    slug: "jobminder",
    name: "JobMinder",
    tagline: "Job recommender chatbot — what should I do next?",
    inputLabel: "Your situation",
    defaultInput:
      "Senior backend engineer, 8 yrs Python, switching into ML platform / LLMOps. Based in Lisbon, open to remote EU.",
    rows: 4,
  },
  "sku-forecast": {
    slug: "sku-forecast",
    name: "SKU Demand Forecaster",
    tagline: "Foundation models (TimesFM, Chronos) for SKU demand.",
    inputLabel: "Describe the SKU & recent demand",
    defaultInput:
      "SKU: organic oat milk 1L. Last 8 weeks units: 320, 340, 360, 410, 480, 520, 560, 610. Seasonality: spring uplift, no promo.",
    rows: 4,
  },
  aifluent: {
    slug: "aifluent",
    name: "AIFluent",
    tagline: "Skills acquisition platform.",
    inputLabel: "What do you want to learn?",
    defaultInput: "Become production-ready in Retrieval-Augmented Generation with pgvector in 4 weeks, ~6h/week.",
    rows: 3,
  },
  chap: {
    slug: "chap",
    name: "CHAP",
    tagline: "Common Hybrid Agent Platform — orchestrate sub-agents.",
    inputLabel: "Task to orchestrate",
    defaultInput:
      "Draft a competitive brief on the top 3 open-source vector DBs: features, pricing, community, with citations.",
    rows: 4,
  },
  auctionlab: {
    slug: "auctionlab",
    name: "AuctionLab",
    tagline: "Auction experimentation platform.",
    inputLabel: "Auction setup",
    defaultInput:
      "Second-price sealed-bid auction, 5 bidders with valuations ~N(100, 20). Reserve price 60. Simulate 3 rounds and discuss strategy.",
    rows: 4,
  },
  emagazzine: {
    slug: "emagazzine",
    name: "EMagazzine",
    tagline: "Price comparator & multi-objective product tracker.",
    inputLabel: "Product + objective weights",
    defaultInput:
      "Looking for: Sony WH-1000XM5 headphones. Weights: price 0.5, delivery speed 0.3, seller trust 0.2. Ship to PT.",
    rows: 4,
  },
  skillsplan: {
    slug: "skillsplan",
    name: "SkillsPlan",
    tagline: "Curriculum builder optimizing cost / time / knowledge impact.",
    inputLabel: "Target role / skill",
    defaultInput:
      "Become a Staff MLOps engineer in 6 months. Budget 1500 EUR, 8h/week. Already strong in Python & Docker.",
    rows: 4,
  },
  mysmartdiet: {
    slug: "mysmartdiet",
    name: "MySmartDiet",
    tagline: "Healthy diet recommender.",
    inputLabel: "Goals, restrictions, preferences",
    defaultInput:
      "Goal: lose 4kg in 8 weeks. 80kg, 178cm, male, runs 3x/week. Lactose-intolerant, dislikes mushrooms. Mediterranean cuisine.",
    rows: 4,
  },
  "cloud-ml-estimator": {
    slug: "cloud-ml-estimator",
    name: "Cloud ML Estimator",
    tagline: "Cloud ML solution & pricing estimator.",
    inputLabel: "Workload description",
    defaultInput:
      "Serve a 7B LLM at ~50 req/s, p95 < 1.5s, EU region, batch fine-tuning weekly on 20GB data. Compare AWS / GCP / Azure.",
    rows: 4,
  },
  // Recommenders
  mywardrobe: {
    slug: "mywardrobe",
    name: "MyWardrobe",
    tagline: "Content-based outfit recommender using cosine similarity on style/season/price.",
    inputLabel: "Context: style, season, budget (e.g., 'smart-casual summer budget 150 EUR')",
    defaultInput: "Smart-casual weekend brunch, neutral palette, budget ~150 EUR",
  },
  cooldrinks: {
    slug: "cooldrinks",
    name: "CoolDrinks",
    tagline: "ALS-based beer SKU recommender for retailers (B2B).",
    inputLabel: "Preferred styles / flavors / ABV",
    defaultInput: "Hoppy IPAs and citrus-forward pales, ABV 5-7%",
  },
  mynexthome: {
    slug: "mynexthome",
    name: "MyNextHome",
    tagline: "Hybrid real-estate recommender with geo + content features.",
    inputLabel: "City, budget, must-haves",
    defaultInput: "Lisbon, up to 450k EUR, 2 bedrooms, near metro, balcony",
  },
  mymedicine: {
    slug: "mymedicine",
    name: "MyMedicine",
    tagline: "Content-based + rule-based travel medicine recommender.",
    inputLabel: "Symptoms / destination / allergies",
    defaultInput: "Stomach upset and mild fever, traveling in Thailand, no allergies",
  },
  ebooks: {
    slug: "ebooks",
    name: "E-book & Audiobook RecSys",
    tagline: "Content-based book recommender using Sentence-BERT embeddings.",
    inputLabel: "Genres / authors / mood",
    defaultInput: "Hard sci-fi and systems thinking, fan of Ted Chiang and Neal Stephenson",
  },
  scitubbies: {
    slug: "scitubbies",
    name: "SciTubbies",
    tagline: "Collaborative filtering for YouTube science video recommendations.",
    inputLabel: "Topics you enjoy",
    defaultInput: "Math intuition, ML research papers, physics explainers",
  },
  jobpromis: {
    slug: "jobpromis",
    name: "JobPromis",
    tagline: "Hybrid job matching with BM25 + semantic reranking.",
    inputLabel: "Skills, role, location, comp",
    defaultInput: "Senior ML engineer, Python+PyTorch+LLMs, remote EU, 90-130k EUR",
  },
  // Search Apps
  assetmanager: {
    slug: "assetmanager",
    name: "AssetManager",
    tagline: "Text search + LLM summarization for articles and documents.",
    inputLabel: "Article text or transcript to process",
    defaultInput:
      "On March 12, OpenAI and Microsoft announced an extended partnership in Redmond focused on Azure-hosted GPT models. CEO Satya Nadella said the deal would accelerate enterprise rollout, while Sam Altman highlighted new safety commitments.",
    rows: 6,
  },
  socialtraces: {
    slug: "socialtraces",
    name: "SocialTraces",
    tagline: "Social-media fuzzy search detective. Find a person across handles, typos, aliases.",
    inputLabel: "Who are you looking for?",
    defaultInput: "Maria Silva, Portuguese ML engineer, into trail running and roastery coffee",
  },
  mylocalradar: {
    slug: "mylocalradar",
    name: "MyLocalRadar",
    tagline: "Location mapping & disambiguation. Which Cambridge did you mean?",
    inputLabel: "Place query",
    defaultInput: "I'm moving to Cambridge next month for a postdoc.",
  },
  "declarative-search": {
    slug: "declarative-search",
    name: "Declarative Search",
    tagline: "Multi-agent declarative search: state what you want, the agent plans & answers.",
    inputLabel: "What do you want to know?",
    defaultInput:
      "Compare TimesFM and Chronos for retail SKU demand forecasting — accuracy, cost, and latency tradeoffs.",
  },
};
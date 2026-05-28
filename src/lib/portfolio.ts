export type AppStatus = "live" | "scaffold" | "planned";

export type PortfolioApp = {
  slug: string;
  name: string;
  tagline: string;
  category: string;
  stack: string[];
  status: AppStatus;
  route?: string;
};

export const CATEGORIES = [
  "Recommenders",
  "Data Products",
  "Chatbots",
  "Forecasters",
  "Search & Info Retrieval",
  "Assistants / Agent Platforms",
  "Comparators / Simulators / Optimizers",
  "Calculators / Estimators",
] as const;

export const APPS: PortfolioApp[] = [
  // Recommenders
  {
    slug: "hotnews4u",
    name: "HotNews4U",
    tagline: "Personalized news & newsletter recommender",
    category: "Recommenders",
    stack: ["LLM ranking", "Gemini Flash", "Edge Function"],
    status: "live",
    route: "/apps/hotnews4u",
  },
  { slug: "mywardrobe", name: "MyWardrobe", tagline: "ShopTheLook outfit recommender (B2B)", category: "Recommenders", stack: ["CV embeddings", "FAISS", "Streamlit"], status: "scaffold" },
  { slug: "cooldrinks", name: "CoolDrinks", tagline: "Beer SKU recommender (B2B)", category: "Recommenders", stack: ["ALS", "LightFM"], status: "scaffold" },
  { slug: "mynexthome", name: "MyNextHome", tagline: "Real-estate object recommender", category: "Recommenders", stack: ["Hybrid CF", "Geo features"], status: "scaffold" },
  { slug: "mymedicine", name: "MyMedicine", tagline: "Travel-abroad medicine lookup (IBM Watson + Vision)", category: "Recommenders", stack: ["Watson NLU", "Vision API"], status: "scaffold" },
  { slug: "ebooks", name: "E-book & Audiobook RecSys", tagline: "Content-based book recommender", category: "Recommenders", stack: ["TF-IDF", "Sentence-BERT"], status: "scaffold" },
  { slug: "scitubbies", name: "SciTubbies", tagline: "YouTube playlist & video content RecSys", category: "Recommenders", stack: ["YT API", "Topic model"], status: "scaffold" },
  { slug: "jobpromis", name: "JobPromis", tagline: "Job recommender app", category: "Recommenders", stack: ["BM25", "Cross-encoder rerank"], status: "scaffold" },

  // Data Products
  { slug: "drift-monitor", name: "Drift Monitor", tagline: "Model & data drift monitoring asset", category: "Data Products", stack: ["Evidently", "Prometheus"], status: "scaffold" },
  { slug: "datalab-aas", name: "DataLab-as-a-Service", tagline: "Jupyter workbench for DS teams", category: "Data Products", stack: ["JupyterHub", "K8s"], status: "scaffold" },
  { slug: "segmentation", name: "Customer Segmentation", tagline: "Representative customers & persona creation", category: "Data Products", stack: ["KMeans", "UMAP"], status: "scaffold" },

  // Chatbots
  {
    slug: "papie",
    name: "PAPIE",
    tagline: "Personal Assistant for Personal Info Exchange",
    category: "Chatbots",
    stack: ["Streaming LLM", "Gemini Flash"],
    status: "live",
    route: "/apps/papie",
  },
  { slug: "jobminder", name: "JobMinder", tagline: "Job recommender chatbot", category: "Chatbots", stack: ["Agentic flow", "Tools"], status: "scaffold" },

  // Forecasters
  { slug: "sku-forecast", name: "SKU Demand Forecaster", tagline: "Foundation models for product SKU demand", category: "Forecasters", stack: ["TimesFM", "Chronos"], status: "scaffold" },

  // Search & Info Retrieval
  { slug: "declarative-search", name: "Declarative Search", tagline: "Multi-agent search & scrape", category: "Search & Info Retrieval", stack: ["LangGraph", "Playwright"], status: "scaffold" },
  {
    slug: "rag",
    name: "Multi-media RAG",
    tagline: "Product discovery, NER, search & Q&A pipeline",
    category: "Search & Info Retrieval",
    stack: ["pgvector", "text-embedding-3-small", "Gemini"],
    status: "live",
    route: "/apps/rag",
  },
  { slug: "socialtraces", name: "SocialTraces", tagline: "Social media fuzzy search detective", category: "Search & Info Retrieval", stack: ["Fuzzy match", "Graph"], status: "scaffold" },
  { slug: "assetmanager", name: "AssetManager", tagline: "Multi-media article summarizer + NER", category: "Search & Info Retrieval", stack: ["spaCy", "LLM summarize"], status: "scaffold" },
  { slug: "mylocalradar", name: "MyLocalRadar", tagline: "Location mapping & disambiguation", category: "Search & Info Retrieval", stack: ["Geo NER", "OSM"], status: "scaffold" },

  // Assistants / Agent Platforms
  { slug: "aifluent", name: "AIFluent", tagline: "Skills acquisition platform", category: "Assistants / Agent Platforms", stack: ["Agentic tutor"], status: "scaffold" },
  { slug: "chap", name: "CHAP", tagline: "Common hybrid agent platform for knowledge exchange", category: "Assistants / Agent Platforms", stack: ["Multi-agent"], status: "scaffold" },

  // Comparators / Simulators / Optimizers
  { slug: "auctionlab", name: "AuctionLab", tagline: "Auction experimentation platform", category: "Comparators / Simulators / Optimizers", stack: ["Sim engine"], status: "scaffold" },
  { slug: "emagazzine", name: "EMagazzine", tagline: "Price comparator & multi-objective product tracker", category: "Comparators / Simulators / Optimizers", stack: ["Scrapy", "Pareto opt"], status: "scaffold" },
  { slug: "skillsplan", name: "SkillsPlan", tagline: "Curriculum builder optimizing cost/time/impact", category: "Comparators / Simulators / Optimizers", stack: ["MILP", "OR-Tools"], status: "scaffold" },
  { slug: "mysmartdiet", name: "MySmartDiet", tagline: "Healthy diet recommender", category: "Comparators / Simulators / Optimizers", stack: ["Constraint solver"], status: "scaffold" },

  // Calculators / Estimators
  { slug: "cloud-ml-estimator", name: "Cloud ML Estimator", tagline: "Cloud ML solution & pricing estimator", category: "Calculators / Estimators", stack: ["Pricing API", "Heuristics"], status: "scaffold" },
];

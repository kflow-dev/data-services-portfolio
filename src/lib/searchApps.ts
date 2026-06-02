export type SearchMode = "declarative" | "socialtraces" | "assetmanager" | "localradar";

export type SearchSpec = {
  slug: string;
  name: string;
  tagline: string;
  mode: SearchMode;
  queryLabel: string;
  defaultQuery: string;
  corpusLabel?: string;
  defaultCorpus?: string;
};

export const SEARCH_APPS: Record<string, SearchSpec> = {
  "declarative-search": {
    slug: "declarative-search",
    name: "Declarative Search",
    tagline: "Multi-agent declarative search: state what you want, the agent plans & answers.",
    mode: "declarative",
    queryLabel: "What do you want to know?",
    defaultQuery:
      "Compare TimesFM and Chronos for retail SKU demand forecasting — accuracy, cost, and latency tradeoffs.",
  },
  socialtraces: {
    slug: "socialtraces",
    name: "SocialTraces",
    tagline: "Social-media fuzzy search detective. Find a person across handles, typos, aliases.",
    mode: "socialtraces",
    queryLabel: "Who are you looking for?",
    defaultQuery: "Maria Silva, Portuguese ML engineer, into trail running and roastery coffee",
    corpusLabel: "Profile corpus (one per line)",
    defaultCorpus: [
      "@m_silva_ml — ML engineer @ Lisbon, runs ultras, espresso snob",
      "@maria.s.codes — backend dev, Porto, surfing",
      "@silva_maria — data scientist, Madrid, road cycling",
      "@msilva.run — trail runner, Coimbra, third-wave coffee",
      "@marie_silvr — designer, Paris, latte art",
    ].join("\n"),
  },
  assetmanager: {
    slug: "assetmanager",
    name: "AssetManager",
    tagline: "Multi-media article summarizer with named-entity extraction.",
    mode: "assetmanager",
    queryLabel: "Paste article / transcript",
    defaultQuery:
      "On March 12, OpenAI and Microsoft announced an extended partnership in Redmond focused on Azure-hosted GPT models. CEO Satya Nadella said the deal would accelerate enterprise rollout, while Sam Altman highlighted new safety commitments.",
  },
  mylocalradar: {
    slug: "mylocalradar",
    name: "MyLocalRadar",
    tagline: "Location mapping & disambiguation — which Cambridge did you mean?",
    mode: "localradar",
    queryLabel: "Place query",
    defaultQuery: "I'm moving to Cambridge next month for a postdoc.",
  },
};
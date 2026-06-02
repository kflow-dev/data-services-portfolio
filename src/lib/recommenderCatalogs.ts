export type CatalogItem = { id: string; title: string; meta?: string; [k: string]: unknown };

export type RecommenderSpec = {
  slug: string;
  name: string;
  domain: string;
  tagline: string;
  promptLabel: string;
  defaultContext: string;
  items: CatalogItem[];
};

export const RECOMMENDERS: Record<string, RecommenderSpec> = {
  mywardrobe: {
    slug: "mywardrobe",
    name: "MyWardrobe",
    domain: "fashion outfits",
    tagline: "Content-based outfit recommender using cosine similarity.",
    promptLabel: "Occasion / style / budget",
    defaultContext: "Smart-casual weekend brunch, neutral palette, budget ~150 EUR",
    items: [
      { id: "w1", title: "Linen blazer + white tee + tapered chinos", meta: "smart-casual · 180 EUR" },
      { id: "w2", title: "Oversized hoodie + cargo pants + chunky sneakers", meta: "streetwear · 140 EUR" },
      { id: "w3", title: "Wool overcoat + turtleneck + dark denim", meta: "winter smart · 240 EUR" },
      { id: "w4", title: "Floral midi dress + leather sandals", meta: "summer · 120 EUR" },
      { id: "w5", title: "Silk shirt + tailored trousers + loafers", meta: "office · 210 EUR" },
      { id: "w6", title: "Denim jacket + graphic tee + black jeans", meta: "casual · 95 EUR" },
      { id: "w7", title: "Linen jumpsuit + espadrilles", meta: "vacation · 130 EUR" },
      { id: "w8", title: "Leather biker jacket + slim jeans + boots", meta: "edgy · 280 EUR" },
    ],
  },
  cooldrinks: {
    slug: "cooldrinks",
    name: "CoolDrinks",
    domain: "craft beer SKUs",
    tagline: "ALS-based beer SKU recommender for retailers (B2B).",
    promptLabel: "Preferred styles / flavors / ABV",
    defaultContext: "Hoppy IPAs and citrus-forward pales, ABV 5-7%",
    items: [
      { id: "b1", title: "Hazy NEIPA - Citra+Mosaic", meta: "6.5% · tropical" },
      { id: "b2", title: "West Coast IPA", meta: "7.0% · pine, grapefruit" },
      { id: "b3", title: "Belgian Tripel", meta: "9.0% · spicy, fruity" },
      { id: "b4", title: "Czech Pilsner", meta: "4.8% · crisp, floral" },
      { id: "b5", title: "Imperial Stout", meta: "10% · coffee, chocolate" },
      { id: "b6", title: "Berliner Weisse - Raspberry", meta: "3.5% · tart" },
      { id: "b7", title: "Session Pale Ale", meta: "4.2% · citrus" },
      { id: "b8", title: "Barrel-Aged Saison", meta: "6.8% · funky, oak" },
    ],
  },
  mynexthome: {
    slug: "mynexthome",
    name: "MyNextHome",
    domain: "real-estate listings",
    tagline: "Hybrid real-estate recommender with geo + content features.",
    promptLabel: "City, budget, must-haves",
    defaultContext: "Lisbon, up to 450k EUR, 2 bedrooms, near metro, balcony",
    items: [
      { id: "h1", title: "2BR apt in Alfama, 78m², balcony", meta: "Lisbon · 410k EUR" },
      { id: "h2", title: "3BR house in Sintra, 140m², garden", meta: "Sintra · 520k EUR" },
      { id: "h3", title: "Studio in Chiado, 42m², top floor", meta: "Lisbon · 290k EUR" },
      { id: "h4", title: "2BR loft in Marvila, 95m², parking", meta: "Lisbon · 445k EUR" },
      { id: "h5", title: "4BR villa in Cascais, 220m², pool", meta: "Cascais · 1.2M EUR" },
      { id: "h6", title: "2BR apt in Anjos, 70m², metro 2min", meta: "Lisbon · 380k EUR" },
      { id: "h7", title: "1BR apt in Porto, 55m², river view", meta: "Porto · 260k EUR" },
      { id: "h8", title: "3BR penthouse Parque das Nações", meta: "Lisbon · 680k EUR" },
    ],
  },
  mymedicine: {
    slug: "mymedicine",
    name: "MyMedicine",
    domain: "travel OTC medicine",
    tagline: "Content-based + rule-based travel medicine lookup.",
    promptLabel: "Symptoms / destination / allergies",
    defaultContext: "Stomach upset and mild fever, traveling in Thailand, no allergies",
    items: [
      { id: "m1", title: "Loperamide 2mg", meta: "anti-diarrheal" },
      { id: "m2", title: "Paracetamol 500mg", meta: "fever, pain" },
      { id: "m3", title: "Ibuprofen 400mg", meta: "anti-inflammatory" },
      { id: "m4", title: "Oral rehydration salts (ORS)", meta: "dehydration" },
      { id: "m5", title: "Bismuth subsalicylate", meta: "stomach upset" },
      { id: "m6", title: "Cetirizine 10mg", meta: "antihistamine" },
      { id: "m7", title: "DEET 30% spray", meta: "insect repellent" },
      { id: "m8", title: "Azithromycin (Rx)", meta: "bacterial infection" },
    ],
  },
  ebooks: {
    slug: "ebooks",
    name: "E-book & Audiobook RecSys",
    domain: "books and audiobooks",
    tagline: "Content-based book recommender using Sentence-BERT.",
    promptLabel: "Genres / authors / mood",
    defaultContext: "Hard sci-fi and systems thinking, fan of Ted Chiang and Neal Stephenson",
    items: [
      { id: "k1", title: "Project Hail Mary - Andy Weir", meta: "hard sci-fi" },
      { id: "k2", title: "Exhalation - Ted Chiang", meta: "speculative short stories" },
      { id: "k3", title: "Anathem - Neal Stephenson", meta: "philosophical sci-fi" },
      { id: "k4", title: "The Three-Body Problem - Liu Cixin", meta: "hard sci-fi" },
      { id: "k5", title: "Thinking in Systems - D. Meadows", meta: "systems thinking" },
      { id: "k6", title: "Sapiens - Y.N. Harari", meta: "history, big ideas" },
      { id: "k7", title: "The Name of the Wind - Rothfuss", meta: "fantasy" },
      { id: "k8", title: "Designing Data-Intensive Apps - Kleppmann", meta: "tech" },
    ],
  },
  scitubbies: {
    slug: "scitubbies",
    name: "SciTubbies",
    domain: "YouTube science channels",
    tagline: "Collaborative filtering for YouTube science video content.",
    promptLabel: "Topics you enjoy",
    defaultContext: "Math intuition, ML research papers, physics explainers",
    items: [
      { id: "y1", title: "3Blue1Brown", meta: "math, intuition" },
      { id: "y2", title: "Two Minute Papers", meta: "AI research" },
      { id: "y3", title: "Yannic Kilcher", meta: "ML paper reviews" },
      { id: "y4", title: "Veritasium", meta: "physics, eng" },
      { id: "y5", title: "Computerphile", meta: "CS" },
      { id: "y6", title: "Numberphile", meta: "math" },
      { id: "y7", title: "Sabine Hossenfelder", meta: "physics critique" },
      { id: "y8", title: "Lex Fridman Podcast", meta: "long-form AI" },
    ],
  },
  jobpromis: {
    slug: "jobpromis",
    name: "JobPromis",
    domain: "job postings",
    tagline: "Hybrid job matching with BM25 + semantic reranking.",
    promptLabel: "Skills, role, location, comp",
    defaultContext: "Senior ML engineer, Python+PyTorch+LLMs, remote EU, 90-130k EUR",
    items: [
      { id: "j1", title: "Senior ML Engineer @ Mistral", meta: "Paris/remote · 110-140k" },
      { id: "j2", title: "LLM Researcher @ DeepL", meta: "Berlin · 100-130k" },
      { id: "j3", title: "MLOps Lead @ Klarna", meta: "Stockholm/remote · 120k" },
      { id: "j4", title: "Data Scientist @ Spotify", meta: "remote EU · 90-110k" },
      { id: "j5", title: "AI Solutions Architect @ AWS", meta: "Madrid · 130k" },
      { id: "j6", title: "Backend Engineer (Go) @ Datadog", meta: "remote · 100k" },
      { id: "j7", title: "Founding ML Eng @ early-stage AI startup", meta: "remote · equity+85k" },
      { id: "j8", title: "Computer Vision Eng @ Wayve", meta: "London · 110k" },
    ],
  },
};

export const RECOMMENDER_SLUGS = Object.keys(RECOMMENDERS);

# HotNews4U

LLM-ranked news recommender using Gemini Flash.

## Overview

HotNews4U personalizes news article recommendations based on user interests using LLM ranking. It processes articles and ranks them by relevance to user-specified topics.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HotNews4U                               │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py recommend --interests "AI, startups"    │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import HotNews4U                    │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/news_articles.csv                     │
│  Model: Google Gemini Flash (JSON mode)                     │
│  Features: title, category, sentiment_score                 │
│  Output: Ranked articles with relevance scores             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit pandas numpy typer requests
```

## Usage

### CLI

```bash
# Recommend articles for specific interests
python cli.py recommend --interests "AI, startups"

# Recommend from custom data directory
python cli.py recommend --interests "finance, crypto" --data-dir data/custom/

# View available categories
python cli.py list-categories
```

### Streamlit UI

```bash
# From project root
streamlit run apps/hotnews4u/streamlit_app.py

# With custom port
streamlit run apps/hotnews4u/streamlit_app.py --server.port 8501
```

### Jupyter Notebook

```bash
# From project root
jupyter notebook apps/hotnews4u/notebooks/news_recommendation_example.ipynb

# Or start jupyter and navigate to:
# apps/hotnews4u/notebooks/news_recommendation_example.ipynb
```

### As a Library

```python
import os
import json
import requests
import pandas as pd

# Set API key
os.environ["LOVABLE_API_KEY"] = "your-api-key"

# Load articles
articles = pd.read_csv("data/synthetic/news_articles.csv")

# Prepare prompt
interests = "AI, machine learning"
prompt = (
    f"User interests: {interests}\n\n"
    f"Articles: {articles.to_json(orient='records')}\n\n"
    'Rank top 5 as JSON: {{"ranked":[{{"id":"...","score":0-1,"reason":"..."}}]}}'
)

# Call LLM
response = requests.post(
    "https://ai.gateway.lovable.dev/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['LOVABLE_API_KEY']}",
        "Content-Type": "application/json"
    },
    json={
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    },
    timeout=60
)

# Parse results
results = response.json()["choices"][0]["message"]["content"]
ranked = json.loads(results)
print(ranked)
```

## Model Details

### Algorithm

**LLM-based Ranking with Gemini Flash**:
- Uses Google's Gemini Flash model for relevance scoring
- JSON mode output for structured results
- Prompt engineering for consistent ranking
- Real-time inference via API

### Features Used

| Feature | Description |
|---------|-------------|
| `title` | Article headline |
| `category` | News category (tech, ai, finance, etc.) |
| `sentiment_score` | Sentiment analysis (0-1) |
| `engagement_score` | Historical engagement (0-10) |
| `published_date` | Publication date |

### Output Format

```json
{
  "ranked": [
    {
      "id": "A003",
      "score": 0.95,
      "reason": "Highly relevant to AI interests - discusses transformer models"
    },
    ...
  ]
}
```

## Data Format

Input CSV format:

```csv
article_id,title,category,published_date,sentiment_score,engagement_score,source,read_time,word_count,url
A001,"Apple announces M5 chip with on-device LLM",tech,2026-05-15,0.75,8.5,techcrunch,4,1250,https://...
A002,"ECB cuts interest rates by 25bps",finance,2026-05-14,0.30,6.2,reuters,5,1400,https://...
```

**Required columns:**
- `article_id`: Unique article identifier
- `title`: Article headline
- `category`: News category
- `published_date`: Publication date (YYYY-MM-DD)

**Optional columns:**
- `sentiment_score`: Sentiment (0-1)
- `engagement_score`: Engagement metric (0-10)
- `source`: News source
- `url`: Article URL

## Project Structure

```
hotnews4u/
├── cli.py              # Command-line interface
├── streamlit_app.py    # Web UI
├── catalog.py          # Public API library (to be created)
├── README.md           # This file
├── data/
│   └── synthetic/
│       └── news_articles.csv
└── notebooks/
    └── news_recommendation_example.ipynb
```

## CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `recommend` | Recommend articles by interests | `python cli.py recommend --interests "AI, startups"` |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LOVABLE_API_KEY` | API key for LLM gateway | Yes |

## License

See parent directory for license information.

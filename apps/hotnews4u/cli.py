"""HotNews4U — LLM-ranked news recommender.

MLOps template:
- Uses synthetic data for demonstration
- Implements LLM-based ranking with Gemini Flash
- Supports JSON-mode output for structured results
- Provides relevance scoring and reasoning

Usage:
    CLI:      python cli.py recommend --interests "AI, startups"
    Streamlit: streamlit run streamlit_app.py
    Notebook: jupyter notebook notebooks/news_recommendation_example.ipynb
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import typer

app = typer.Typer(help="HotNews4U: LLM-ranked news recommendations.")

# ============================================================================
# DATA LOADING
# ============================================================================


def load_news_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load news articles or generate synthetic data.

    Args:
        data_dir: Directory containing news data

    Returns:
        DataFrame with news articles
    """
    filepath = Path(data_dir) / "news_articles.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    # Generate synthetic data
    typer.echo("Generating synthetic news data...")
    import numpy as np
    np.random.seed(42)

    n_articles = 30
    categories = ["tech", "ai", "finance", "sports", "policy", "climate"]
    sources = ["techcrunch", "reuters", "wired", "espn", "euronews", "bloomberg"]

    titles = [
        "Apple announces M5 chip with on-device LLM acceleration",
        "ECB cuts interest rates by 25bps citing inflation concerns",
        "New transformer model beats Gemini 2.5 on MMLU benchmark",
        "Real Madrid wins Club World Cup in extra time thriller",
        "EU AI Act compliance deadline extended to 2027",
        "Tesla unveils 4680 battery tech promising 1000-mile range",
        "Global markets rally as US inflation drops to 2.1%",
        "OpenAI releases GPT-5 with 1M context window",
        "Bitcoin surges past $100,000 amid institutional adoption",
        "Google DeepMind achieves breakthrough in protein folding",
    ]

    articles = []
    base_date = "2026-05-15"
    for i in range(n_articles):
        category = categories[i % len(categories)]
        articles.append({
            "article_id": f"A{i+1:03d}",
            "title": titles[i % len(titles)],
            "category": category,
            "published_date": base_date,
            "sentiment_score": round(np.random.uniform(0.3, 0.9), 2),
            "engagement_score": round(np.random.uniform(5, 10), 1),
            "source": sources[i % len(sources)],
            "read_time": np.random.randint(3, 8),
            "word_count": np.random.randint(800, 2000),
            "url": f"https://{sources[i % len(sources)]}.com/article-{i+1}",
        })

    df = pd.DataFrame(articles)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


# ============================================================================
# LLM RANKING
# ============================================================================


def rank_articles(
    articles: pd.DataFrame,
    interests: str,
    api_key: str = None,
    top_k: int = 5,
) -> dict:
    """Rank articles by relevance to user interests using LLM.

    Args:
        articles: DataFrame with news articles
        interests: Comma-separated list of user interests
        api_key: API key for LLM gateway
        top_k: Number of top articles to return

    Returns:
        Dictionary with ranked articles and scores
    """
    if not api_key:
        api_key = os.environ.get("LOVABLE_API_KEY")

    if not api_key:
        typer.echo("Error: LOVABLE_API_KEY environment variable not set", err=True)
        sys.exit(1)

    # Prepare prompt
    prompt = (
        f"User interests: {interests}\n\n"
        f"Articles:\n{articles.to_json(orient='records')}\n\n"
        f"Rank top {top_k} articles as JSON: "
        '{{"ranked": [{{"id": "...", "score": 0-1, "reason": "..."}}]}}'
    )

    # Call LLM
    response = requests.post(
        "https://ai.gateway.lovable.dev/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "Output ONLY valid JSON. Be strict."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


# ============================================================================
# CLI COMMANDS
# ============================================================================


@app.command()
def recommend(
    interests: str = typer.Option(
        ...,
        "--interests",
        "-i",
        help="Comma-separated interests (e.g., 'AI, startups, finance')"
    ),
    top_k: int = typer.Option(
        5,
        "--top",
        "-k",
        help="Number of top recommendations to return"
    ),
    data_dir: str = typer.Option(
        "data/synthetic",
        "--data-dir",
        "-d",
        help="Data directory path"
    ),
):
    """Recommend news articles based on user interests."""
    typer.echo(f"Loading news data from {data_dir}...")

    articles = load_news_data(data_dir)
    typer.echo(f"Loaded {len(articles)} articles")

    typer.echo(f"Ranking for interests: {interests}")
    typer.echo(f"Top {top_k} recommendations:\n")

    try:
        result = rank_articles(articles, interests, top_k=top_k)
        ranked = result.get("ranked", [])

        for i, item in enumerate(ranked, 1):
            typer.echo(f"{i}. [{item['category']}] {item['title']}")
            typer.echo(f"   Score: {item['score']:.2f}")
            typer.echo(f"   Reason: {item['reason']}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


@app.command()
def list_categories(
    data_dir: str = typer.Option(
        "data/synthetic",
        "--data-dir",
        "-d",
        help="Data directory path"
    ),
):
    """List available article categories."""
    articles = load_news_data(data_dir)
    categories = articles["category"].unique().tolist()

    typer.echo("Available categories:")
    for cat in sorted(categories):
        count = len(articles[articles["category"] == cat])
        typer.echo(f"  - {cat}: {count} articles")


@app.command()
def list_articles(
    data_dir: str = typer.Option(
        "data/synthetic",
        "--data-dir",
        "-d",
        help="Data directory path"
    ),
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category"
    ),
):
    """List available articles."""
    articles = load_news_data(data_dir)

    if category:
        articles = articles[articles["category"] == category]

    typer.echo(f"Articles ({len(articles)} total):")
    for _, row in articles.head(10).iterrows():
        typer.echo(f"  {row['article_id']}: {row['title'][:60]}... [{row['category']}]")

    if len(articles) > 10:
        typer.echo(f"  ... and {len(articles) - 10} more")


@app.command(name="generate-data")
def generate_data(
    output_dir: str = typer.Option(
        "data/synthetic",
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    n_articles: int = typer.Option(
        30,
        "--count",
        "-n",
        help="Number of articles to generate"
    ),
):
    """Generate synthetic news data."""
    import numpy as np

    np.random.seed(42)
    categories = ["tech", "ai", "finance", "sports", "policy", "climate"]
    sources = ["techcrunch", "reuters", "wired", "espn", "euronews", "bloomberg"]

    titles = [
        "Apple announces M5 chip with on-device LLM acceleration",
        "ECB cuts interest rates by 25bps citing inflation concerns",
        "New transformer model beats Gemini 2.5 on MMLU benchmark",
        "Real Madrid wins Club World Cup in extra time thriller",
        "EU AI Act compliance deadline extended to 2027",
        "Tesla unveils 4680 battery tech promising 1000-mile range",
        "Global markets rally as US inflation drops to 2.1%",
        "OpenAI releases GPT-5 with 1M context window",
        "Bitcoin surges past $100,000 amid institutional adoption",
        "Google DeepMind achieves breakthrough in protein folding",
    ]

    articles = []
    for i in range(n_articles):
        category = categories[i % len(categories)]
        articles.append({
            "article_id": f"A{i+1:03d}",
            "title": titles[i % len(titles)],
            "category": category,
            "published_date": "2026-05-15",
            "sentiment_score": round(np.random.uniform(0.3, 0.9), 2),
            "engagement_score": round(np.random.uniform(5, 10), 1),
            "source": sources[i % len(sources)],
            "read_time": np.random.randint(3, 8),
            "word_count": np.random.randint(800, 2000),
            "url": f"https://{sources[i % len(sources)]}.com/article-{i+1}",
        })

    df = pd.DataFrame(articles)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "news_articles.csv"

    df.to_csv(filepath, index=False)
    typer.echo(f"Generated {n_articles} articles to: {filepath}")


if __name__ == "__main__":
    app()

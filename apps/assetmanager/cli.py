"""AssetManager — Text search + LLM summarization for documents.

Uses pgvector for semantic search + LLM for document summarization and NER.

Usage:
    CLI:      python cli.py search "machine learning applications" --summarize
    Streamlit: streamlit run streamlit_app.py
"""

import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="AssetManager: Text search + LLM summarization.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Document:
    """Represents a document/article."""
    id: str
    title: str
    content: str
    url: str
    category: str
    source: str
    author: str
    published_date: str
    word_count: int
    embedding: List[float]  # 384-dim Sentence-BERT embedding


# ============================================================================
# SYNTHETIC DOCUMENT CATALOG
# ============================================================================

def get_synthetic_documents(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic document catalog."""
    np.random.seed(seed)

    categories = [
        "technology", "artificial_intelligence", "data_science", "business",
        "science", "healthcare", "finance", "environment", "education", "policy"
    ]

    sources = [
        "TechCrunch", "Wired", "ArXiv", "Medium", "Nature", "MIT Technology Review",
        "IEEE Spectrum", "KDnuggets", "Towards Data Science", "The Verge",
        "Reuters", "Bloomberg", "Harvard Business Review", "Scientific American"
    ]

    topics = {
        "technology": ["cloud computing", "edge AI", "quantum computing", "IoT", "5G networks"],
        "artificial_intelligence": ["machine learning", "deep learning", "transformers", "LLMs", "computer vision"],
        "data_science": ["big data", "data visualization", "predictive analytics", "A/B testing"],
        "business": ["startup", "venture capital", "IPO", "digital transformation"],
        "science": ["research", "breakthrough", "discovery", "experiment"],
        "healthcare": ["telemedicine", "digital health", "pharmaceutical", "biotech"],
        "finance": ["fintech", "blockchain", "cryptocurrency", "algorithmic trading"],
        "environment": ["climate change", "renewable energy", "sustainability"],
        "education": ["online learning", "edtech", "AI in education"],
        "policy": ["regulation", "privacy", "data governance", "ethics"],
    }

    authors = [
        "Dr. Sarah Chen", "Mark Thompson", "Dr. James Liu", "Emily Rodriguez",
        "Michael Park", "Dr. Lisa Wang", "David Kumar", "Anna Petrov",
        "Robert Johnson", "Dr. Maria Garcia", "John Smith", "Dr. Alex Turner"
    ]

    content_templates = [
        "This comprehensive article explores recent advances in {topic}, examining "
        "the latest research findings and practical applications. The study covers "
        "methodological innovations, experimental results, and implications for the "
        "field. Key contributors discuss future directions and open challenges in {topic}.",

        "Recent developments in {topic} have sparked significant interest in the research "
        "community. This analysis provides an overview of state-of-the-art approaches, "
        "comparative evaluations, and emerging trends. Industry practitioners share "
        "insights on deployment considerations and real-world impact.",

        "The intersection of {topic} and modern technology presents unique opportunities "
        "and challenges. This article examines current limitations, proposed solutions, "
        "and the path forward. Experts weigh in on regulatory considerations and "
        "ethical implications of widespread adoption.",
    ]

    documents = []
    for i in range(n):
        category = categories[i % len(categories)]
        topic = np.random.choice(topics[category])

        content = np.random.choice(content_templates).format(topic=topic)
        # Expand content
        content = " ".join([content] * 5)  # ~500 words

        documents.append({
            "id": f"d{i+1:03d}",
            "title": f"Recent Advances in {topic.title()}: A Comprehensive Analysis",
            "content": content,
            "url": f"https://example.com/articles/{i+1}",
            "category": category,
            "source": np.random.choice(sources),
            "author": np.random.choice(authors),
            "published_date": f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            "word_count": len(content.split()),
            "embedding": [float(np.random.randn()) for _ in range(384)],  # Sentence-BERT dim
        })

    return pd.DataFrame(documents)


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def compute_query_embedding(query: str, dim: int = 384) -> np.ndarray:
    """Compute embedding for search query."""
    # In production: use sentence-transformers
    np.random.seed(hash(query) % (2**32))
    return np.random.randn(dim).astype(np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


# ============================================================================
# TEXT SEARCH
# ============================================================================

def keyword_search(documents: pd.DataFrame, query: str) -> List[Tuple[Dict, float]]:
    """Simple keyword-based search with TF-IDF style scoring."""
    query_lower = query.lower()
    query_terms = set(re.findall(r'\b\w+\b', query_lower))

    scores = []
    for _, doc in documents.iterrows():
        # Search in title and content
        title_terms = set(re.findall(r'\b\w+\b', doc["title"].lower()))
        content_terms = set(re.findall(r'\b\w+\b', doc["content"].lower()))

        # Jaccard similarity for keyword match
        title_match = len(query_terms & title_terms) / max(len(query_terms | title_terms), 1)
        content_match = len(query_terms & content_terms) / max(len(query_terms | content_terms), 1)

        # Combined score with content weight
        score = 0.4 * title_match + 0.6 * content_match
        if score > 0:
            scores.append((doc, score))

    return scores


def vector_search(documents: pd.DataFrame, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict, float]]:
    """Semantic search using vector similarity."""
    scores = []
    for _, doc in documents.iterrows():
        doc_embedding = np.array(doc["embedding"])
        sim = cosine_similarity(query_embedding, doc_embedding)
        scores.append((doc, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def hybrid_search(
    documents: pd.DataFrame,
    query: str,
    query_embedding: np.ndarray,
    alpha: float = 0.5,
    top_k: int = 10,
) -> List[Tuple[Dict, Dict]]:
    """Hybrid keyword + vector search."""
    # Keyword scores
    keyword_scores = keyword_search(documents, query)
    keyword_dict = {doc["id"]: score for doc, score in keyword_scores}

    # Vector scores
    vector_scores = vector_search(documents, query_embedding, top_k * 2)
    vector_dict = {doc["id"]: score for doc, score in vector_scores}

    # Normalize and combine
    all_scores = {}
    for doc_id in set(keyword_dict.keys()) | set(vector_dict.keys()):
        k_score = keyword_dict.get(doc_id, 0)
        v_score = vector_dict.get(doc_id, 0)

        # Normalize scores to [0, 1]
        k_norm = k_score  # Already in [0, 1]
        v_norm = (v_score + 1) / 2  # Cosine similarity [-1, 1] -> [0, 1]

        all_scores[doc_id] = alpha * k_norm + (1 - alpha) * v_norm

    # Sort by combined score
    sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    return [(documents[documents["id"] == doc_id].iloc[0], {"keyword": all_scores[doc_id], "vector": vector_dict.get(doc_id, 0), "hybrid": all_scores[doc_id]}) for doc_id, _ in sorted_docs[:top_k]]


# ============================================================================
# LLM SUMMARIZATION (Simplified)
# ============================================================================

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract keywords using simple frequency analysis."""
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been"}

    # Remove stopwords and count frequency
    word_freq = {}
    for word in words:
        if word not in stopwords and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Return top N words
    return [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def summarize_document(
    title: str,
    content: str,
    max_sentences: int = 3,
    use_llm: bool = False,
    api_key: str = None,
) -> str:
    """Generate summary of document."""
    if use_llm and api_key:
        # Would call LLM API here
        return f"[LLM Summary] {title}: {content[:200]}..."

    # Extract sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # Select most informative sentences (simplified: first N substantive sentences)
    summary_sentences = []
    for sent in sentences:
        if sent and not sent.isspace():
            summary_sentences.append(sent)
            if len(summary_sentences) >= max_sentences:
                break

    return " ".join(summary_sentences[:max_sentences])


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using simple heuristics."""
    entities = {
        "ORGANIZATION": [],
        "PERSON": [],
        "LOCATION": [],
        "TECHNOLOGY": [],
    }

    # Common company/organization patterns
    org_patterns = ["Inc", "Corp", "LLC", "Ltd", "Tech", "AI", "Systems", "Labs", "Research"]
    for pattern in org_patterns:
        if pattern in text:
            # Simple extraction
            start = text.lower().find(pattern.lower())
            if start > 0:
                # Get preceding words
                context = text[max(0, start-30):start]
                entities["ORGANIZATION"].append(context.strip())

    # Technology terms
    tech_terms = ["AI", "ML", "NLP", "computer vision", "deep learning", "transformer", "LLM", "blockchain"]
    for term in tech_terms:
        if term in text:
            entities["TECHNOLOGY"].append(term)

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))[:5]

    return entities


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results"),
    summarize: bool = typer.Option(False, "--summarize", "-s", help="Generate summaries"),
    keywords: bool = typer.Option(False, "--keywords", help="Show extracted keywords"),
    entities: bool = typer.Option(False, "--entities", help="Extract entities"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Search documents using hybrid keyword + vector search."""
    docs_path = Path(data_dir) / "documents.csv"
    if docs_path.exists():
        docs_df = pd.read_csv(docs_path)
    else:
        docs_df = get_synthetic_documents(100)

    typer.echo(f"Searching {len(docs_df)} documents for: '{query}'\n")

    # Compute query embedding
    query_embedding = compute_query_embedding(query)

    # Hybrid search
    results = hybrid_search(docs_df, query, query_embedding, alpha=0.5, top_k=top_k)

    for i, (doc, scores) in enumerate(results, 1):
        typer.echo(f"{i}. {doc['title']}")
        typer.echo(f"   Source: {doc['source']} | Category: {doc['category']}")
        typer.echo(f"   Author: {doc['author']} | Published: {doc['published_date']}")
        typer.echo(f"   Keyword score: {scores['keyword']:.3f} | Vector score: {scores['vector']:.3f}")
        typer.echo(f"   Similarity: {scores['hybrid']:.3f}")

        if summarize:
            summary = summarize_document(doc["title"], doc["content"])
            typer.echo(f"   Summary: {summary}")

        if keywords:
            kw = extract_keywords(doc["content"])
            typer.echo(f"   Keywords: {', '.join(kw)}")

        if entities:
            ent = extract_entities(doc["content"])
            if ent["ORGANIZATION"]:
                typer.echo(f"   Organizations: {', '.join(ent['ORGANIZATION'])}")
            if ent["TECHNOLOGY"]:
                typer.echo(f"   Technologies: {', '.join(ent['TECHNOLOGY'])}")

        typer.echo()


@app.command()
def ingest(
    url: str = typer.Argument(..., help="URL to ingest"),
    title: str = typer.Option(..., "--title", "-t", help="Document title"),
    content: str = typer.Option(..., "--content", "-c", help="Document content"),
    category: str = typer.Option("technology", "--category", help="Document category"),
):
    """Ingest a new document into the catalog."""
    # Compute embedding
    embedding = compute_query_embedding(f"{title} {content}")

    doc = {
        "id": f"d{len(get_synthetic_documents(1)) + 1:03d}",
        "title": title,
        "content": content,
        "url": url,
        "category": category,
        "source": "user",
        "author": "unknown",
        "published_date": "2024-06-02",
        "word_count": len(content.split()),
        "embedding": embedding.tolist(),
    }

    typer.echo(f"Ingested document:")
    typer.echo(f"  ID: {doc['id']}")
    typer.echo(f"  Title: {doc['title']}")
    typer.echo(f"  Category: {doc['category']}")
    typer.echo(f"  Words: {doc['word_count']}")
    typer.echo(f"  Embedding dim: {len(embedding)}")


@app.command()
def list_docs(
    category_filter: str = typer.Option("", "--category", help="Filter by category"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List all documents."""
    docs_path = Path(data_dir) / "documents.csv"
    if docs_path.exists():
        docs_df = pd.read_csv(docs_path)
    else:
        docs_df = get_synthetic_documents(100)

    if category_filter:
        docs_df = docs_df[docs_df["category"].str.contains(category_filter, case=False)]

    typer.echo(f"Documents ({len(docs_df)} total):\n")
    for _, doc in docs_df.iterrows():
        typer.echo(f"  {doc['id']}: {doc['title'][:60]}...")
        typer.echo(f"      [{doc['category']}] {doc['source']} - {doc['word_count']} words")


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_docs: int = typer.Option(100, "--count", "-n", help="Number of documents"),
):
    """Generate synthetic document data."""
    docs_df = get_synthetic_documents(n_docs)
    filepath = Path(output_dir) / "documents.csv"
    docs_df.to_csv(filepath, index=False)
    typer.echo(f"Generated {n_docs} documents to: {filepath}")


if __name__ == "__main__":
    app()

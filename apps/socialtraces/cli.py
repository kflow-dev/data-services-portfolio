"""SocialTraces — Fuzzy entity resolution + vector search for social media.

Combines fuzzy string matching for entity resolution with vector search
for semantic similarity across social media profiles and posts.

Usage:
    CLI:      python cli.py search "John Smith engineer" --platforms all
    Streamlit: streamlit run streamlit_app.py
"""

import re
import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="SocialTraces: Fuzzy entity resolution + vector search.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SocialEntity:
    """Represents a social media entity (person, org, etc.)."""
    id: str
    name: str
    handle: str
    platform: str  # twitter, linkedin, github
    bio: str
    location: str
    verified: bool
    followers: int
    embeddings: Dict[str, np.ndarray]  # name, bio, combined


@dataclass
class SocialPost:
    """Represents a social media post."""
    id: str
    entity_id: str
    content: str
    platform: str
    timestamp: str
    likes: int
    retweets: int
    embedding: np.ndarray


# ============================================================================
# FUZZY MATCHING
# ============================================================================

def normalized_similarity(s1: str, s2: str) -> float:
    """Compute normalized string similarity (0-1)."""
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    # Use difflib's SequenceMatcher
    return difflib.SequenceMatcher(None, s1_lower, s2_lower).ratio()


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def fuzzy_match_entities(
    entities: List[SocialEntity],
    query: str,
    threshold: float = 0.6,
) -> List[Tuple[SocialEntity, float]]:
    """Find entities matching query using fuzzy string matching."""
    matches = []

    query_lower = query.lower()
    query_words = set(query_lower.split())

    for entity in entities:
        # Name similarity
        name_sim = normalized_similarity(entity.name, query)

        # Handle similarity
        handle_sim = normalized_similarity(entity.handle, query)

        # Bio similarity
        bio_sim = normalized_similarity(entity.bio, query)

        # Location similarity
        loc_sim = normalized_similarity(entity.location, query) if entity.location else 0

        # Word overlap score
        entity_words = set(entity.name.lower().split()) | set(entity.bio.lower().split())
        word_overlap = jaccard_similarity(query_words, entity_words)

        # Combined score
        combined = 0.4 * name_sim + 0.2 * handle_sim + 0.2 * bio_sim + 0.1 * loc_sim + 0.1 * word_overlap

        if combined >= threshold:
            matches.append((entity, combined, {
                "name_sim": name_sim,
                "handle_sim": handle_sim,
                "bio_sim": bio_sim,
                "loc_sim": loc_sim,
                "word_overlap": word_overlap,
            }))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


# ============================================================================
# VECTOR SEARCH
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


def vector_search_entities(
    entities: List[SocialEntity],
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[SocialEntity, float]]:
    """Find entities using vector similarity."""
    scores = []

    for entity in entities:
        # Use combined embedding if available
        if "combined" in entity.embeddings:
            emb = entity.embeddings["combined"]
        else:
            emb = entity.embeddings.get("name", np.zeros(384))

        sim = cosine_similarity(query_embedding, emb)
        scores.append((entity, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ============================================================================
# ENTITY RESOLUTION
# ============================================================================

def resolve_entities(
    entities: List[SocialEntity],
    query: str,
    fuzzy_threshold: float = 0.5,
    vector_top_k: int = 20,
) -> List[Dict]:
    """Resolve entities: fuzzy match + vector reranking."""
    # Step 1: Fuzzy matching
    fuzzy_matches = fuzzy_match_entities(entities, query, threshold=fuzzy_threshold)

    # Step 2: Vector search (for semantic similarity)
    # In production, would use sentence-transformers for query embedding
    np.random.seed(hash(query) % (2**32))
    query_embedding = np.random.randn(384).astype(np.float32) * 0.1

    vector_matches = vector_search_entities(entities, query_embedding, vector_top_k)

    # Step 3: Combine scores
    combined_scores = {}

    # Add fuzzy matches
    for entity, score, details in fuzzy_matches:
        key = entity.id
        if key not in combined_scores:
            combined_scores[key] = {"entity": entity, "fuzzy_score": 0, "vector_score": 0, "details": details}
        combined_scores[key]["fuzzy_score"] = score

    # Add vector matches
    vector_dict = {e: s for e, s in vector_matches}
    for entity, score in vector_matches:
        key = entity.id
        if key not in combined_scores:
            combined_scores[key] = {"entity": entity, "fuzzy_score": 0, "vector_score": score, "details": {}}
        else:
            combined_scores[key]["vector_score"] = score

    # Compute hybrid score
    results = []
    for key, data in combined_scores.items():
        fuzzy = data["fuzzy_score"]
        vector = data["vector_score"]

        # Normalize vector score (cosine can be negative)
        vector_norm = (vector + 1) / 2

        hybrid = 0.6 * fuzzy + 0.4 * vector_norm
        data["hybrid_score"] = hybrid
        results.append(data)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results


# ============================================================================
# NETWORK ANALYSIS (Simplified)
# ============================================================================

def compute_network_metrics(
    entity: SocialEntity,
    entities: List[SocialEntity],
    depth: int = 2,
) -> Dict:
    """Compute basic network metrics for an entity."""
    # Simulated metrics (would be computed from actual graph)
    followers = entity.followers
    following = int(followers * np.random.uniform(0.01, 0.1))
    engagement_rate = np.random.uniform(0.02, 0.1)

    # Influence score based on followers and engagement
    influence = min(1.0, np.log10(followers + 1) * engagement_rate)

    # Top connections (simulated)
    connections = []
    for other in entities:
        if other.id != entity.id and np.random.random() < 0.01:
            connections.append({
                "entity_id": other.id,
                "name": other.name,
                "connection_strength": np.random.uniform(0.1, 0.9),
            })

    connections.sort(key=lambda x: x["connection_strength"], reverse=True)

    return {
        "followers": followers,
        "following": following,
        "engagement_rate": round(engagement_rate, 3),
        "influence_score": round(influence, 3),
        "top_connections": connections[:5],
    }


# ============================================================================
# SYNTHETIC DATA
# ============================================================================

def get_synthetic_entities(n: int = 200, seed: int = 42) -> List[SocialEntity]:
    """Generate synthetic social entities."""
    np.random.seed(seed)

    names = [
        "John Smith", "Jane Doe", "Michael Chen", "Sarah Johnson", "David Williams",
        "Emily Brown", "Robert Taylor", "Lisa Anderson", "James Wilson", "Maria Garcia",
        "William Martinez", "Jennifer Lee", "Christopher Davis", "Amanda Miller",
        "Daniel Moore", "Jessica Jackson", "Matthew Thompson", "Ashley White",
        "Andrew Harris", "Melissa Clark", "Joshua Lewis", "Stephanie Robinson",
        "Kevin Walker", "Nicole Hall", "Brian Young", "Rachel King", "Jason Wright",
        "Lauren Scott", "Ryan Green", "Megan Adams", "Justin Baker", "Kimberly Nelson"
    ]

    handles = [
        "@johnsmith", "@janedoe", "@mchen_ai", "@sarahj", "@davidw",
        "@emily_b", "@rtaylor", "@lisaa", "@jwilson", "@mariagarcia",
        "@williamm", "@jennlee", "@cdavis", "@amandam", "@dmoore",
        "@jessj", "@mtompson", "@ashwhite", "@aharris", "@melissac",
        "@joshlew", "@srobinson", "@kwalker", "@nicoleh", "@brian_young",
        "@rachelk", "@jasonwright", "@laurens", "@ryangreen", "@meganad",
        "@justinb", "@knelson"
    ]

    platforms = ["twitter", "linkedin", "github"]

    bios = [
        "Software engineer | AI/ML enthusiast | Building the future",
        "Data scientist at Tech Corp | Python, TensorFlow, PyTorch",
        "Product manager | AI products | Ex-Google, Ex-Microsoft",
        "Research scientist | NLP | PhD from Stanford",
        "Full-stack developer | Open source contributor",
        "Machine learning engineer | Computer vision | Speaker",
        "AI researcher | Publishing papers | Building AGI",
        "Tech entrepreneur | Startup advisor | Investor",
        "DevOps engineer | Cloud infrastructure | Kubernetes",
        "Cybersecurity expert | Penetration tester",
    ]

    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Boston, MA",
        "Austin, TX", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
        "London, UK", "Berlin, Germany", "Toronto, Canada", "Singapore"
    ]

    entities = []
    for i in range(n):
        # Generate embeddings (pseudo-random for demo)
        np.random.seed(hash(names[i]) % (2**32))
        name_emb = np.random.randn(384).astype(np.float32) * 0.1
        bio_emb = np.random.randn(384).astype(np.float32) * 0.1
        combined_emb = (name_emb + bio_emb) / 2

        entities.append(SocialEntity(
            id=f"e{i+1:03d}",
            name=names[i % len(names)],
            handle=handles[i % len(handles)],
            platform=np.random.choice(platforms),
            bio=np.random.choice(bios),
            location=np.random.choice(locations),
            verified=np.random.random() < 0.2,
            followers=int(np.random.exponential(10000)),
            embeddings={"name": name_emb, "bio": bio_emb, "combined": combined_emb},
        ))

    return entities


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def fuzzy_search(
    query: str = typer.Argument(..., help="Search query for entity resolution"),
    platforms: str = typer.Option("all", "--platforms", "-p", help="Platforms: 'twitter', 'linkedin', 'github', 'all'"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Fuzzy match threshold"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results"),
):
    """Fuzzy search across social media entities."""
    entities = get_synthetic_entities(200)

    # Filter by platform
    if platforms != "all":
        entities = [e for e in entities if e.platform == platforms]

    typer.echo(f"Searching for: '{query}'")
    typer.echo(f"Platform: {platforms} | Threshold: {threshold}")

    # Resolve entities
    results = resolve_entities(entities, query, fuzzy_threshold=threshold)

    typer.echo(f"\nTop {len(results)} matches:\n")
    for i, result in enumerate(results[:top_k], 1):
        entity = result["entity"]
        typer.echo(f"{i}. {entity.name}")
        typer.echo(f"   Handle: {entity.handle}")
        typer.echo(f"   Platform: {entity.platform} | Verified: {entity.verified}")
        typer.echo(f"   Location: {entity.location}")
        typer.echo(f"   Bio: {entity.bio}")
        typer.echo(f"   Fuzzy score: {result['fuzzy_score']:.3f} | Vector score: {result['vector_score']:.3f}")
        typer.echo(f"   Hybrid score: {result['hybrid_score']:.3f}")
        if result["details"]:
            typer.echo(f"   Details: name={result['details'].get('name_sim', 0):.2f}, handle={result['details'].get('handle_sim', 0):.2f}")
        typer.echo()


@app.command()
def network_analysis(
    entity_handle: str = typer.Argument(..., help="Entity handle or name to analyze"),
    depth: int = typer.Option(2, "--depth", "-d", help="Network traversal depth"),
):
    """Analyze social network for an entity."""
    entities = get_synthetic_entities(200)

    # Find entity
    target = None
    for e in entities:
        if entity_handle.lower() in e.name.lower() or entity_handle.lower() in e.handle.lower():
            target = e
            break

    if not target:
        # Return first entity if not found
        target = entities[0]
        typer.echo(f"Entity '{entity_handle}' not found. Showing: {target.name}")
    else:
        typer.echo(f"Analyzing: {target.name} ({target.handle})")

    # Compute network metrics
    metrics = compute_network_metrics(target, entities, depth)

    typer.echo(f"\nNetwork Metrics:")
    typer.echo(f"  Followers: {metrics['followers']:,}")
    typer.echo(f"  Following: {metrics['following']:,}")
    typer.echo(f"  Engagement rate: {metrics['engagement_rate']:.1%}")
    typer.echo(f"  Influence score: {metrics['influence_score']:.3f}")

    if metrics["top_connections"]:
        typer.echo(f"\nTop Connections:")
        for conn in metrics["top_connections"]:
            typer.echo(f"  - {conn['name']} (strength: {conn['connection_strength']:.2f})")


@app.command()
def compare_entities(
    entity1: str = typer.Argument(..., help="First entity name/handle"),
    entity2: str = typer.Argument(..., help="Second entity name/handle"),
):
    """Compare two entities for similarity."""
    entities = get_synthetic_entities(200)

    # Find entities
    e1 = None
    e2 = None
    for e in entities:
        if entity1.lower() in e.name.lower() or entity1.lower() in e.handle.lower():
            e1 = e
        if entity2.lower() in e.name.lower() or entity2.lower() in e.handle.lower():
            e2 = e

    if not e1:
        typer.echo(f"Entity '{entity1}' not found")
        return
    if not e2:
        typer.echo(f"Entity '{entity2}' not found")
        return

    # Compute similarities
    name_sim = normalized_similarity(e1.name, e2.name)
    handle_sim = normalized_similarity(e1.handle, e2.handle)
    loc_sim = normalized_similarity(e1.location, e2.location) if e1.location and e2.location else 0

    # Bio similarity using Jaccard
    e1_bio_words = set(e1.bio.lower().split())
    e2_bio_words = set(e2.bio.lower().split())
    bio_sim = jaccard_similarity(e1_bio_words, e2_bio_words)

    typer.echo(f"Comparison: {e1.name} vs {e2.name}")
    typer.echo(f"  Name similarity: {name_sim:.3f}")
    typer.echo(f"  Handle similarity: {handle_sim:.3f}")
    typer.echo(f"  Location similarity: {loc_sim:.3f}")
    typer.echo(f"  Bio similarity: {bio_sim:.3f}")
    typer.echo(f"  Platform match: {e1.platform == e2.platform}")


@app.command()
def list_entities(
    platform_filter: str = typer.Option("", "--platform", help="Filter by platform"),
):
    """List all entities."""
    entities = get_synthetic_entities(200)

    if platform_filter:
        entities = [e for e in entities if e.platform == platform_filter]

    typer.echo(f"Entities ({len(entities)} total):\n")
    for e in entities[:50]:  # Limit output
        typer.echo(f"  {e.id}: {e.name} (@{e.handle.split('@')[-1]})")
        typer.echo(f"      {e.platform} | {e.location}")
    if len(entities) > 50:
        typer.echo(f"\n... and {len(entities) - 50} more")


if __name__ == "__main__":
    app()

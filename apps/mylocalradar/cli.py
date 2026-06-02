"""MyLocalRadar — Geo + semantic location search and disambiguation.

Combines geospatial filtering with semantic similarity for location discovery.
Uses Haversine distance for geo search and vector embeddings for semantic matching.

Usage:
    CLI:      python cli.py search "cafes near downtown" --radius 5
    Streamlit: streamlit run streamlit_app.py
"""

import math
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="MyLocalRadar: Geo + semantic location search.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Location:
    """Represents a geographic location."""
    id: str
    name: str
    latitude: float
    longitude: float
    address: str
    city: str
    country: str
    category: str  # cafe, restaurant, park, museum, etc.
    rating: float
    reviews_count: int
    description: str
    tags: List[str]
    embedding: np.ndarray  # 384-dim semantic embedding


# ============================================================================
# GEOGRAPHICAL UTILITIES
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points (km)."""
    R = 6371  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_city_center(city: str) -> Tuple[float, float]:
    """Get center coordinates for a city."""
    city_coords = {
        "lisbon": (38.7223, -9.1393),
        "porto": (41.1579, -8.6291),
        "madrid": (40.4168, -3.7038),
        "barcelona": (41.3851, 2.1734),
        "paris": (48.8566, 2.3522),
        "london": (51.5074, -0.1278),
        "berlin": (52.5200, 13.4050),
        "amsterdam": (52.3676, 4.9041),
        "rome": (41.9028, 12.4964),
        "tokyo": (35.6762, 139.6503),
        "new york": (40.7128, -74.0060),
        "san francisco": (37.7749, -122.4194),
    }
    return city_coords.get(city.lower(), (48.8566, 2.3522))  # Default to Paris


# ============================================================================
# SEMANTIC MATCHING
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


def generate_location_embedding(name: str, description: str, tags: List[str]) -> np.ndarray:
    """Generate pseudo-embedding for location."""
    text = f"{name} {description} {' '.join(tags)}"
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(384).astype(np.float32) * 0.1


def semantic_match(query: str, locations: List[Location], top_k: int = 10) -> List[Tuple[Location, float]]:
    """Find locations using semantic similarity to query."""
    # Generate query embedding
    np.random.seed(hash(query) % (2**32))
    query_emb = np.random.randn(384).astype(np.float32) * 0.1

    # Score locations
    scores = []
    for loc in locations:
        emb = loc.embedding if loc.embedding else generate_location_embedding(
            loc.name, loc.description, loc.tags
        )
        sim = cosine_similarity(query_emb, emb)
        scores.append((loc, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def parse_location_query(query: str) -> Dict:
    """Parse location query into components."""
    result = {
        "city": None,
        "category": None,
        "keywords": [],
        "radius": 10,  # default 10 km
    }

    query_lower = query.lower()

    # Detect city
    cities = ["lisbon", "porto", "madrid", "barcelona", "paris", "london", "berlin", "rome", "tokyo"]
    for city in cities:
        if city in query_lower:
            result["city"] = city
            break

    # Detect category
    categories = ["cafe", "restaurant", "park", "museum", "hotel", "shop", "gym", "library", "theater"]
    for cat in categories:
        if cat in query_lower:
            result["category"] = cat
            break

    # Extract radius if specified
    radius_match = re.search(r'(?:within|near|km)?\s*(\d+)\s*km', query_lower)
    if radius_match:
        result["radius"] = int(radius_match.group(1))

    # Extract keywords (remove known words)
    stopwords = {"near", "within", "km", "best", "top", "good", "nice", "around", "close"}
    keywords = [w for w in query_lower.split() if w not in stopwords and len(w) > 2]
    result["keywords"] = keywords

    return result


# ============================================================================
# HYBRID LOCATION SEARCH
# ============================================================================

def hybrid_location_search(
    locations: List[Location],
    query: str,
    center_lat: float,
    center_lon: float,
    radius_km: float = 10,
) -> List[Dict]:
    """Combine geo distance + semantic similarity for location search."""
    results = []

    for loc in locations:
        # Geo distance score (0-1, higher is closer)
        distance = haversine_distance(center_lat, center_lon, loc.latitude, loc.longitude)
        geo_score = max(0, 1 - distance / radius_km) if distance <= radius_km else 0

        # Semantic score
        semantic_score = cosine_similarity(
            loc.embedding if loc.embedding else generate_location_embedding(
                loc.name, loc.description, loc.tags
            ),
            np.random.randn(384).astype(np.float32) * 0.1  # Would be query embedding
        )

        # Hybrid score
        hybrid_score = 0.6 * geo_score + 0.4 * (semantic_score + 1) / 2  # Normalize semantic

        results.append({
            "location": loc,
            "distance_km": round(distance, 2),
            "geo_score": round(geo_score, 3),
            "semantic_score": round(semantic_score, 3),
            "hybrid_score": round(hybrid_score, 3),
        })

    # Filter by radius
    results = [r for r in results if r["distance_km"] <= radius_km]

    # Sort by hybrid score
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results


# ============================================================================
# LOCATION DISAMBIGUATION
# ============================================================================

def disambiguate_location(
    name: str,
    locations: List[Location],
    context: str = "",
    top_k: int = 5,
) -> List[Tuple[Location, float]]:
    """Disambiguate location name using context and similarity."""
    candidates = [loc for loc in locations if name.lower() in loc.name.lower()]

    if not candidates:
        return []

    results = []
    for loc in candidates:
        # Base confidence from name match
        name_match = len(set(name.lower().split()) & set(loc.name.lower().split())) / max(len(name.lower().split()), 1)

        # Context boost
        context_boost = 0
        if context:
            context_words = set(context.lower().split())
            loc_context = set(loc.description.lower().split()) | set(loc.tags)
            context_overlap = len(context_words & loc_context) / max(len(context_words), 1)
            context_boost = context_overlap * 0.2

        # Rating boost
        rating_boost = (loc.rating - 3) / 2 * 0.1  # Normalize rating contribution

        confidence = name_match + context_boost + rating_boost
        results.append((loc, confidence))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================================
# SYNTHETIC DATA
# ============================================================================

def get_synthetic_locations(n: int = 200, seed: int = 42) -> List[Location]:
    """Generate synthetic locations (simulating a city like Lisbon)."""
    np.random.seed(seed)

    categories = ["cafe", "restaurant", "park", "museum", "hotel", "shop", "gym", "library"]
    tags_options = [
        ["wifi", "outdoor_seating", "breakfast", "vegan"],
        ["fine_dining", "romantic", "terrace", "wine"],
        ["playground", "walking", "green", "family"],
        ["art", "history", "guided_tours", "audio_guide"],
        ["luxury", "pool", "spa", "business"],
        ["fashion", "electronics", "local", "handmade"],
        ["crossfit", "yoga", "personal_trainer", "24_7"],
        ["quiet", "study_space", "events", "free_wifi"],
    ]

    names_templates = {
        "cafe": ["Blue Bottle", "Starbucks", "Local Roast", "Café Central", "Pastelaria"],
        "restaurant": ["The Capital Grille", "Nobu", "José Avillez", "Time Out Market", "A Cevicheria"],
        "park": ["Parque Eduardo VII", "Jardim Botânico", "Parque das Nações", "Campo Grande"],
        "museum": ["Museu Nacional", "Museu de Arte", "Gulbenkian", "Pavilhão do Conhecimento"],
        "hotel": ["Four Seasons", "Tivoli", "Sana", "Corinthia"],
        "shop": ["El Corte Inglés", "Fnac", "Woolworth", "Continente"],
        "gym": ["Hydro Fit", "SmartFit", "Mind Body Soul", "CrossFit Lisbon"],
        "library": ["Biblioteca Nacional", "Livraria Bertrand", "Mediateca"],
    }

    descriptions = {
        "cafe": "Cozy café with artisan coffee and pastries, perfect for working or relaxing",
        "restaurant": "Fine dining restaurant with local cuisine and excellent wine selection",
        "park": "Urban park with walking paths, playgrounds, and green spaces",
        "museum": "Museum featuring art, history, and cultural exhibitions",
        "hotel": "Luxury hotel with modern amenities and city views",
        "shop": "Retail store offering fashion, electronics, or local products",
        "gym": "Fitness center with modern equipment and group classes",
        "library": "Public library with books, study spaces, and events",
    }

    # Generate locations around Lisbon center
    center_lat, center_lon = 38.7223, -9.1393

    locations = []
    for i in range(n):
        category = np.random.choice(list(categories))
        name_template = np.random.choice(names_templates[category])
        name = f"{name_template} {i+1}"

        # Random position around center (within ~10km)
        lat_offset = np.random.normal(0, 0.05)  # ~5km std
        lon_offset = np.random.normal(0, 0.05)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset

        embedding = generate_location_embedding(
            name,
            descriptions[category],
            np.random.choice(tags_options, 1)[0]
        )

        locations.append(Location(
            id=f"loc{i+1:03d}",
            name=name,
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            address=f"{np.random.randint(1, 999)} {['Rua', 'Avenida', 'Praça'][np.random.randint(3)]} {['do', 'da', 'de'][np.random.randint(3)]} {np.random.randint(1000)}",
            city="Lisbon",
            country="Portugal",
            category=category,
            rating=round(np.random.uniform(3.0, 5.0), 1),
            reviews_count=int(np.random.exponential(100)),
            description=descriptions[category],
            tags=np.random.choice(tags_options, 1)[0].tolist(),
            embedding=embedding,
        ))

    return locations


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def search(
    query: str = typer.Argument(..., help="Location search query (e.g., 'cafes near downtown')"),
    city: str = typer.Option("", "--city", "-c", help="Override city"),
    radius: float = typer.Option(5.0, "--radius", "-r", help="Search radius in km"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results"),
):
    """Search for locations using geo + semantic matching."""
    locations = get_synthetic_locations(200)

    # Parse query
    parsed = parse_location_query(query)

    # Get city center
    target_city = city or parsed["city"] or "lisbon"
    center_lat, center_lon = get_city_center(target_city)

    typer.echo(f"Searching for: '{query}'")
    typer.echo(f"City: {target_city.title()} | Radius: {radius}km")
    typer.echo()

    # Hybrid search
    results = hybrid_location_search(locations, query, center_lat, center_lon, radius)

    typer.echo(f"Found {len(results)} locations:\n")
    for i, result in enumerate(results[:top_k], 1):
        loc = result["location"]
        typer.echo(f"{i}. {loc.name}")
        typer.echo(f"   {loc.address}, {loc.city}")
        typer.echo(f"   Category: {loc.category} | Rating: {loc.rating} ({loc.reviews_count} reviews)")
        typer.echo(f"   Tags: {', '.join(loc.tags)}")
        typer.echo(f"   Distance: {result['distance_km']:.1f}km | Hybrid score: {result['hybrid_score']:.3f}")
        typer.echo()


@app.command()
def disambiguate(
    location_name: str = typer.Argument(..., help="Location name to disambiguate"),
    context: str = typer.Option("", "--context", "-c", help="Context (country, region, nearby places)"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of disambiguation options"),
):
    """Disambiguate location name using context."""
    locations = get_synthetic_locations(200)

    typer.echo(f"Disambiguating: '{location_name}'")
    if context:
        typer.echo(f"Context: '{context}'")

    results = disambiguate_location(location_name, locations, context, top_k)

    if not results:
        typer.echo("No matching locations found.")
        return

    typer.echo(f"\nPossible matches:\n")
    for i, (loc, confidence) in enumerate(results, 1):
        typer.echo(f"{i}. {loc.name}")
        typer.echo(f"   {loc.address}, {loc.city}, {loc.country}")
        typer.echo(f"   Confidence: {confidence:.2f} | Rating: {loc.rating}")
        typer.echo()


@app.command()
def geocode(
    address: str = typer.Argument(..., help="Address to geocode"),
):
    """Convert address to coordinates (simulated)."""
    typer.echo(f"Geocoding: {address}")
    typer.echo()

    # Simulated geocoding
    lat = round(38.7223 + np.random.normal(0, 0.001), 6)
    lon = round(-9.1393 + np.random.normal(0, 0.001), 6)

    typer.echo("Result:")
    typer.echo(f"  Latitude: {lat}")
    typer.echo(f"  Longitude: {lon}")
    typer.echo(f"  Confidence: 0.95")
    typer.echo(f"  Match type: ROOFTOP")


@app.command()
def list_locations(
    category_filter: str = typer.Option("", "--category", help="Filter by category"),
    city_filter: str = typer.Option("lisbon", "--city", help="Filter by city"),
):
    """List all locations."""
    locations = get_synthetic_locations(200)

    if category_filter:
        locations = [loc for loc in locations if loc.category == category_filter]
    if city_filter:
        locations = [loc for loc in locations if loc.city.lower() == city_filter.lower()]

    typer.echo(f"Locations ({len(locations)} total):\n")
    for loc in locations[:30]:
        typer.echo(f"  {loc.id}: {loc.name}")
        typer.echo(f"      {loc.category} | {loc.city} | Rating: {loc.rating}")
    if len(locations) > 30:
        typer.echo(f"\n... and {len(locations) - 30} more")


@app.command()
def nearby(
    lat: float = typer.Argument(..., help="Reference latitude"),
    lon: float = typer.Argument(..., help="Reference longitude"),
    radius: float = typer.Option(5.0, "--radius", "-r", help="Search radius in km"),
    category: str = typer.Option("", "--category", help="Filter by category"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results"),
):
    """Find locations near coordinates."""
    locations = get_synthetic_locations(200)

    if category:
        locations = [loc for loc in locations if loc.category == category]

    typer.echo(f"Nearby locations from ({lat}, {lon}) within {radius}km:\n")

    results = []
    for loc in locations:
        distance = haversine_distance(lat, lon, loc.latitude, loc.longitude)
        if distance <= radius:
            results.append((loc, distance))

    results.sort(key=lambda x: x[1])

    for i, (loc, distance) in enumerate(results[:top_k], 1):
        typer.echo(f"{i}. {loc.name}")
        typer.echo(f"   {distance:.1f}km away | {loc.category} | Rating: {loc.rating}")


if __name__ == "__main__":
    app()

"""E-books & Audiobook RecSys — Content-based recommender using Sentence-BERT embeddings.

Uses semantic text embeddings for book recommendation based on plot, themes, and style.

Usage:
    CLI:      python cli.py recommend "hard sci-fi, Ted Chiang" --format both
    Streamlit: streamlit run streamlit_app.py
    Notebook:  jupyter notebooks/book_recommender_example.ipynb
"""

import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="Ebooks: Content-based book recommendations using embeddings.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Book:
    """Represents a book (e-book or audiobook)."""
    id: str
    title: str
    author: str
    genre: str
    subgenre: str
    description: str
    pages: int
    format: str  # ebook, audiobook, both
    rating: float
    publication_year: int
    language: str
    narrator: str = ""  # For audiobooks


# ============================================================================
# SYNTHETIC BOOK CATALOG
# ============================================================================

def get_synthetic_book_catalog(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic book catalog for testing."""
    np.random.seed(seed)

    genres = [
        "science_fiction", "fantasy", "mystery", "romance", "thriller",
        "non_fiction", "biography", "history", "self_help", "philosophy"
    ]

    subgenres = {
        "science_fiction": ["hard_sci_fi", "space_opera", "cyberpunk", "dystopian", "time_travel", "aliens"],
        "fantasy": ["high_fantasy", "urban_fantasy", "dark_fantasy", "epic_fantasy", "fae"],
        "mystery": ["cozy_mystery", "noir", "police_procedural", "psychological_thriller"],
        "romance": ["contemporary", "historical", "paranormal", "young_adult"],
        "thriller": ["political", "medical", "legal", "action", "espionage"],
        "non_fiction": ["science", "technology", "psychology", "economics", "politics"],
        "biography": ["celebrity", "historical_figure", "artist", "scientist", "athlete"],
        "history": ["ancient", "medieval", "modern", "military", "cultural"],
        "self_help": ["productivity", "mindfulness", "relationships", "career", "finance"],
        "philosophy": ["ethics", "existentialism", "political_philosophy", "eastern_philosophy"],
    }

    descriptions = {
        "hard_sci_fi": "Rigorous scientific concepts explored through compelling storytelling",
        "space_opera": "Epic adventures across galaxies with grand stakes",
        "cyberpunk": "High tech, low life in dystopian future societies",
        "dystopian": "Dark visions of oppressive futures and resistance",
        "time_travel": "Paradoxes and consequences of altering the past",
        "aliens": "First contact and interspecies encounters",
        "high_fantasy": "Magic and mythical creatures in secondary worlds",
        "urban_fantasy": "Magic hidden in modern cities",
        "noir": "Dark, cynical detective stories",
        "psychological_thriller": "Mind games and unreliable narrators",
        "contemporary": "Modern love stories and relationships",
        "science": "Scientific discoveries and their implications",
        "philosophy": "Deep questions about existence and morality",
    }

    authors = [
        "Ted Chiang", "Andy Weir", "Becky Chambers", "N.K. Jemisin", "Liu Cixin",
        "Brandon Sanderson", "Patrick Rothfuss", "Joe Abercrombie", "Sarah J. Maas",
        "Gillian Flynn", "Alex Michaelides", "Paula Hawkins", "Stieg Larsson",
        "Malcolm Gladwell", "Yuval Noah Harari", "Carl Sagan", "Neil deGrasse Tyson",
        "Chimamanda Ngozi Adichie", "Margaret Atwood", "Ursula K. Le Guin",
        "Frank Herbert", "Isaac Asimov", "Philip K. Dick", "Jorge Luis Borges",
        "Hermann Hesse", "Albert Camus", "Friedrich Nietzsche", "Sima Beck",
        "Elizabeth Gilbert", "Brené Brown", "James Clear"
    ]

    titles = [
        "The Three-Body Problem", "Project Hail Mary", "The Martian", "Dune",
        "Snow Crash", "Neuromancer", "The Time Machine", "Slaughterhouse-Five",
        "Daisy Jones & The Six", "Gone Girl", "The Silent Patient", "The Girl with the Dragon Tattoo",
        "Sapiens", "Homo Deus", "The Power", "Klara and the Sun", "Station Eleven",
        "The Fifth Season", "The Broken Earth", "The Priory of the Orange Tree",
        "American Gods", "The Name of the Wind", "The Wise Man's Fear",
        "Thinking, Fast and Slow", "Outliers", "The Tipping Point", "Atomic Habits",
        "The Subtle Art of Not Giving a F*ck", "Educated", "Becoming", "Wild"
    ]

    books = []
    for i in range(n):
        genre = genres[i % len(genres)]
        subgenre = np.random.choice(subgenres[genre])
        author = np.random.choice(authors)

        # Format distribution
        format_choice = np.random.choice(["ebook", "audiobook", "both"], p=[0.5, 0.3, 0.2])

        books.append({
            "id": f"b{i+1:03d}",
            "title": titles[i % len(titles)] if i < len(titles) else f"Book Title {i+1}",
            "author": author if i < len(authors) else f"Author {i+1}",
            "genre": genre,
            "subgenre": subgenre,
            "description": descriptions.get(subgenre, f"An engaging {genre} novel with {subgenre} elements"),
            "pages": int(np.random.randint(200, 600)),
            "format": format_choice,
            "rating": round(np.random.uniform(3.0, 5.0), 1),
            "publication_year": np.random.randint(1950, 2024),
            "language": "en",
            "narrator": f"Narrator {i+1}" if format_choice in ["audiobook", "both"] else "",
        })

    return pd.DataFrame(books)


# ============================================================================
# SEMANTIC EMBEDDINGS (Simplified for demo)
# ============================================================================

def generate_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate pseudo-embedding from text using simple hashing.

    In production, use sentence-transformers:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text)
    """
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(dim).astype(np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


# ============================================================================
# CONTENT-BASED RECOMMENDATION
# ============================================================================

class BookRecommender:
    """Content-based book recommender using text embeddings."""

    def __init__(self):
        self.books: List[Book] = []
        self.embeddings: Dict[str, np.ndarray] = {}

    def load_catalog(self, books_df: pd.DataFrame):
        """Load book catalog and generate embeddings."""
        self.books = [Book(**row.to_dict()) for _, row in books_df.iterrows()]

        # Generate embeddings for all books
        for book in self.books:
            # Embed description + title for content-based recommendation
            text = f"{book.title} by {book.author}. {book.description}"
            self.embeddings[book.id] = generate_text_embedding(text)

    def recommend_by_genre(
        self,
        genre: str,
        format: str = "both",
        top_k: int = 10,
    ) -> List[Dict]:
        """Recommend books by genre."""
        genre_lower = genre.lower()

        # Filter by genre
        filtered = []
        for book in self.books:
            book_genre = book.genre.lower()
            if genre_lower in book_genre or book_genre in genre_lower:
                if format == "both" or book.format == format:
                    filtered.append(book)

        # Sort by rating
        filtered.sort(key=lambda b: b.rating, reverse=True)
        return [{"book": b, "score": b.rating} for b in filtered[:top_k]]

    def recommend_by_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """Recommend books based on search query using semantic similarity."""
        query_embedding = generate_text_embedding(query)

        # Compute similarities
        scores = []
        for book in self.books:
            book_embedding = self.embeddings[book.id]
            sim = cosine_similarity(query_embedding, book_embedding)
            scores.append({
                "book": book,
                "similarity": float(sim),
            })

        # Sort by similarity
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]

    def find_similar(
        self,
        book_id: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """Find books similar to a given book."""
        if book_id not in self.embeddings:
            return []

        target_embedding = self.embeddings[book_id]

        scores = []
        for book in self.books:
            if book.id == book_id:
                continue
            book_embedding = self.embeddings[book.id]
            sim = cosine_similarity(target_embedding, book_embedding)
            scores.append({
                "book": book,
                "similarity": float(sim),
            })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def recommend(
    query: str = typer.Argument(..., help="Search query or genre (e.g., 'hard sci-fi, Ted Chiang')"),
    format: str = typer.Option("both", "--format", "-f", help="Format: 'ebook', 'audiobook', 'both'"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Recommend books based on query."""
    books_path = Path(data_dir) / "books.csv"
    if books_path.exists():
        books_df = pd.read_csv(books_path)
    else:
        books_df = generate_synthetic_book_catalog(100)

    typer.echo(f"Loaded {len(books_df)} books")
    typer.echo(f"\nQuery: '{query}'")
    typer.echo(f"Format: {format}")

    recommender = BookRecommender()
    recommender.load_catalog(books_df)

    # Try to detect if query is genre-based or semantic
    genres = ["science fiction", "fantasy", "mystery", "romance", "thriller", "non-fiction", "biography", "history", "self-help", "philosophy"]
    is_genre = any(g in query.lower() for g in genres)

    if is_genre:
        recs = recommender.recommend_by_genre(query, format, top_k)
    else:
        recs = recommender.recommend_by_query(query, top_k)

    typer.echo(f"\nTop {top_k} recommendations:\n")
    for i, rec in enumerate(recs, 1):
        book = rec["book"]
        score = rec["score"]
        score_label = "Rating" if "score" == "rating" else "Similarity"
        typer.echo(f"{i}. {book.title}")
        typer.echo(f"   by {book.author}")
        typer.echo(f"   Genre: {book.genre} ({book.subgenre})")
        typer.echo(f"   Format: {book.format} | {book.pages} pages | Rating: {book.rating}")
        typer.echo(f"   {score_label}: {score:.2f}")
        if book.description:
            typer.echo(f"   {book.description}")
        typer.echo()


@app.command()
def similar(
    book_id: str = typer.Argument(..., help="Book ID to find similar books for"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Find books similar to a given book."""
    books_path = Path(data_dir) / "books.csv"
    if books_path.exists():
        books_df = pd.read_csv(books_path)
    else:
        books_df = generate_synthetic_book_catalog(100)

    typer.echo(f"Finding similar books to ID: {book_id}")

    recommender = BookRecommender()
    recommender.load_catalog(books_df)

    recs = recommender.find_similar(book_id, top_k)

    if not recs:
        typer.echo("Book not found or no similar books available.")
        return

    typer.echo(f"\nTop {top_k} similar books:\n")
    for i, rec in enumerate(recs, 1):
        book = rec["book"]
        typer.echo(f"{i}. {book.title} by {book.author}")
        typer.echo(f"   Similarity: {rec['similarity']:.3f}")
        typer.echo()


@app.command()
def list_books(
    genre_filter: str = typer.Option("", "--genre", help="Filter by genre"),
    format_filter: str = typer.Option("", "--format", help="Filter by format"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List available books."""
    books_path = Path(data_dir) / "books.csv"
    if books_path.exists():
        books_df = pd.read_csv(books_path)
    else:
        books_df = generate_synthetic_book_catalog(100)

    if genre_filter:
        books_df = books_df[books_df["genre"].str.contains(genre_filter, case=False)]
    if format_filter:
        books_df = books_df[books_df["format"] == format_filter]

    typer.echo(f"Books ({len(books_df)} total):\n")
    for _, book in books_df.iterrows():
        typer.echo(f"  {book['id']}: {book['title']} by {book['author']}")
        typer.echo(f"      {book['genre']} ({book['subgenre']}) | {book['format']} | Rating: {book['rating']}")


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_books: int = typer.Option(100, "--count", "-n", help="Number of books"),
):
    """Generate synthetic book data."""
    books_df = generate_synthetic_book_catalog(n_books)
    filepath = Path(output_dir) / "books.csv"
    books_df.to_csv(filepath, index=False)
    typer.echo(f"Generated {n_books} books to: {filepath}")


if __name__ == "__main__":
    app()

"""SciTubbies — Collaborative filtering YouTube content recommender.

Uses user-item matrix from implicit library for collaborative filtering
recommendations based on viewing patterns.

Usage:
    CLI:      python cli.py recommend "machine learning, physics" --type educational
    Streamlit: streamlit run streamlit_app.py
    Notebook:  jupyter notebooks/youtube_recommender_example.ipynb
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="SciTubbies: Collaborative filtering video recommendations.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Video:
    """Represents a YouTube video."""
    id: str
    title: str
    channel: str
    topic: str
    subtopic: str
    duration_seconds: int
    views: int
    likes: int
    uploaded_date: str
    description: str


@dataclass
class UserView:
    """Represents a user's viewing history."""
    user_id: str
    video_id: str
    watch_duration_seconds: int
    liked: bool
    timestamp: str


# ============================================================================
# CONTENT CATALOG
# ============================================================================

def get_youtube_content_catalog(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic YouTube content catalog."""
    np.random.seed(seed)

    topics = [
        "machine_learning", "artificial_intelligence", "physics", "mathematics",
        "computer_science", "biology", "chemistry", "astronomy", "psychology",
        "neuroscience", "data_science", "robotics", "quantum_computing"
    ]

    subtopics = {
        "machine_learning": ["neural_networks", "deep_learning", "reinforcement_learning", "nlp", "computer_vision"],
        "artificial_intelligence": ["generative_ai", "transformers", "llms", "agent_systems", "ethics"],
        "physics": ["quantum_physics", "relativity", "particle_physics", "astrophysics", "thermodynamics"],
        "mathematics": ["calculus", "linear_algebra", "statistics", "probability", "number_theory"],
        "computer_science": ["algorithms", "data_structures", "operating_systems", "networking", "security"],
        "biology": ["genetics", "evolution", "ecology", "molecular_biology", "neuroscience"],
        "chemistry": ["organic_chemistry", "biochemistry", "physical_chemistry", "materials"],
        "astronomy": ["exoplanets", "cosmology", "black_holes", "solar_system", "galaxies"],
        "psychology": ["cognitive_science", "behavioral_psychology", "neuroscience", "mental_health"],
        "data_science": ["visualization", "big_data", "sql", "python", "ml_operations"],
        "robotics": ["autonomous_vehicles", "drones", "industrial_robotics", "humanoid_robots"],
        "quantum_computing": ["qubits", "quantum_algorithms", "error_correction", "applications"],
    }

    channels = [
        "3Blue1Brown", "Veritasium", "Kurzgesagt", "Computerphile", "Numberphile",
        "Sentdex", "Two Minute Papers", "StatQuest", "Lex Fridman", " Yannic Kilcher",
        "ArjanCodes", "Fireship", "The Engineering Mindset", "Sabine Hossenfelder",
        "Dr. Becky", "PBS Space Time", "MinutePhysics", "Physics Girl", "Ali the Engineer",
        "Great Scott!", "Andreas Spiess", "Real Engineering", "Real Engineering", "Mark Rober"
    ]

    titles = [
        "Introduction to Neural Networks", "The Math of Machine Learning", "Deep Learning Explained",
        "Transformers Explained", "Quantum Computing for Beginners", "The Future of AI",
        "Linear Algebra Fundamentals", "Calculus Made Easy", "Statistics for Data Science",
        "Python for Data Analysis", "Neural Network Architecture", "Reinforcement Learning Basics",
        "NLP with Transformers", "Computer Vision Deep Dive", "Quantum Mechanics Basics",
        "Relativity Explained", "Particle Physics Introduction", "Black Holes Explained",
        "Genetics and Evolution", "Molecular Biology Basics", "Algorithm Complexity",
        "Database Design", "Network Security Basics", "Operating Systems Concepts",
        "Robotics and Automation", "Autonomous Vehicles", "Drones and UAVs",
        "Generative AI Models", "Large Language Models", "AI Ethics and Safety"
    ]

    descriptions = {
        "machine_learning": "Comprehensive guide to machine learning concepts and applications",
        "deep_learning": "Deep dive into neural networks and training methodologies",
        "physics": "Understanding the fundamental laws of physics",
        "mathematics": "Mathematical foundations for science and engineering",
        "computer_science": "Core CS concepts and programming fundamentals",
        "biology": "Life sciences from molecular to ecosystem level",
        "astronomy": "Exploring the cosmos and celestial phenomena",
        "data_science": "Data analysis and visualization techniques",
        "robotics": "Building and programming robotic systems",
        "quantum_computing": "Quantum mechanics meets computation",
    }

    videos = []
    for i in range(n):
        topic = topics[i % len(topics)]
        subtopic = np.random.choice(subtopics[topic])
        channel = np.random.choice(channels)

        # Views distribution (power law-like)
        views = int(np.random.pareto(1.0) * 10000 + 1000)
        likes = int(views * np.random.uniform(0.01, 0.05))
        duration = int(np.random.uniform(300, 3600))  # 5 min to 1 hour

        videos.append({
            "id": f"v{i+1:03d}",
            "title": titles[i % len(titles)] if i < len(titles) else f"Video {i+1}",
            "channel": channel,
            "topic": topic,
            "subtopic": subtopic,
            "duration_seconds": duration,
            "views": views,
            "likes": likes,
            "uploaded_date": f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            "description": descriptions.get(topic, f"An educational video about {topic}"),
        })

    return pd.DataFrame(videos)


def generate_user_views(videos_df: pd.DataFrame, n_users: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic user viewing history."""
    np.random.seed(seed)

    views = []
    for user_num in range(n_users):
        user_id = f"u{user_num+1:03d}"

        # Each user watches a subset of videos
        n_watched = np.random.randint(10, 50)
        watched_videos = videos_df.sample(n=min(n_watched, len(videos_df)))

        for _, video in watched_videos.iterrows():
            # Watch duration (fraction of video watched)
            watch_pct = np.random.uniform(0.1, 1.0)
            watch_duration = int(video["duration_seconds"] * watch_pct)

            # Like probability based on views ratio
            like_prob = min(0.5, video["likes"] / video["views"] * 2)
            liked = np.random.random() < like_prob

            views.append({
                "user_id": user_id,
                "video_id": video["id"],
                "watch_duration_seconds": watch_duration,
                "liked": liked,
                "timestamp": f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            })

    return pd.DataFrame(views)


# ============================================================================
# COLLABORATIVE FILTERING (Implicit ALS-like)
# ============================================================================

class CollaborativeFilterRecommender:
    """Collaborative filtering recommender using user-item interactions."""

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.users = []
        self.items = []
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.video_catalog = {}

    def build_mappings(self, views_df: pd.DataFrame):
        """Build user and item ID mappings."""
        self.users = views_df["user_id"].unique().tolist()
        self.items = views_df["video_id"].unique().tolist()

        self.user_to_idx = {uid: i for i, uid in enumerate(self.users)}
        self.item_to_idx = {iid: i for i, iid in enumerate(self.items)}

    def build_user_item_matrix(self, views_df: pd.DataFrame) -> np.ndarray:
        """Build user-item interaction matrix (watch count weighted)."""
        n_users = len(self.users)
        n_items = len(self.items)
        matrix = np.zeros((n_users, n_items))

        for _, row in views_df.iterrows():
            u_idx = self.user_to_idx[row["user_id"]]
            i_idx = self.item_to_idx[row["video_id"]]
            # Weight by watch duration and like status
            weight = row["watch_duration_seconds"] / 3600  # Fraction of hour
            if row["liked"]:
                weight *= 2
            matrix[u_idx, i_idx] = weight

        return matrix

    def fit(self, views_df: pd.DataFrame, videos_df: pd.DataFrame):
        """Train the collaborative filtering model."""
        self.build_mappings(views_df)
        R = self.build_user_item_matrix(views_df)

        n_users, n_items = R.shape

        # Initialize factors
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Regularization
        reg = 0.1

        # ALS iterations
        for iteration in range(20):
            # Update user factors
            for u in range(n_users):
                item_indices = np.where(R[u, :] > 0)[0]
                if len(item_indices) == 0:
                    continue

                R_u = self.item_factors[item_indices, :]
                c_u = R[u, item_indices]

                A = R_u.T @ R_u + reg * np.eye(self.n_factors)
                b = R_u.T @ (c_u[:, np.newaxis] * R_u)
                self.user_factors[u, :] = np.linalg.solve(A, b.flatten())

            # Update item factors
            for i in range(n_items):
                user_indices = np.where(R[:, i] > 0)[0]
                if len(user_indices) == 0:
                    continue

                R_i = self.user_factors[user_indices, :]
                c_i = R[user_indices, i]

                A = R_i.T @ R_i + reg * np.eye(self.n_factors)
                b = R_i.T @ (c_i[:, np.newaxis] * R_i)
                self.item_factors[i, :] = np.linalg.solve(A, b.flatten())

        # Build video catalog
        for _, video in videos_df.iterrows():
            self.video_catalog[video["id"]] = video.to_dict()

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Dict]:
        """Recommend videos for a specific user."""
        if user_id not in self.user_to_idx:
            return self.recommend_popular(top_k)

        user_idx = self.user_to_idx[user_id]
        user_factor = self.user_factors[user_idx]

        # Compute scores for all items
        scores = []
        for i, video_id in enumerate(self.items):
            score = np.dot(user_factor, self.item_factors[i])
            scores.append({
                "video_id": video_id,
                "score": float(score),
            })

        # Exclude seen videos if requested
        if exclude_seen:
            # Get seen videos for this user
            seen = set(
                views_df[views_df["user_id"] == user_id]["video_id"].tolist()
                for views_df in [self._load_views()]
            )[0]
            scores = [s for s in scores if s["video_id"] not in seen]

        # Sort by score
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def recommend_by_topic(
        self,
        topic: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """Recommend videos based on topic similarity."""
        topic_lower = topic.lower()

        # Score videos by topic match
        scores = []
        for video_id, video in self.video_catalog.items():
            topic_match = 0
            if topic_lower in video["topic"].lower():
                topic_match += 2
            if topic_lower in video["subtopic"].lower():
                topic_match += 1
            if topic_match > 0:
                scores.append({
                    "video": video,
                    "topic_score": float(topic_match),
                    "cf_score": 0.0,
                })

        scores.sort(key=lambda x: x["topic_score"], reverse=True)
        return scores[:top_k]

    def recommend_popular(self, top_k: int = 10) -> List[Dict]:
        """Recommend most popular videos (fallback for new users)."""
        popular = sorted(
            self.video_catalog.values(),
            key=lambda v: v["views"],
            reverse=True
        )[:top_k]

        return [{"video": v, "popularity": v["views"]} for v in popular]


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def recommend(
    query: str = typer.Argument(..., help="Topic or interest (e.g., 'machine learning, physics')"),
    channel_type: str = typer.Option("all", "--type", "-t", help="Channel type: 'educational', 'research', 'news', 'all'"),
    duration: str = typer.Option("medium", "--duration", "-d", help="Video length: 'short' (<10min), 'medium' (10-30min), 'long' (>30min)"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Recommend science/tech videos based on topic."""
    videos_path = Path(data_dir) / "videos.csv"
    if videos_path.exists():
        videos_df = pd.read_csv(videos_path)
    else:
        videos_df = get_youtube_content_catalog(200)

    views_path = Path(data_dir) / "user_views.csv"
    if views_path.exists():
        views_df = pd.read_csv(views_path)
    else:
        views_df = generate_user_views(videos_df)

    typer.echo(f"Loaded {len(videos_df)} videos")
    typer.echo(f"Loaded {len(views_df)} viewing records")
    typer.echo(f"\nQuery: '{query}'")

    recommender = CollaborativeFilterRecommender()
    recommender.fit(views_df, videos_df)

    # Parse duration filter
    duration_filter = {
        "short": (0, 600),
        "medium": (600, 1800),
        "long": (1800, float("inf")),
    }.get(duration, (0, float("inf")))

    recs = recommender.recommend_by_topic(query, top_k * 2)

    # Apply duration filter
    filtered = []
    for rec in recs:
        dur = rec["video"]["duration_seconds"]
        if duration_filter[0] < dur <= duration_filter[1]:
            filtered.append(rec)

    if len(filtered) < top_k and len(recs) > len(filtered):
        filtered = recs[:top_k]

    typer.echo(f"\nTop {len(filtered)} recommendations:\n")
    for i, rec in enumerate(filtered[:top_k], 1):
        video = rec["video"]
        typer.echo(f"{i}. {video['title']}")
        typer.echo(f"   Channel: {video['channel']}")
        typer.echo(f"   Topic: {video['topic']} ({video['subtopic']})")
        typer.echo(f"   Duration: {video['duration_seconds'] // 60}min | Views: {video['views']:,} | Likes: {video['likes']:,}")
        typer.echo(f"   Uploaded: {video['uploaded_date']}")
        if video.get("description"):
            typer.echo(f"   {video['description']}")
        typer.echo()


@app.command()
def analyze_channel(
    channel_name: str = typer.Argument(..., help="YouTube channel name"),
):
    """Analyze a science/tech YouTube channel."""
    videos_path = Path("data/synthetic/videos.csv")
    if videos_path.exists():
        videos_df = pd.read_csv(videos_path)
    else:
        videos_df = get_youtube_content_catalog(200)

    channel_videos = videos_df[videos_df["channel"].str.contains(channel_name, case=False)]

    if channel_videos.empty:
        typer.echo(f"Channel '{channel_name}' not found.")
        return

    typer.echo(f"Channel Analysis: {channel_name}")
    typer.echo(f"Total videos: {len(channel_videos)}")
    typer.echo(f"Total views: {channel_videos['views'].sum():,}")
    typer.echo(f"Average views: {channel_videos['views'].mean():.0f}")
    typer.echo(f"Average likes: {channel_videos['likes'].mean():.0f}")
    typer.echo(f"Like ratio: {(channel_videos['likes'].sum() / channel_videos['views'].sum() * 100):.1f}%")

    typer.echo("\nTopics covered:")
    for topic, count in channel_videos["topic"].value_counts().items():
        typer.echo(f"  {topic}: {count} videos")


@app.command()
def list_videos(
    topic_filter: str = typer.Option("", "--topic", help="Filter by topic"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List available videos."""
    videos_path = Path(data_dir) / "videos.csv"
    if videos_path.exists():
        videos_df = pd.read_csv(videos_path)
    else:
        videos_df = get_youtube_content_catalog(200)

    if topic_filter:
        videos_df = videos_df[videos_df["topic"].str.contains(topic_filter, case=False)]

    typer.echo(f"Videos ({len(videos_df)} total):\n")
    for _, video in videos_df.iterrows():
        typer.echo(f"  {video['id']}: {video['title'][:50]}... ({video['channel']})")
        typer.echo(f"      {video['topic']} | {video['duration_seconds'] // 60}min | {video['views']:,} views")


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_videos: int = typer.Option(200, "--count", "-n", help="Number of videos"),
    n_users: int = typer.Option(100, "--users", help="Number of users"),
):
    """Generate synthetic video and user data."""
    videos_df = get_youtube_content_catalog(n_videos)
    views_df = generate_user_views(videos_df, n_users)

    videos_path = Path(output_dir) / "videos.csv"
    views_path = Path(output_dir) / "user_views.csv"

    videos_df.to_csv(videos_path, index=False)
    views_df.to_csv(views_path, index=False)

    typer.echo(f"Generated {n_videos} videos to: {videos_path}")
    typer.echo(f"Generated {len(views_df)} user views to: {views_path}")


if __name__ == "__main__":
    app()

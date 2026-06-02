"""JobPromis — Hybrid job matching platform using BM25 + semantic reranking.

Combines keyword-based BM25 retrieval with semantic cross-encoder reranking
for optimal job recommendations.

Usage:
    CLI:      python cli.py recommend "ML engineer, remote, $150k" --location San Francisco
    Streamlit: streamlit run streamlit_app.py
    Notebook:  jupyter notebooks/job_matching_example.ipynb
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="JobPromis: Hybrid job matching with BM25 + reranking.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class JobPosting:
    """Represents a job posting."""
    id: str
    title: str
    company: str
    location: str
    location_type: str  # remote, hybrid, onsite
    salary_min: int
    salary_max: int
    salary_currency: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_level: str  # entry, mid, senior, lead
    job_type: str  # full_time, contract, part_time
    description: str
    posted_date: str


@dataclass
class CandidateProfile:
    """Represents a job seeker's profile."""
    current_role: str
    target_role: str
    location: str
    desired_salary: int
    skills: List[str]
    experience_years: int
    preferences: Dict[str, str]


# ============================================================================
# JOB CATALOG
# ============================================================================

def get_synthetic_job_catalog(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic job postings."""
    np.random.seed(seed)

    titles = [
        "Machine Learning Engineer", "Data Scientist", "Software Engineer",
        "AI Research Scientist", "Computer Vision Engineer", "NLP Engineer",
        "MLOps Engineer", "Data Engineer", "Research Scientist", "AI Engineer",
        "Deep Learning Engineer", "Robotics Engineer", "Applied Scientist",
        "Staff Machine Learning Engineer", "Principal Data Scientist",
        "Senior Software Engineer", "Lead AI Engineer", "Research Engineer"
    ]

    companies = [
        "TechCorp", "DataAI Inc", "ML Solutions", "DeepLearn Systems",
        "Neural Networks Co", "AI First", "Cognitive Computing", "Smart Algorithms",
        "Future Labs", "Intelligent Systems", "Machine Intellect", "DataDriven",
        "Algorithmics", "NeuroTech", "AI Dynamics", "SmartAI", "TechGiant",
        "StartupXYZ", "InnovateAI", "QuantumML"
    ]

    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Boston, MA",
        "Austin, TX", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
        "Remote", "Hybrid - San Francisco", "Hybrid - New York", "London, UK",
        "Berlin, Germany", "Toronto, Canada", "Singapore"
    ]

    skills = [
        "python", "tensorflow", "pytorch", "scikit-learn", "sql", "aws",
        "gcp", "azure", "kubernetes", "docker", "spark", "hadoop",
        "nlp", "computer_vision", "reinforcement_learning", "transformers",
        "llm", "mlops", "data_engineering", "deep_learning", "statistics",
        "mathematics", "algorithms", "data_structures", "git", "ci_cd"
    ]

    experience_levels = ["entry", "mid", "senior", "lead", "principal"]
    job_types = ["full_time", "contract", "part_time"]

    descriptions = [
        "Join our AI team to build cutting-edge machine learning systems. "
        "Work on scalable models, collaborate with researchers, and ship products "
        "that impact millions of users.",

        "We're looking for a data scientist to drive insights from data. "
        "Build predictive models, conduct A/B tests, and work cross-functionally "
        "to improve our product.",

        "Build and deploy ML models at scale. Work with MLOps tools, "
        "optimize inference, and ensure reliable model serving in production.",

        "Conduct research on state-of-the-art AI algorithms. Publish papers, "
        "collaborate with academic partners, and translate research into products.",
    ]

    jobs = []
    for i in range(n):
        title = titles[i % len(titles)]
        company = companies[i % len(companies)]

        # Location type based on location string
       # Location type based on location string
        loc = np.random.choice(locations)
        if loc == "Remote":
            location_type = "remote"
        elif "Hybrid" in loc:
            location_type = "hybrid"
        else:
            location_type = "onsite"

        # Select skills
        n_required = np.random.randint(5, 12)
        n_preferred = np.random.randint(3, 7)
        all_skills = list(skills)
        np.random.shuffle(all_skills)

        salary_base = {"Machine Learning Engineer": 150000, "Data Scientist": 140000,
                       "Software Engineer": 130000, "AI Research Scientist": 180000,
                       "MLOps Engineer": 160000, "Data Engineer": 135000}[title[:len("Data Scientist")]]
        salary_min = int(salary_base * np.random.uniform(0.8, 1.0))
        salary_max = int(salary_base * np.random.uniform(1.2, 1.6))

      jobs.append({
            "id": f"j{i+1:03d}",
            "title": title,
            "company": company,
            "location": loc,
            "location_type": location_type,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": "USD",
            "required_skills": all_skills[:n_required],
            "preferred_skills": all_skills[n_required:n_required+n_preferred],
            "experience_level": np.random.choice(experience_levels),
            "job_type": np.random.choice(job_types, p=[0.7, 0.2, 0.1]),
            "description": np.random.choice(descriptions),
            "posted_date": f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
        })

    return pd.DataFrame(jobs)


# ============================================================================
# BM25 RETRIEVAL (Simplified Implementation)
# ============================================================================

class BM25Retriever:
    """Simplified BM25 keyword-based retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.idf = {}
        self.documents = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents: List[Dict], text_field: str = "search_text"):
        """Build BM25 index."""
        self.documents = documents

        # Build search text for each document
        for doc in documents:
            doc["search_text"] = " ".join([
                doc.get("title", ""),
                doc.get("company", ""),
                doc.get("description", ""),
                " ".join(doc.get("required_skills", [])),
                " ".join(doc.get("preferred_skills", [])),
            ])

        # Compute document lengths
        doc_lengths = []
        for doc in documents:
            tokens = self._tokenize(doc["search_text"])
            self.doc_lengths[doc["id"]] = len(tokens)
            doc_lengths.append(len(tokens))

        self.avg_doc_length = np.mean(doc_lengths) if doc_lengths else 1

        # Compute IDF
        vocab = defaultdict(int)
        for doc in documents:
            tokens = set(self._tokenize(doc["search_text"]))
            for token in tokens:
                vocab[token] += 1

        n_docs = len(documents)
        for token, count in vocab.items():
            self.idf[token] = np.log((n_docs - count + 0.5) / (count + 0.5) + 1)

    def score_query(self, query: str) -> List[Tuple[Dict, float]]:
        """Score documents for a query."""
        query_tokens = self._tokenize(query)
        scores = []

        for doc in self.documents:
            doc_tokens = self._tokenize(doc["search_text"])
            doc_len = self.doc_lengths[doc["id"]]

            # BM25 score
            score = 0.0
            for token in query_tokens:
                if token in self.idf and token in doc_tokens:
                    tf = doc_tokens.count(token) / (doc_len * self.b / self.avg_doc_length + 1)
                    score += self.idf[token] * tf * (self.k1 + 1) / (tf + self.k1)

            scores.append((doc, score))

        return scores


# ============================================================================
# SEMANTIC RERANKING (Simplified)
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


def generate_semantic_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Generate pseudo-embedding for text."""
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(dim).astype(np.float32) * 0.1


class SemanticReranker:
    """Cross-encoder style reranking using semantic similarity."""

    def __init__(self):
        self.embeddings = {}

    def fit(self, documents: List[Dict]):
        """Generate embeddings for documents."""
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('company', '')} {doc.get('description', '')}"
            self.embeddings[doc["id"]] = generate_semantic_embedding(text)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict, float]],
        top_k: int = 20,
    ) -> List[Tuple[Dict, float, float]]:
        """Rerank candidates based on semantic similarity to query."""
        query_embedding = generate_semantic_embedding(query)

        # Score candidates
        reranked = []
        for doc, bm25_score in candidates[:top_k * 2]:
            doc_embedding = self.embeddings.get(doc["id"])
            if doc_embedding is None:
                doc_embedding = generate_semantic_embedding(f"{doc.get('title', '')} {doc.get('description', '')}")

            semantic_score = cosine_similarity(query_embedding, doc_embedding)

            # Hybrid score
            hybrid_score = 0.3 * bm25_score + 0.7 * semantic_score

            reranked.append((doc, bm25_score, semantic_score, hybrid_score))

        reranked.sort(key=lambda x: x[3], reverse=True)
        return reranked


# ============================================================================
# SKILLS GAP ANALYSIS
# ============================================================================

def analyze_skills_gap(
    candidate_skills: List[str],
    job_required_skills: List[str],
    job_preferred_skills: List[str],
) -> Dict:
    """Analyze skills gap between candidate and job."""
    candidate_set = set(s.lower() for s in candidate_skills)
    required_set = set(s.lower() for s in job_required_skills)
    preferred_set = set(s.lower() for s in job_preferred_skills)

    missing_required = required_set - candidate_set
    missing_preferred = preferred_set - candidate_set
    matched_required = candidate_set & required_set
    matched_preferred = candidate_set & preferred_set

    # Compute coverage
    required_coverage = len(matched_required) / len(required_set) if required_set else 1.0
    total_coverage = (len(matched_required) + 0.5 * len(matched_preferred)) / (len(required_set) + len(preferred_set))

    return {
        "matched_required": list(matched_required),
        "missing_required": list(missing_required),
        "matched_preferred": list(matched_preferred),
        "missing_preferred": list(missing_preferred),
        "required_coverage": round(required_coverage, 2),
        "total_coverage": round(total_coverage, 2),
    }


# ============================================================================
# HYBRID JOB RECOMMENDER
# ============================================================================

class HybridJobRecommender:
    """Hybrid job matching using BM25 + semantic reranking."""

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.reranker = SemanticReranker()
        self.jobs = []

    def load_jobs(self, jobs_df: pd.DataFrame):
        """Load job catalog."""
        self.jobs = [JobPosting(**row.to_dict()) for _, row in jobs_df.iterrows()]
        self.bm25.fit([j.__dict__ for j in self.jobs])
        self.reranker.fit([j.__dict__ for j in self.jobs])

    def recommend(
        self,
        candidate: CandidateProfile,
        top_k: int = 10,
    ) -> List[Dict]:
        """Recommend jobs for a candidate."""
        # Build query from candidate profile
        query_parts = [
            candidate.current_role,
            candidate.target_role,
            " ".join(candidate.skills[:10]),
            candidate.location,
        ]
        query = " ".join(query_parts)

        # BM25 retrieval
        bm25_results = self.bm25.score_query(query)

        # Semantic reranking
        reranked = self.reranker.rerank(query, bm25_results, top_k * 2)

        # Score jobs
        scores = []
        for doc, bm25_score, semantic_score, hybrid_score in reranked:
            job = JobPosting(**doc)

            # Location match bonus
            location_score = 0.0
            if candidate.location.lower() in job.location.lower():
                location_score = 0.2
            elif job.location_type == "remote":
                location_score = 0.15
            elif job.location_type == "hybrid" and candidate.location.lower() in job.location.lower():
                location_score = 0.1

            # Salary match
            salary_score = 0.0
            if job.salary_max >= candidate.desired_salary:
                salary_score = 0.15
            elif job.salary_min >= candidate.desired_salary * 0.9:
                salary_score = 0.08

            # Skills coverage
            skills_analysis = analyze_skills_gap(candidate.skills, job.required_skills, job.preferred_skills)
            skills_score = skills_analysis["total_coverage"]

            final_score = hybrid_score * 0.5 + location_score + salary_score + skills_score * 0.3

            scores.append({
                "job": job,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "final_score": final_score,
                "skills_analysis": skills_analysis,
            })

        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores[:top_k]


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def recommend(
    current_role: str = typer.Argument(..., help="Current job title or role"),
    target_role: str = typer.Option("", "--target", "-t", help="Target job title (optional)"),
    location: str = typer.Option("", "--location", "-l", help="Preferred location"),
    skills: str = typer.Option("", "--skills", "-s", help="Current skills (comma-separated)"),
    desired_salary: int = typer.Option(150000, "--salary", help="Desired salary"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of recommendations"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """Recommend jobs based on candidate profile."""
    jobs_path = Path(data_dir) / "jobs.csv"
    if jobs_path.exists():
        jobs_df = pd.read_csv(jobs_path)
    else:
        jobs_df = get_synthetic_job_catalog(150)

    typer.echo(f"Loaded {len(jobs_df)} job postings")

    # Parse skills
    candidate_skills = [s.strip().lower() for s in skills.split(",")] if skills else ["python", "machine_learning", "data_science"]

    candidate = CandidateProfile(
        current_role=current_role,
        target_role=target_role or current_role,
        location=location or "San Francisco",
        desired_salary=desired_salary,
        skills=candidate_skills,
        experience_years=5,
        preferences={"location_type": "remote"} if "remote" in location.lower() else {},
    )

    typer.echo(f"\nProfile:")
    typer.echo(f"  Current: {candidate.current_role}")
    typer.echo(f"  Target: {candidate.target_role}")
    typer.echo(f"  Location: {candidate.location}")
    typer.echo(f"  Skills: {', '.join(candidate.skills[:5])}...")
    typer.echo(f"  Desired salary: ${candidate.desired_salary:,}")

    recommender = HybridJobRecommender()
    recommender.load_jobs(jobs_df)

    recs = recommender.recommend(candidate, top_k)

    typer.echo(f"\nTop {len(recs)} job recommendations:\n")
    for i, rec in enumerate(recs, 1):
        job = rec["job"]
        typer.echo(f"{i}. {job.title} at {job.company}")
        typer.echo(f"   Location: {job.location} ({job.location_type})")
        typer.echo(f"   Salary: ${job.salary_min:,} - ${job.salary_max:,} {job.salary_currency}")
        typer.echo(f"   Experience: {job.experience_level}")
        typer.echo(f"   Skills match: {rec['skills_analysis']['required_coverage']*100:.0f}% required, {rec['skills_analysis']['total_coverage']*100:.0f}% total")
        if rec["skills_analysis"]["missing_required"]:
            typer.echo(f"   Missing: {', '.join(rec['skills_analysis']['missing_required'][:3])}")
        typer.echo(f"   Score: {rec['final_score']:.3f}")
        typer.echo()


@app.command()
def skills_gap(
    target_role: str = typer.Argument(..., help="Target job title"),
    current_skills: str = typer.Option("", "--skills", "-s", help="Current skills (comma-separated)"),
):
    """Analyze skills gap to reach target role."""
    jobs_path = Path("data/synthetic/jobs.csv")
    if jobs_path.exists():
        jobs_df = pd.read_csv(jobs_path)
    else:
        jobs_df = get_synthetic_job_catalog(150)

    # Find jobs matching target role
    target_jobs = jobs_df[
        jobs_df["title"].str.contains(target_role, case=False, na=False)
    ]

    if target_jobs.empty:
        typer.echo(f"No jobs found for '{target_role}'")
        return

    # Aggregate required skills from target jobs
    all_required = []
    all_preferred = []
    for _, job in target_jobs.iterrows():
        all_required.extend(job.get("required_skills", []))
        all_preferred.extend(job.get("preferred_skills", []))

    # Count skill frequency
    required_freq = defaultdict(int)
    preferred_freq = defaultdict(int)
    for skill in all_required:
        required_freq[skill] += 1
    for skill in all_preferred:
        preferred_freq[skill] += 1

    # Parse current skills
    current_skills_set = set(s.strip().lower() for s in current_skills.split(",")) if current_skills else set()

    # Analyze gaps
    all_required_set = set(required_freq.keys())
    missing = all_required_set - current_skills_set

    typer.echo(f"Skills gap analysis for '{target_role}':")
    typer.echo()
    typer.echo("Required skills (by frequency):")
    for skill, count in sorted(required_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        status = "✓" if skill in current_skills_set else "✗"
        typer.echo(f"  {status} {skill} (required in {count} jobs)")

    if missing:
        typer.echo("\nTop skills to develop:")
        for skill in sorted(missing, key=lambda s: required_freq[s], reverse=True)[:5]:
            typer.echo(f"  - {skill}")


@app.command()
def list_jobs(
    role_filter: str = typer.Option("", "--role", help="Filter by role"),
    location_filter: str = typer.Option("", "--location", help="Filter by location"),
    data_dir: str = typer.Option("data/synthetic", "--data-dir", "-d", help="Data directory"),
):
    """List available job postings."""
    jobs_path = Path(data_dir) / "jobs.csv"
    if jobs_path.exists():
        jobs_df = pd.read_csv(jobs_path)
    else:
        jobs_df = get_synthetic_job_catalog(150)

    if role_filter:
        jobs_df = jobs_df[jobs_df["title"].str.contains(role_filter, case=False)]
    if location_filter:
        jobs_df = jobs_df[jobs_df["location"].str.contains(location_filter, case=False)]

    typer.echo(f"Jobs ({len(jobs_df)} total):\n")
    for _, job in jobs_df.iterrows():
        typer.echo(f"  {job['id']}: {job['title']} at {job['company']}")
        typer.echo(f"      {job['location']} | ${job['salary_min']:,}-{job['salary_max']:,}")
        typer.echo(f"      Skills: {', '.join(job['required_skills'][:4])}")


@app.command()
def generate_data(
    output_dir: str = typer.Option("data/synthetic", "--output-dir", "-o", help="Output directory"),
    n_jobs: int = typer.Option(150, "--count", "-n", help="Number of jobs"),
):
    """Generate synthetic job data."""
    jobs_df = get_synthetic_job_catalog(n_jobs)
    filepath = Path(output_dir) / "jobs.csv"
    jobs_df.to_csv(filepath, index=False)
    typer.echo(f"Generated {n_jobs} jobs to: {filepath}")


if __name__ == "__main__":
    app()

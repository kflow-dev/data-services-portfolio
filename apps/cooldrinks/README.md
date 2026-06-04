# CoolDrinks - Context-Aware Beverage Recommender

SOTA context-aware beverage recommender using **Transformer-based sequential recommendation (SASRec)**, **Multi-modal fusion**, and **Bandit-based exploration-exploitation (LinUCB)**.

## Overview

**Problem**: Personalized beverage recommendation based on context (weather, time of day, occasion, taste preferences).

**Approach**: Hybrid ensemble combining:
- **SASRec**: Models sequential preference evolution using self-attention
- **Multi-Modal Fusion**: Fuses drink content features with context embeddings
- **LinUCB**: Balances exploration vs. exploitation for diverse recommendations
- **Cold-Start Fallback**: Content + popularity hybrid for new items

**Libraries**: `tensorflow`, `tensorflow-recommenders`, `implicit`, `scikit-learn`, `pandas`, `numpy`, `typer`, `streamlit`

---

## Quick Start

### CLI Usage

```bash
# Get recommendations
python cli.py recommend --weather sunny --hour 14 --occasion casual

# With taste preferences
python cli.py recommend -w rainy -h 20 -o celebration -b 0.7 -s 0.3 -t 0.5

# List available drinks
python cli.py list-drinks --type beer --min-abv 4

# Generate synthetic data
python cli.py generate-data --drinks 120 --interactions 10000 --users 500

# View statistics
python cli.py stats
```

### Streamlit UI

```bash
# Start Streamlit server
streamlit run streamlit_app.py --server.port 3000
```

Navigate to `http://localhost:3000` for interactive UI with:
- Context selector (weather, time, occasion)
- Taste preference sliders (bitterness, sweetness, strength)
- Top-K recommendation display with explanations
- Drink details and flavor profile visualization

### Docker

```bash
# Build image
docker build -t cooldrinks .

# Run container
docker run -p 3000:3000 cooldrinks
```

---

## CLI Reference

### `recommend` - Get Context-Aware Recommendations

```bash
python cli.py recommend \
  --user U001 \
  --weather sunny \
  --hour 14 \
  --occasion casual \
  --bitterness 0.5 \
  --sweetness 0.5 \
  --strength 0.5 \
  --top 5
```

**Parameters**:
- `--user, -u`: User identifier (default: U001)
- `--weather, -w`: Weather condition (sunny, rainy, cloudy, snowy, stormy)
- `--hour, -h`: Hour of day 6-23 (default: 14)
- `--occasion, -o`: Occasion type (casual, celebration, pairing, recovery, social, business)
- `--bitterness, -b`: Bitterness preference [0-1] (default: 0.5)
- `--sweetness, -s`: Sweetness preference [0-1] (default: 0.5)
- `--strength, -t`: Strength preference [0-1] (default: 0.5)
- `--top, -k`: Number of recommendations (default: 5)

### `list-drinks` - Browse Drink Catalog

```bash
python cli.py list-drinks --type beer --min-abv 4 --limit 20
```

**Filters**:
- `--type, -t`: Filter by type (beer, wine, coffee, tea, cocktail, non-alcoholic)
- `--min-abv`: Minimum ABV
- `--max-abv`: Maximum ABV
- `--style`: Filter by style
- `--seasonality`: Filter by seasonality
- `--limit, -l`: Max drinks to display

### `generate-data` - Generate Synthetic Dataset

```bash
python cli.py generate-data --drinks 120 --interactions 10000 --users 500
```

**Parameters**:
- `--drinks, -n`: Number of drinks (default: 120)
- `--interactions, -i`: Number of interactions (default: 10000)
- `--users, -u`: Number of unique users (default: 500)
- `--output-dir, -o`: Output directory (default: data/synthetic)

### `stats` - View Dataset Statistics

```bash
python cli.py stats
```

---

## ML Architecture

### 1. SASRec (Sequence-Aware Session-based Recommender)

**Architecture**:
- Input: User session history `[d_{t-3}, d_{t-2}, d_{t-1}]`
- Embedding: Drink ID → `d_model` (64 dimensions)
- Transformer: 2-layer self-attention, 4 heads
- Output: Next drink probability distribution
- Loss: Cross-entropy on next item prediction

**Purpose**: Models sequential preference evolution from user interaction history.

### 2. Multi-Modal Fusion

**Architecture**:
- Drink embedding: Content features → dense(64) → tanh
- Context embedding: One-hot(weather, time, occasion) → dense(64)
- Cross-attention: Models interaction between drink and context
- Output: Ranking scores for all drinks
- Loss: Pairwise ranking loss (BPR)

**Purpose**: Combines drink content with contextual signals for context-aware scoring.

### 3. LinUCB Bandit

**Algorithm**:
- For each drink `k`: Maintain `A_k` (information matrix), `b_k` (sufficient statistic)
- Linear model: `u_k = A_k^(-1) b_k`
- Confidence width: `c_k = sqrt(x_k^T A_k^(-1) x_k)`
- UCB score: `s_k = u_k^T x_k + alpha * c_k`

**Purpose**: Balances exploration (trying new drinks) vs. exploitation (familiar favorites).

### 4. Hybrid Scoring

**Combined Score**:
```
Score = w_sasrec * sasrec + w_fusion * fusion + w_linucb * linucb

with w_sasrec = 0.3, w_fusion = 0.5, w_linucb = 0.2
```

**Cold-Start Handling**: Hybrid content + popularity fallback for new drinks.

---

## Data Structure

### Drink Catalog

```csv
drink_id,name,type,style,abv,bitterness,sweetness,carbonation,seasonality,origin
D001,IPA 1,beer,ipa,6.2,75,30,3.2,summer,usa
D002,Cold Brew 1,coffee,cold_brew,0,80,40,0.5,any,germany
...
```

**Attributes**:
- `type`: beer, wine, coffee, tea, cocktail, non-alcoholic
- `style`: IPA, stout, pilsner, espresso, cold_brew, red, white, etc.
- `abv`: Alcohol by volume (0-40% for alcoholic, 0 for non-alcoholic)
- `bitterness`: Bitterness scale [0-100]
- `sweetness`: Sweetness scale [0-100]
- `carbonation`: Carbonation level [0-5]
- `strength`: Normalized strength [0-1]
- `seasonality`: Preferred season (summer, winter, spring, fall, any)

### Context Features

- **Weather**: sunny, rainy, cloudy, snowy, stormy
- **Time Period**: morning (6-12), afternoon (12-18), evening (18-24)
- **Occasion**: casual, celebration, pairing, recovery, social, business

### Interaction Logs

```csv
user_id,drink_id,weather,temperature,time_period,hour,dayofweek,occasion,interaction_type,value
U001,D042,sunny,28,afternoon,14,3,casual,rate,4.5
U001,D042,sunny,28,afternoon,14,3,casual,view,0
...
```

---

## Evaluation Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain for ranking quality
- **Precision@k**: Fraction of recommended items that are relevant
- **Recall@k**: Fraction of relevant items that are recommended
- **Diversity**: Coverage of different drink types/styles

---

## Jupyter Notebook

For detailed ML exploration, see `notebooks/advanced_context_aware_recommender.ipynb`:

1. Problem framing & context-aware recommendation
2. Data exploration (120 drinks, 10K interactions)
3. SASRec implementation (sequence modeling)
4. Multi-modal fusion architecture
5. LinUCB exploration-exploitation analysis
6. Model evaluation (NDCG, precision@k, diversity)
7. Ablation studies (component contribution)
8. Interactive demo

---

## Library Requirements

```txt
streamlit>=1.36
requests>=2.32
pandas>=2.2
numpy>=1.26
typer>=0.12
python-dotenv>=1.0
tensorflow>=2.15
tensorflow-recommenders>=0.7
implicit>=0.7
transformers>=4.35
```

---

## Known Questions

**a. User base size**: Synthetic data assumes ~500 users with ~10K interactions for demonstration.

**b. Explicit vs implicit feedback**: Using implicit feedback (views, ratings 1-5) - standard for recommendation systems.

**c. Cold-start handling**: Hybrid content-based + popularity fallback for new drinks with no interaction history.

---

## License

Part of the Data Services Portfolio project.

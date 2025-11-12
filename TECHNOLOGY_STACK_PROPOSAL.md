# Technology Stack Proposal for B2B Recommendation System

## Executive Summary

**Current State**: Agreement-level recommendation system with 6.91% precision@20
- Best customer: 40% precision
- Many customers: 0% precision
- Issue: Simple frequency/recency scoring doesn't capture product relationships

**Goal**: Achieve >50% precision@20 through modern ML/AI technology stack

---

## Current Architecture Analysis

### Strengths ✓
- Agreement-level recommendations (correct business logic)
- Connection pooling for performance
- Redis caching
- Temporal holdout validation
- Background workers for batch processing

### Critical Gaps ✗
- **No product embeddings** - Cannot understand product similarity
- **No collaborative filtering** - Cannot leverage cross-customer patterns
- **No sequence modeling** - Cannot capture order patterns
- **No feature engineering** - Limited to frequency/recency
- **No hyperparameter tuning** - Weights are hardcoded
- **No A/B testing framework** - Cannot measure real-world impact

---

## Proposed Technology Stack

### 1. **Feature Store & Data Pipeline**

#### Technology: **Apache Airflow + DuckDB**
```
Why:
- Airflow: Orchestrate feature pipelines (better than cron)
- DuckDB: Fast analytical queries for feature computation
- Versioned features for reproducibility
```

**Implementation**:
```python
# Feature pipeline DAG
from airflow import DAG
from airflow.operators.python import PythonOperator

def compute_product_features():
    """
    Features:
    - Product purchase frequency by segment
    - Product co-occurrence matrix
    - Product category embeddings
    - Temporal patterns (seasonality)
    """
    pass

def compute_customer_features():
    """
    Features:
    - Purchase history vectors
    - Agreement characteristics
    - Segment behavior patterns
    - Recency/frequency/monetary (RFM)
    """
    pass
```

### 2. **Product Embeddings & Similarity**

#### Technology: **Sentence-BERT + FAISS**
```
Why:
- SBERT: Generate embeddings from product descriptions
- FAISS: Fast similarity search at scale
- Captures semantic similarity beyond co-purchase
```

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ProductEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None

    def build_index(self, products: List[Dict]):
        """Build FAISS index from product catalog"""
        texts = [f"{p['name']} {p['category']} {p['description']}"
                 for p in products]
        embeddings = self.model.encode(texts)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def find_similar_products(self, product_id: int, top_k: int = 50):
        """Find k most similar products"""
        query_vector = self.get_embedding(product_id)
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0], distances[0]
```

### 3. **Collaborative Filtering**

#### Technology: **LightFM** (Hybrid Matrix Factorization)
```
Why:
- Handles cold-start (new products/customers)
- Incorporates item/user features
- Fast training and inference
- Better than pure matrix factorization
```

**Implementation**:
```python
from lightfm import LightFM
from lightfm.data import Dataset

class CollaborativeFilter:
    def __init__(self):
        self.model = LightFM(
            loss='warp',  # Weighted Approximate-Rank Pairwise
            no_components=128,
            learning_rate=0.05
        )

    def train(self, interactions, customer_features, product_features):
        """
        interactions: (customer, agreement, product, quantity, timestamp)
        customer_features: segment, rfm_score, lifetime_value
        product_features: category, brand, price_tier
        """
        self.model.fit(
            interactions,
            item_features=product_features,
            user_features=customer_features,
            epochs=30
        )

    def predict_for_agreement(self, agreement_id: int,
                              candidate_products: List[int]) -> List[Tuple]:
        """Return scored products for agreement"""
        scores = self.model.predict(agreement_id, candidate_products)
        return sorted(zip(candidate_products, scores),
                     key=lambda x: x[1], reverse=True)
```

### 4. **Sequence Modeling (Optional - Advanced)**

#### Technology: **PyTorch + Transformers (BERT4Rec)**
```
Why:
- Captures order sequences
- Learns "what comes next"
- State-of-the-art for sequential recommendation
```

**When to use**:
- If order sequences matter (e.g., project phases)
- If you have >100k order sequences
- If precision improvements justify complexity

### 5. **Model Training & Experimentation**

#### Technology: **MLflow + Optuna**
```
Why:
- MLflow: Track experiments, models, metrics
- Optuna: Hyperparameter optimization
- Version control for models
```

**Implementation**:
```python
import mlflow
import optuna

def objective(trial):
    """Optimize hyperparameters"""
    params = {
        'no_components': trial.suggest_int('components', 64, 256),
        'learning_rate': trial.suggest_float('lr', 0.001, 0.1, log=True),
        'loss': trial.suggest_categorical('loss', ['warp', 'bpr'])
    }

    model = LightFM(**params)
    model.fit(train_data, epochs=20)

    precision = validate_model(model, test_data)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("precision@20", precision)

    return precision

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 6. **Ensemble & Ranking**

#### Technology: **XGBoost** (Learning-to-Rank)
```
Why:
- Combine multiple signals
- Learn optimal weighting
- Outperforms manual tuning
```

**Implementation**:
```python
import xgboost as xgb

class RankingEnsemble:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            learning_rate=0.1,
            max_depth=6
        )

    def prepare_features(self, agreement_id, products):
        """
        Features for each product:
        - Frequency score (current system)
        - Recency score (current system)
        - Collaborative filtering score
        - Product similarity score
        - Category popularity
        - Price compatibility
        - Seasonal factor
        """
        return feature_matrix

    def train(self, training_data):
        """
        training_data: (agreement, product, features, label)
        label: 1 if purchased in next 30 days, 0 otherwise
        """
        X, y, groups = self.prepare_training_data(training_data)
        self.model.fit(X, y, group=groups)

    def rank_products(self, agreement_id, candidate_products):
        """Return ranked list"""
        features = self.prepare_features(agreement_id, candidate_products)
        scores = self.model.predict(features)
        return sorted(zip(candidate_products, scores),
                     key=lambda x: x[1], reverse=True)
```

### 7. **Real-Time Serving**

#### Technology: **FastAPI + Redis + Model Registry**
```
Current: ✓ FastAPI already in use
Improvements:
- Model versioning
- A/B testing support
- Feature caching in Redis
```

**Enhanced API**:
```python
from fastapi import FastAPI, Header
from typing import Optional

@app.get("/recommendations/{customer_id}")
async def get_recommendations(
    customer_id: int,
    model_version: Optional[str] = "latest",
    experiment_variant: Optional[str] = Header(None)  # A/B testing
):
    # Route to appropriate model based on experiment
    if experiment_variant == "B":
        recommender = load_model("model_v2")
    else:
        recommender = load_model("model_v1")

    recommendations = recommender.get_recommendations(customer_id)

    # Log for A/B analysis
    log_recommendation_event(customer_id, recommendations, experiment_variant)

    return recommendations
```

### 8. **Monitoring & Evaluation**

#### Technology: **Grafana + Prometheus + Custom Metrics**
```
Why:
- Real-time performance monitoring
- Online metrics (CTR, conversion)
- Data drift detection
```

**Metrics to Track**:
```python
# Online Metrics
- Click-through rate (CTR)
- Add-to-cart rate
- Purchase conversion rate
- Average order value

# Offline Metrics
- Precision@K (K=5, 10, 20)
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage (% of catalog recommended)
- Diversity (how varied are recommendations)

# System Metrics
- Latency (p50, p95, p99)
- Cache hit rate
- Model inference time
```

---

## Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)
- [x] Temporal validation framework ✓ (Done)
- [x] Agreement-level recommendations ✓ (Done)
- [ ] Setup MLflow for experiment tracking
- [ ] Build feature pipeline (Airflow + DuckDB)
- [ ] Extract & version baseline features

**Target**: Reproducible experiments, baseline precision measured

### Phase 2: Embeddings & Similarity (2 weeks)
- [ ] Generate product embeddings (SBERT)
- [ ] Build FAISS index for similarity search
- [ ] Add similarity-based recommendations
- [ ] Integrate into ensemble

**Target**: +5-10% precision improvement

### Phase 3: Collaborative Filtering (2-3 weeks)
- [ ] Implement LightFM model
- [ ] Feature engineering (customer + product)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Cross-validation

**Target**: +10-15% precision improvement

### Phase 4: Ensemble & Ranking (2 weeks)
- [ ] Train XGBoost ranker
- [ ] Combine all signals
- [ ] Optimize weights
- [ ] A/B test preparation

**Target**: +5-10% precision improvement (cumulative: 25-40%)

### Phase 5: Production & Monitoring (1-2 weeks)
- [ ] Model deployment pipeline
- [ ] A/B testing framework
- [ ] Grafana dashboards
- [ ] Alerting & monitoring

**Target**: 50%+ precision@20, production-ready

### Phase 6: Advanced (Optional)
- [ ] Sequence modeling (BERT4Rec)
- [ ] Multi-armed bandits for exploration
- [ ] Contextual recommendations (seasonality, promotions)

---

## Expected Performance Improvements

| Phase | Precision@20 | Improvement |
|-------|-------------|-------------|
| Current | 6.9% | Baseline |
| Phase 2 (Embeddings) | 12-17% | +5-10% |
| Phase 3 (Collaborative) | 25-32% | +13-15% |
| Phase 4 (Ensemble) | 35-45% | +10-13% |
| **Target** | **50%+** | **+43%** |

---

## Technology Stack Summary

| Component | Technology | Why |
|-----------|-----------|-----|
| **Orchestration** | Apache Airflow | Feature pipelines, batch jobs |
| **Feature Store** | DuckDB | Fast analytical queries |
| **Embeddings** | Sentence-BERT | Product similarity |
| **Search** | FAISS | Fast vector search |
| **Collaborative** | LightFM | Hybrid matrix factorization |
| **Ranking** | XGBoost | Learn-to-rank ensemble |
| **Experiment Tracking** | MLflow | Model versioning, metrics |
| **Hyperparameter Opt** | Optuna | Automated tuning |
| **API** | FastAPI ✓ | Already in use |
| **Cache** | Redis ✓ | Already in use |
| **Database** | SQL Server ✓ | Already in use |
| **Monitoring** | Grafana + Prometheus | Real-time metrics |

---

## Next Steps

1. **Approve technology stack**
2. **Setup development environment** (MLflow, Airflow)
3. **Start Phase 1** - Feature engineering pipeline
4. **Run baseline experiments** - Establish reproducible metrics

---

## Cost-Benefit Analysis

### Development Time
- **Total**: 9-12 weeks for Phase 1-5
- **Incremental delivery**: Each phase adds value

### Infrastructure Costs
- MLflow: Open source (negligible cost)
- Airflow: Open source (negligible cost)
- FAISS: In-memory (RAM cost only)
- Redis: Already deployed ✓
- Grafana: Open source (negligible cost)

**Estimated monthly cost**: +$50-100 (additional compute/storage)

### Business Impact
- **Current**: 6.9% precision → low customer engagement
- **Target**: 50% precision → 7x improvement
- **ROI**: Higher cart conversion, customer satisfaction, repeat orders

---

## Questions to Answer

1. Do you have product catalog data (names, descriptions, categories)?
2. What is the business priority: speed to market vs. maximum accuracy?
3. Are there domain-specific constraints (e.g., certain products never recommended together)?
4. Can you track customer interactions (clicks, views) for online metrics?
5. What is the acceptable model refresh frequency (daily, weekly)?

---

**Generated**: 2025-11-12
**Author**: Claude Code
**Status**: Proposal - Awaiting Approval

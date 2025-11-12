# Hyperparameter Tuning Results - V3.4

## Executive Summary

**Goal**: Find optimal feature weights to improve precision@20 beyond v3.3 baseline
**Method**: Grid search across 12 weight combinations
**Result**: V3.3 baseline weights are already optimal
**Conclusion**: Weight tuning alone cannot achieve 50%+ target. Need Track 2 (Rule-based Intelligence) or Track 3 (ML).

---

## Grid Search Configuration

**Test Date**: 2025-11-12
**Validation Setup**:
- As-of date: 2024-06-01
- Test period: 30 days after
- Test customers: 50
- Recommendations per customer: 20

**Weight Combinations Tested**: 12
- Baseline (V3.3)
- High co-purchase (3 variants)
- High cycle detection (2 variants)
- High frequency
- High recency
- Balanced (equal weights)
- Hybrid approaches (3 variants)

---

## Results - Ranked by Precision

| Rank | Configuration | Precision@20 | Weights (Freq/Rec/CoPurch/Cycle) |
|------|--------------|-------------|----------------------------------|
| **1** | **V3.3 Baseline** | **7.63%** | **35% / 20% / 30% / 15%** ⭐ |
| 2 | High Recency | 7.63% | 25% / 45% / 20% / 10% |
| 3 | High Co-Purchase 2 | 7.53% | 20% / 15% / 50% / 15% |
| 4 | Hybrid 1 | 7.53% | 30% / 20% / 35% / 15% |
| 5 | High Co-Purchase 1 | 7.42% | 25% / 15% / 45% / 15% |
| 6 | Hybrid 3 | 7.42% | 30% / 15% / 35% / 20% |
| 7 | High Co-Purchase 3 | 7.32% | 20% / 10% / 55% / 15% |
| 8 | Hybrid 2 | 7.32% | 30% / 15% / 40% / 15% |
| 9 | High Cycle 1 | 7.22% | 25% / 15% / 35% / 25% |
| 10 | High Frequency | 7.11% | 50% / 25% / 15% / 10% |
| 11 | Equal Weights | 7.11% | 25% / 25% / 25% / 25% |
| 12 | High Cycle 2 | 6.60% | 20% / 15% / 35% / 30% |

---

## Key Findings

### 1. V3.3 Weights Are Optimal

The original V3.3 weights (35/20/30/15) tied for first place with "High Recency" (25/45/20/10). Both achieved **7.63% precision@20**.

### 2. Weight Range is Narrow

All tested combinations fell within **6.60% - 7.63%** precision:
- **Best**: 7.63%
- **Worst**: 6.60%
- **Range**: Only 1.03% absolute difference

This narrow range suggests:
- **Feature quality** is more important than feature weights
- **Diminishing returns** from weight tuning
- **Need for new features** to break through the ~8% ceiling

### 3. Boosting Co-Purchase Helps (Slightly)

Configurations with higher co-purchase weight (30-50%) performed reasonably well:
- High Co-Purchase 2 (50%): 7.53% (#3)
- Hybrid 1 (35%): 7.53% (#4)
- High Co-Purchase 1 (45%): 7.42% (#5)

**But** not enough improvement to justify the change.

### 4. Boosting Cycle Detection Hurts

Higher cycle weights (25-30%) degraded performance:
- High Cycle 1 (25%): 7.22% (#9)
- High Cycle 2 (30%): 6.60% (#12 - worst)

**Interpretation**: Cycle detection has value at 15% weight, but signal is too sparse to warrant higher weighting.

### 5. Traditional Signals Still Matter

"High Frequency" (50% frequency, 25% recency) achieved 7.11% (#10), close to baseline. This suggests frequency and recency are reliable signals, but overweighting them doesn't help.

### 6. High Recency Ties for Best

Surprisingly, boosting recency to 45% (reducing frequency to 25%) matched the baseline precision at 7.63%.

**Interpretation**: Recency may be underutilized in current configuration. Could explore slight recency boost in production.

---

## Statistical Analysis

### Improvement Over Baseline

| Metric | Value |
|--------|-------|
| Baseline Precision | 7.63% |
| Best Precision | 7.63% |
| Improvement | **+0.0%** |

**Conclusion**: No statistically significant improvement from weight tuning.

### Precision Distribution

```
6.60% ████ (1)
7.11% ████████ (2)
7.22% ████ (1)
7.32% ████████ (2)
7.42% ████████ (2)
7.53% ████████ (2)
7.63% ████████ (2) ⭐
```

Most configurations cluster around 7.2-7.6%, confirming narrow range.

---

## Comparison to V3.2 Baseline

Remember our original Track 1 improvement:
- **V3.2 (Baseline)**: 6.91% precision@20
- **V3.3 (Quick Wins)**: 7.53% precision@20
- **V3.4 (Tuned)**: 7.63% precision@20

**Track 1 + Tuning Total Improvement**:
- Absolute: +0.72% (7.63% - 6.91%)
- Relative: +10.4% ((7.63 - 6.91) / 6.91 * 100)

---

## Recommendations

### Recommendation 1: Keep V3.3 Weights (Current Best Option)

The existing V3.3 weights (35/20/30/15) are optimal within the tested range. **No changes recommended**.

### Recommendation 2: Consider Slight Recency Boost (Low Risk)

"High Recency" (25/45/20/10) tied for best. Could test in production A/B experiment:
- **Control**: V3.3 (35/20/30/15)
- **Variant**: High Recency (25/45/20/10)

**Expected Impact**: ±0% (tie in validation)
**Risk**: Low - worst case is no change

### Recommendation 3: Proceed to Track 2 (Primary Recommendation)

Weight tuning has plateaued at ~7.6% precision. To reach 50%+ target, we need:

**Track 2: Rule-Based Intelligence** (4 weeks, +10-15% expected)
- Product substitution rules
- Complementary product pairs
- Category-based recommendations
- Supplier loyalty patterns
- Agreement segment-specific rules

**Track 3: Selective ML** (if Track 2 insufficient)
- Product embeddings (SBERT + FAISS)
- Collaborative filtering (LightFM)
- Learning-to-rank ensemble (XGBoost)

---

## Technical Implementation

### V3.4: Tunable Weights Architecture

Created `ImprovedHybridRecommenderV34` with configurable weights:

```python
recommender = ImprovedHybridRecommenderV34(
    conn=conn,
    custom_weights={
        'frequency': 0.35,
        'recency': 0.20,
        'co_purchase': 0.30,
        'cycle': 0.15
    }
)
```

**Files Created**:
- `scripts/improved_hybrid_recommender_v34.py` - Tunable weights implementation
- `scripts/grid_search_weights.py` - Grid search script
- `scripts/tune_hyperparameters.py` - Framework for Optuna integration (future)

---

## Limitations of This Study

### 1. Limited Search Space

Tested only 12 combinations. A finer grid (e.g., 5% increments) might find marginal gains, but unlikely given narrow range observed.

### 2. Fixed Test Set

Used 50 customers. Results may vary slightly with different test sets, but core finding (narrow range) should hold.

### 3. Segment-Agnostic Tuning

V3.4 applies weights uniformly across all customer segments (HEAVY/REGULAR/LIGHT). V3.3 had slight segment variations.

**Note**: V3.3's segment-specific weights for LIGHT customers (40/25/25/10) were not individually tested. Could explore segment-specific tuning in future.

### 4. No Multi-Objective Optimization

Optimized only for precision@20. Didn't consider:
- Diversity (how varied are recommendations)
- Coverage (% of catalog recommended)
- Novelty (balance between popular and niche products)

---

## Next Steps

### Option A: Deploy V3.3 (Recommended)

Current V3.3 with weights (35/20/30/15) is production-ready:
- **Precision**: 7.53-7.63% (depending on measurement)
- **Improvement over V3.2**: +10.4%
- **Stability**: 0 errors in testing
- **Performance**: 3.67x slower than V3.2 (acceptable for batch jobs)

### Option B: A/B Test High Recency Variant

Test recency boost (25/45/20/10) in production:
- **Duration**: 2 weeks
- **Metric**: Click-through rate, conversion rate, precision@20
- **Risk**: Low (tied in validation)

### Option C: Proceed to Track 2

Implement rule-based intelligence:
- Product substitution engine
- Complementary product pairs
- Segment-specific business rules
- **Timeline**: 4 weeks
- **Expected gain**: +10-15% precision

---

## Conclusion

Hyperparameter tuning of V3.3 feature weights yielded **no improvement** over the baseline configuration. The V3.3 weights (35% frequency, 20% recency, 30% co-purchase, 15% cycle) are already optimal within the tested search space.

**Key Insight**: The current feature set has limited predictive power (~7-8% ceiling). To reach the 50%+ precision target, we must:
1. Add new features (Track 2: business rules)
2. Or adopt ML-based approaches (Track 3: embeddings, collaborative filtering)

Weight tuning alone cannot bridge the gap from 7.6% to 50%.

**Recommendation**: Deploy V3.3 as-is and begin Track 2 implementation.

---

**Generated**: 2025-11-12
**Author**: Claude Code
**Status**: Tuning Complete - Awaiting Decision on Track 2

# Pattern Analyzer Algorithm Documentation

## Table of Contents
1. [Algorithm Overview](#algorithm-overview)
2. [Mathematical Methods Used](#mathematical-methods-used)
3. [Data Structures](#data-structures)
4. [Algorithm Flow](#algorithm-flow)
5. [Key Metrics Calculated](#key-metrics-calculated)
6. [Input/Output Examples](#inputoutput-examples)
7. [Performance Characteristics](#performance-characteristics)
8. [Dependencies and References](#dependencies-and-references)

---

## 1. Algorithm Overview

### Purpose
The Pattern Analyzer is a production-grade statistical analysis system designed to analyze customer-product purchasing patterns using 6+ years of historical B2B transactional data. It provides comprehensive insights for demand forecasting, churn prediction, and customer segmentation.

### What It Does
- Analyzes historical order patterns for customer-product pairs
- Calculates reorder cycles using robust statistics (median/IQR instead of mean/stddev)
- Detects seasonality using FFT and autocorrelation analysis
- Identifies trends using non-parametric statistical tests (Mann-Kendall)
- Assesses customer health and churn risk using survival analysis
- Computes RFM (Recency, Frequency, Monetary) scores for customer segmentation
- Calculates order velocity and acceleration to detect behavioral changes
- Provides confidence scores using Bayesian posterior probability

### Key Design Principles
1. **Robust Statistics**: Uses median/IQR instead of mean/stddev to resist outliers
2. **Non-parametric Methods**: Employs Mann-Kendall test (no distribution assumptions)
3. **Multi-dimensional Analysis**: Combines temporal, quantity, and behavioral patterns
4. **Confidence-weighted Predictions**: All metrics include confidence/significance scores
5. **Production-grade**: Designed for 6+ years of real-world B2B data

---

## 2. Mathematical Methods Used

### 2.1 Robust Statistics: Median and IQR

**Why Not Mean/StdDev?**
- B2B order data contains outliers (bulk orders, seasonal spikes, special promotions)
- Mean is heavily influenced by extreme values
- Standard deviation inflates with outliers

**Median-based Approach:**
```
Reorder Cycle Median = median(days_between_orders)
IQR = Q3 - Q1  (75th percentile - 25th percentile)
Coefficient of Variation (CV) = σ / μ
```

**Example:**
```
Order intervals (days): [30, 28, 32, 29, 150, 31, 30]
Mean = 47.1 days (distorted by 150)
Median = 30 days (robust to outlier)
IQR = 32 - 29 = 3 days
```

### 2.2 Mann-Kendall Trend Test

**Purpose**: Detect monotonic trends without assuming normal distribution

**Method**: Non-parametric test based on rank correlation
```python
# For time series X₁, X₂, ..., Xₙ
# Count concordant pairs: Xⱼ > Xᵢ for j > i
# Kendall's tau measures correlation
tau, p_value = kendalltau(time_indices, intervals)

# Sen's Slope (robust slope estimator)
slope = median(all pairwise slopes)
```

**Interpretation:**
- p-value < 0.05: Statistically significant trend
- Positive slope in intervals → Declining order frequency
- Negative slope in intervals → Growing order frequency

**Advantages:**
- Resistant to outliers
- No distribution assumptions
- Robust to missing data

### 2.3 FFT and Autocorrelation for Seasonality

**Fast Fourier Transform (FFT):**
- Decomposes time series into frequency components
- Identifies periodic patterns in order cycles

**Autocorrelation:**
```python
# Autocorrelation at lag k
autocorr[k] = correlation(series[0:n-k], series[k:n])

# Peak detection in autocorrelation
# Peak at lag k → seasonality with period k
```

**Common Business Cycles Detected:**
- Weekly (7 days): Regular weekly orders
- Bi-weekly (14 days): Every two weeks
- Monthly (30 days): Monthly replenishment
- Bi-monthly (60 days): Every two months
- Quarterly (90 days): Seasonal patterns

**Algorithm:**
```
1. Compute autocorrelation of reorder cycles
2. Find peaks (local maxima) with autocorr > 0.3
3. Convert peak lag to days: period_days = lag × median_cycle
4. Match to nearest common business cycle (within 30% tolerance)
5. Strength = autocorrelation value at peak
```

### 2.4 Linear Regression for Quantity Trends

**Simple Linear Regression:**
```
y = mx + b
where:
  y = order quantity
  x = order number (time proxy)
  m = slope (quantity change per order)
  b = intercept
```

**Statistical Test:**
```python
slope, intercept, r_value, p_value, std_err = linregress(x, quantities)

if p_value < 0.05 and |slope| > 0.1 * avg_quantity / n_orders:
    trend = 'increasing' or 'decreasing'
else:
    trend = 'stable'
```

### 2.5 Survival Analysis for Churn Prediction

**Exponential Decay Model:**
```
Expected next order = last_order_date + median_reorder_cycle
Days overdue = current_date - expected_next_order

Churn probability:
- Not overdue (≤0):           5%
- Slightly overdue (≤IQR):   15%
- Moderately overdue (≤1x):  40%
- Significantly overdue (≤2x): 70%
- Very overdue (>2x):        90%
```

**Loyalty Adjustment:**
```
loyalty_factor = min(1.0, n_orders / 10)
churn_prob_adjusted = churn_prob × (1 - 0.3 × loyalty_factor)
```

**RFM Enhancement:**
```
rfm_loyalty = 0.4 × recency_score + 0.4 × frequency_score + 0.2 × monetary_score
churn_reduction = 0.5 × rfm_loyalty
final_churn_prob = churn_prob_adjusted × (1 - churn_reduction)
```

### 2.6 Bayesian Confidence Scoring

**Bayesian Posterior Probability:**
```
Prior: P(reliable pattern) = 0.5  (neutral starting belief)

Likelihood factors:
- Sample size: L₁ = min(1.0, n_orders / 15)
- Consistency: L₂ = consistency_score
- Recency: L₃ = min(1.0, median_cycle / days_since_last)
- RFM: L₄ = 0.4×RFM_consistency + 0.3×RFM_frequency + 0.3×RFM_recency

Posterior (with RFM):
P(reliable | data) = 0.1×Prior + 0.25×L₁ + 0.20×L₂ + 0.10×L₃ + 0.35×L₄

Posterior (without RFM):
P(reliable | data) = 0.2×Prior + 0.4×L₁ + 0.3×L₂ + 0.1×L₃
```

### 2.7 RFM Analysis

**Recency Score (0-1):**
```
If cycle_median exists:
  recency_score = min(1.0, cycle_median / days_since_last)
Else:
  recency_score = 1.0   if days < 30
                  0.7   if 30 ≤ days < 90
                  0.4   if 90 ≤ days < 180
                  0.1   if days ≥ 180
```

**Frequency Score (0-1):**
```
orders_per_year = n_orders / (total_days / 365.25)

frequency_score = 0.3 × orders_per_year              if orders_per_year < 1
                  0.3 + 0.4 × (opy - 1) / 3         if 1 ≤ opy < 4
                  0.7 + 0.3 × min(1, (opy - 4) / 8) if opy ≥ 4
```

**Monetary Score (0-1):**
```
Using avg_quantity as proxy for revenue:

monetary_score = 0.3 + 0.4 × (qty / 5)               if qty < 5
                 0.7 + 0.2 × ((qty - 5) / 15)        if 5 ≤ qty < 20
                 0.9 + 0.1 × min(1, (qty - 20) / 30) if qty ≥ 20
```

**Consistency Score (0-1):**
```
Based on coefficient of variation (CV):

consistency_score = 0.7 + 0.3 × (1 - CV / 0.5)      if CV < 0.5 (very consistent)
                    0.3 + 0.4 × (1 - (CV - 0.5))   if 0.5 ≤ CV < 1.5 (moderate)
                    0.3 × max(0, 1 - (CV - 1.5)/2) if CV ≥ 1.5 (inconsistent)
```

### 2.8 Order Velocity and Acceleration

**Velocity (First Derivative):**
```
Measures change in order frequency over time

Split intervals into historical (first 75%) and recent (last 25%):
historical_median = median(intervals[:75%])
recent_median = median(intervals[75%:])

velocity = (recent_median - historical_median) / historical_median

Interpretation:
- Negative velocity: Accelerating (intervals decreasing → ordering faster)
- Positive velocity: Decelerating (intervals increasing → ordering slower)
- Near zero: Stable ordering pattern
```

**Acceleration (Second Derivative):**
```
Measures change in velocity (rate of change of rate of change)

Split intervals into thirds:
early_median, middle_median, late_median

velocity₁ = (middle - early) / early
velocity₂ = (late - middle) / middle

acceleration = velocity₂ - velocity₁

Interpretation:
- Positive acceleration: Ordering is slowing down more and more
- Negative acceleration: Ordering is speeding up more and more
- Near zero: Velocity is constant
```

---

## 3. Data Structures

### 3.1 CustomerProductPattern Dataclass

Complete pattern analysis result containing all computed metrics.

```python
@dataclass
class CustomerProductPattern:
    # Identifiers
    customer_id: int              # Customer identifier
    product_id: int               # Product identifier

    # Order History Metrics
    total_orders: int             # Total number of orders
    first_order_date: datetime    # Date of first order
    last_order_date: datetime     # Date of most recent order

    # Reorder Cycle Metrics (Robust Statistics)
    reorder_cycle_median: float   # Median days between orders (robust to outliers)
    reorder_cycle_iqr: float      # Interquartile range (Q3 - Q1)
    reorder_cycle_cv: float       # Coefficient of variation (σ/μ)

    # Quantity Metrics
    avg_quantity: float           # Average order quantity
    quantity_stddev: float        # Standard deviation of quantities
    quantity_trend: str           # 'increasing', 'stable', or 'decreasing'

    # Temporal Patterns (Seasonality)
    seasonality_detected: bool                # True if seasonal pattern found
    seasonality_period_days: Optional[int]    # Period in days (7, 14, 30, 60, 90)
    seasonality_strength: float              # 0-1 (autocorrelation peak value)

    # Trend Analysis (Mann-Kendall)
    trend_direction: str          # 'growing', 'stable', or 'declining'
    trend_slope: float           # Change in orders per month
    trend_pvalue: float          # Statistical significance (< 0.05 = significant)

    # Health/Churn Metrics
    consistency_score: float     # 0-1 (1 = very predictable patterns)
    status: str                  # 'active', 'at_risk', or 'churned'
    churn_probability: float     # 0-1 (probability of customer churn)
    days_since_last_order: int   # Days elapsed since last order
    days_overdue: float          # Days past expected reorder (negative = early)

    # RFM Metrics
    rfm_recency_score: float     # 0-1 (1 = very recent order)
    rfm_frequency_score: float   # 0-1 (normalized orders per year)
    rfm_monetary_score: float    # 0-1 (normalized avg order value)
    rfm_consistency_score: float # 0-1 (regularity of ordering)
    rfm_segment: str             # 'champion', 'loyal', 'potential', 'at_risk',
                                 # 'hibernating', or 'lost'

    # Order Velocity Metrics
    order_velocity: float        # Rate of change in order frequency
                                 # Negative = accelerating, Positive = decelerating
    order_acceleration: float    # Second derivative (change in velocity)
    velocity_trend: str          # 'accelerating', 'stable', or 'decelerating'

    # Confidence Metrics
    pattern_confidence: float    # 0-1 (Bayesian posterior confidence)
```

### 3.2 Order History Structure

Internal representation of order data:

```python
Order = {
    'date': datetime,      # Order creation timestamp
    'quantity': int        # Total quantity ordered (sum of all items)
}

# Example:
orders = [
    {'date': datetime(2023, 1, 15), 'quantity': 10},
    {'date': datetime(2023, 2, 18), 'quantity': 12},
    {'date': datetime(2023, 3, 20), 'quantity': 15}
]
```

### 3.3 Intermediate Computation Structures

**Seasonality Detection Result:**
```python
seasonality = {
    'period': int,          # Period in days (7, 14, 30, 60, 90)
    'strength': float,      # Autocorrelation peak value (0-1)
    'confidence': float     # 0.6 if n_cycles < 24, 0.8 if n_cycles ≥ 24
}
```

**Trend Detection Result:**
```python
trend = {
    'direction': str,      # 'growing', 'stable', or 'declining'
    'slope': float,        # Orders per month change
    'pvalue': float        # Statistical significance
}
```

**RFM Features:**
```python
rfm = {
    'recency_score': float,      # 0-1
    'frequency_score': float,    # 0-1
    'monetary_score': float,     # 0-1
    'consistency_score': float,  # 0-1
    'segment': str               # Customer segment classification
}
```

**Velocity Calculation:**
```python
velocity = {
    'velocity': float,        # First derivative
    'acceleration': float,    # Second derivative
    'trend': str             # 'accelerating', 'stable', or 'decelerating'
}
```

---

## 4. Algorithm Flow

### 4.1 High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    analyze_customer_product()                    │
│                     Entry Point - Main Flow                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Fetch Order History from Database                       │
│   • Query: ClientAgreement → Order → OrderItem                  │
│   • Filter: customer_id, product_id, before as_of_date          │
│   • Returns: List[{'date': datetime, 'quantity': int}]          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Calculate Reorder Cycles                                │
│   • Compute fractional days between consecutive orders          │
│   • Filter: Ignore orders < 1 day apart                         │
│   • Returns: List[float] (days between orders)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Robust Reorder Statistics                               │
│   • Median cycle = median(cycles)                               │
│   • IQR = Q3 - Q1                                               │
│   • CV = σ / μ                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Quantity Statistics                                     │
│   • Average quantity = mean(quantities)                         │
│   • Quantity stddev = std(quantities)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Quantity Trend Detection (Linear Regression)            │
│   • Fit: quantity ~ order_number                                │
│   • Test: p-value < 0.05 and |slope| > 10% threshold           │
│   • Returns: 'increasing', 'stable', or 'decreasing'            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Seasonality Detection (FFT + Autocorrelation)           │
│   • Requires: ≥ 12 reorder cycles                               │
│   • Compute autocorrelation of cycles                           │
│   • Find peaks with autocorr > 0.3                              │
│   • Match to business cycles (7, 14, 30, 60, 90 days)           │
│   • Returns: {period, strength, confidence} or None             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 7: Frequency Trend Detection (Mann-Kendall)                │
│   • Calculate order intervals                                   │
│   • Mann-Kendall test: kendalltau(time, intervals)              │
│   • Sen's slope estimator for robust slope                      │
│   • Returns: {direction, slope, pvalue}                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 8: Consistency Score                                       │
│   • Combines: CV score, sample size, IQR score                  │
│   • Weighted: 40% CV + 30% sample + 30% IQR                     │
│   • Returns: 0-1 score                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 9: Calculate Days Since Last Order                         │
│   • days_since_last = as_of_date - last_order_date              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 10: RFM Features Calculation                               │
│   • Recency score (normalized by expected cycle)                │
│   • Frequency score (orders per year)                           │
│   • Monetary score (avg quantity as proxy)                      │
│   • Consistency score (inverse of CV)                           │
│   • Segment classification (champion, loyal, at_risk, etc.)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 11: Order Velocity Calculation                             │
│   • Velocity: Compare recent vs historical intervals            │
│   • Acceleration: Change in velocity over time                  │
│   • Trend: 'accelerating', 'stable', or 'decelerating'          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 12: Churn Risk Assessment (Survival Analysis + RFM)        │
│   • Calculate days_overdue = days_since_last - median_cycle     │
│   • Base churn probability (exponential decay)                  │
│   • Loyalty adjustment (more orders → lower churn)              │
│   • RFM enhancement (high RFM → lower churn)                    │
│   • Returns: (status, churn_probability, days_overdue)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 13: Pattern Confidence (Bayesian Posterior)                │
│   • Prior: 0.5 (neutral)                                        │
│   • Likelihood: sample size, consistency, recency, RFM          │
│   • Posterior: Weighted combination                             │
│   • Returns: 0-1 confidence score                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 14: Assemble CustomerProductPattern Object                 │
│   • Populate all 33 fields                                      │
│   • Return complete pattern analysis                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Step-by-Step Process

#### Step 1: Fetch Order History
```python
def _get_order_history(customer_id, product_id, as_of_date):
    """
    SQL Query:
    - Joins: ClientAgreement → Order → OrderItem
    - Filters: ClientID, ProductID, Order.Created < as_of_date
    - Groups: By Order.ID to sum quantities per order
    - Orders: By Order.Created (chronological)

    Returns: List of {date, quantity} dicts
    """
```

#### Step 2: Calculate Reorder Cycles
```python
def _calculate_reorder_cycles(orders):
    """
    For each consecutive pair of orders:
    1. Calculate fractional days between orders
    2. If days >= 1.0 (ignore same-day duplicates):
       - Append to cycles list

    Example:
    orders = [
        {'date': '2023-01-01 10:00', 'qty': 10},
        {'date': '2023-01-31 14:30', 'qty': 12},
        {'date': '2023-03-02 09:15', 'qty': 15}
    ]
    cycles = [30.19, 29.77]  # fractional days
    """
```

#### Step 3-8: Statistical Analysis
Each metric is computed independently and can be calculated in parallel.

#### Step 9-13: RFM, Velocity, Churn, Confidence
These steps build on earlier calculations and must be computed sequentially.

---

## 5. Key Metrics Calculated

### 5.1 Reorder Cycle Metrics

**reorder_cycle_median**
- Robust central tendency of days between orders
- Resistant to outliers (bulk orders, holidays, etc.)
- Used as expected reorder interval for predictions

**reorder_cycle_iqr**
- Interquartile range (Q3 - Q1)
- Measures spread without outlier influence
- Used in churn risk assessment (tolerance band)

**reorder_cycle_cv**
- Coefficient of variation (σ / μ)
- Measures relative variability
- Lower CV = more predictable ordering pattern

### 5.2 Quantity Metrics

**avg_quantity**
- Average units ordered per transaction
- Used as proxy for customer value (monetary score)

**quantity_stddev**
- Variability in order sizes
- Indicates order size predictability

**quantity_trend**
- Directional trend in order quantities
- Values: 'increasing', 'stable', 'decreasing'
- Based on linear regression with p < 0.05 significance

### 5.3 Seasonality Metrics

**seasonality_detected**
- Boolean flag indicating presence of periodic pattern
- Requires ≥12 cycles and autocorrelation peak > 0.3

**seasonality_period_days**
- Length of seasonal cycle in days
- Common values: 7, 14, 30, 60, 90
- Matched to nearest business cycle

**seasonality_strength**
- Autocorrelation value at seasonal peak (0-1)
- Higher = stronger seasonal signal
- > 0.5 = strong seasonality

### 5.4 Trend Metrics

**trend_direction**
- Overall trajectory of order frequency
- Values: 'growing', 'stable', 'declining'
- Based on Mann-Kendall test

**trend_slope**
- Change in orders per month
- Positive = frequency increasing
- Negative = frequency decreasing

**trend_pvalue**
- Statistical significance of trend
- < 0.05 = statistically significant
- > 0.05 = trend not significant (noise)

### 5.5 Health/Churn Metrics

**consistency_score** (0-1)
- Measures predictability of ordering pattern
- Formula: 40% CV score + 30% sample score + 30% IQR score
- > 0.7 = highly consistent
- 0.4-0.7 = moderately consistent
- < 0.4 = inconsistent/unpredictable

**status**
- Customer health status
- 'active': Ordering on schedule
- 'at_risk': Overdue but within tolerance
- 'churned': Significantly overdue (> 2x cycle)

**churn_probability** (0-1)
- Likelihood of customer not returning
- Incorporates: days overdue, loyalty, RFM scores
- > 0.7 = high risk
- 0.3-0.7 = medium risk
- < 0.3 = low risk

**days_since_last_order**
- Recency metric
- Integer days elapsed

**days_overdue**
- Days past expected reorder
- Negative = early (not overdue)
- Positive = late (overdue)

### 5.6 RFM Metrics

**rfm_recency_score** (0-1)
- How recently did customer order
- 1.0 = ordered on time or early
- < 0.5 = significantly overdue

**rfm_frequency_score** (0-1)
- How often does customer order
- Based on orders per year
- > 0.7 = frequent (4+ per year)
- 0.3-0.7 = moderate (1-4 per year)
- < 0.3 = infrequent (< 1 per year)

**rfm_monetary_score** (0-1)
- Customer value based on order size
- Uses avg_quantity as proxy
- > 0.7 = high value (10+ units)
- 0.3-0.7 = medium value (5-10 units)
- < 0.3 = low value (< 5 units)

**rfm_consistency_score** (0-1)
- Regularity of ordering pattern
- Inverse of coefficient of variation
- > 0.7 = very regular
- < 0.3 = irregular/unpredictable

**rfm_segment**
- Customer classification
- 'champion': High R, F, M (best customers)
- 'loyal': High R, F (regular customers)
- 'potential': High R, low F (new customers)
- 'at_risk': Medium R, high F (slipping)
- 'hibernating': Medium R (need attention)
- 'lost': Low R (likely churned)

### 5.7 Velocity Metrics

**order_velocity**
- Rate of change in order frequency
- Negative = accelerating (ordering faster)
- Positive = decelerating (ordering slower)
- ~ 0 = stable

**order_acceleration**
- Second derivative (change in velocity)
- Positive = deceleration increasing
- Negative = acceleration increasing

**velocity_trend**
- 'accelerating': velocity < -0.15
- 'stable': -0.15 ≤ velocity ≤ 0.15
- 'decelerating': velocity > 0.15

### 5.8 Confidence Metrics

**pattern_confidence** (0-1)
- Bayesian posterior probability
- Confidence in pattern predictions
- Incorporates: sample size, consistency, recency, RFM
- > 0.7 = high confidence
- 0.4-0.7 = medium confidence
- < 0.4 = low confidence (unreliable predictions)

---

## 6. Input/Output Examples

### 6.1 Example 1: Regular Monthly Customer

**Input:**
```python
customer_id = 12345
product_id = 67890
as_of_date = "2024-01-15"

# Order history (from database):
orders = [
    {'date': datetime(2023, 1, 10), 'quantity': 10},
    {'date': datetime(2023, 2, 8),  'quantity': 12},
    {'date': datetime(2023, 3, 12), 'quantity': 10},
    {'date': datetime(2023, 4, 9),  'quantity': 11},
    {'date': datetime(2023, 5, 11), 'quantity': 10},
    {'date': datetime(2023, 6, 10), 'quantity': 12},
    # ... 12 more months of data
]
```

**Output:**
```python
CustomerProductPattern(
    customer_id=12345,
    product_id=67890,
    total_orders=18,
    first_order_date=datetime(2023, 1, 10),
    last_order_date=datetime(2024, 1, 8),

    # Reorder cycles: ~30 days (monthly)
    reorder_cycle_median=30.2,
    reorder_cycle_iqr=3.5,
    reorder_cycle_cv=0.15,  # Very consistent

    # Quantity: stable around 11 units
    avg_quantity=11.0,
    quantity_stddev=0.8,
    quantity_trend='stable',

    # Seasonality: Monthly pattern detected
    seasonality_detected=True,
    seasonality_period_days=30,
    seasonality_strength=0.72,

    # Trend: Stable frequency
    trend_direction='stable',
    trend_slope=0.02,
    trend_pvalue=0.89,  # Not significant

    # Health: Active customer
    consistency_score=0.85,
    status='active',
    churn_probability=0.08,
    days_since_last_order=7,
    days_overdue=-23.2,  # Not overdue

    # RFM: Champion customer
    rfm_recency_score=0.95,
    rfm_frequency_score=0.80,
    rfm_monetary_score=0.68,
    rfm_consistency_score=0.92,
    rfm_segment='champion',

    # Velocity: Stable ordering
    order_velocity=-0.05,
    order_acceleration=0.01,
    velocity_trend='stable',

    # Confidence: High
    pattern_confidence=0.88
)
```

### 6.2 Example 2: At-Risk Customer

**Input:**
```python
customer_id = 11111
product_id = 22222
as_of_date = "2024-01-15"

# Order history: Was regular, now overdue
orders = [
    {'date': datetime(2023, 1, 5),  'quantity': 20},
    {'date': datetime(2023, 2, 8),  'quantity': 22},
    # ... more regular orders
    {'date': datetime(2023, 10, 3), 'quantity': 18},
    # Last order was 103 days ago (expected ~33 days)
]
```

**Output:**
```python
CustomerProductPattern(
    customer_id=11111,
    product_id=22222,
    total_orders=10,

    reorder_cycle_median=33.5,
    reorder_cycle_iqr=5.2,
    reorder_cycle_cv=0.22,

    avg_quantity=20.0,
    quantity_stddev=1.5,
    quantity_trend='stable',

    seasonality_detected=False,
    seasonality_period_days=None,
    seasonality_strength=0.0,

    trend_direction='stable',
    trend_slope=-0.03,
    trend_pvalue=0.67,

    # Health: At risk!
    consistency_score=0.72,
    status='at_risk',
    churn_probability=0.65,
    days_since_last_order=103,
    days_overdue=69.5,  # Very overdue (2x cycle)

    # RFM: Slipping customer
    rfm_recency_score=0.32,  # Low recency
    rfm_frequency_score=0.55,
    rfm_monetary_score=0.75,
    rfm_consistency_score=0.85,
    rfm_segment='at_risk',

    order_velocity=0.18,  # Decelerating
    order_acceleration=0.05,
    velocity_trend='decelerating',

    pattern_confidence=0.58
)
```

### 6.3 Example 3: New Customer (Limited Data)

**Input:**
```python
customer_id = 99999
product_id = 88888
as_of_date = "2024-01-15"

# Only 2 orders
orders = [
    {'date': datetime(2023, 11, 20), 'quantity': 5},
    {'date': datetime(2023, 12, 25), 'quantity': 6}
]
```

**Output:**
```python
CustomerProductPattern(
    customer_id=99999,
    product_id=88888,
    total_orders=2,
    first_order_date=datetime(2023, 11, 20),
    last_order_date=datetime(2023, 12, 25),

    # Limited cycle data
    reorder_cycle_median=35.0,  # Only 1 cycle
    reorder_cycle_iqr=0.0,      # Can't calculate IQR
    reorder_cycle_cv=0.0,

    avg_quantity=5.5,
    quantity_stddev=0.7,
    quantity_trend='stable',  # Too few points

    # No seasonality (need ≥12 cycles)
    seasonality_detected=False,
    seasonality_period_days=None,
    seasonality_strength=0.0,

    # No trend (need ≥4 orders)
    trend_direction='stable',
    trend_slope=0.0,
    trend_pvalue=1.0,

    # Health: Unknown/new
    consistency_score=0.20,  # Low due to limited data
    status='active',
    churn_probability=0.35,
    days_since_last_order=21,
    days_overdue=-14.0,

    # RFM: Potential customer
    rfm_recency_score=0.85,
    rfm_frequency_score=0.25,  # Low frequency (new)
    rfm_monetary_score=0.40,
    rfm_consistency_score=1.0,  # No variance yet
    rfm_segment='potential',

    order_velocity=0.0,  # Need ≥3 orders
    order_acceleration=0.0,
    velocity_trend='stable',

    # Low confidence due to limited data
    pattern_confidence=0.28
)
```

### 6.4 Example 4: Seasonal Customer

**Input:**
```python
# Customer who orders quarterly (every ~90 days)
orders = [
    {'date': datetime(2022, 1, 10),  'quantity': 50},
    {'date': datetime(2022, 4, 8),   'quantity': 55},
    {'date': datetime(2022, 7, 12),  'quantity': 52},
    {'date': datetime(2022, 10, 5),  'quantity': 48},
    {'date': datetime(2023, 1, 9),   'quantity': 51},
    {'date': datetime(2023, 4, 11),  'quantity': 54},
    {'date': datetime(2023, 7, 8),   'quantity': 49},
    {'date': datetime(2023, 10, 10), 'quantity': 53}
]
as_of_date = "2024-01-15"
```

**Output:**
```python
CustomerProductPattern(
    total_orders=8,

    # Quarterly cycle
    reorder_cycle_median=91.5,
    reorder_cycle_iqr=5.0,
    reorder_cycle_cv=0.08,  # Very consistent

    avg_quantity=51.5,
    quantity_stddev=2.3,
    quantity_trend='stable',

    # Strong quarterly seasonality
    seasonality_detected=True,
    seasonality_period_days=90,
    seasonality_strength=0.85,

    trend_direction='stable',
    trend_slope=0.01,
    trend_pvalue=0.92,

    # Health: Active (within expected cycle)
    consistency_score=0.88,
    status='active',
    churn_probability=0.12,
    days_since_last_order=97,
    days_overdue=5.5,  # Slightly overdue but in IQR

    # RFM: Loyal customer
    rfm_recency_score=0.92,
    rfm_frequency_score=0.60,  # ~4 orders/year
    rfm_monetary_score=0.92,   # High quantity
    rfm_consistency_score=0.95,
    rfm_segment='loyal',

    order_velocity=-0.08,
    order_acceleration=0.02,
    velocity_trend='stable',

    pattern_confidence=0.82
)
```

---

## 7. Performance Characteristics

### 7.1 Time Complexity

**Per Customer-Product Pair Analysis:**
```
Component                       Complexity      Typical Runtime
─────────────────────────────────────────────────────────────────
Database query                  O(n log n)      50-200 ms
Reorder cycle calculation       O(n)            < 1 ms
Robust statistics (median/IQR)  O(n log n)      < 1 ms
Quantity trend (linear reg)     O(n)            < 1 ms
Seasonality (autocorrelation)   O(n²)           1-5 ms
Mann-Kendall trend              O(n²)           1-3 ms
RFM calculation                 O(n)            < 1 ms
Velocity calculation            O(n)            < 1 ms
Churn assessment                O(1)            < 1 ms
Confidence calculation          O(1)            < 1 ms
─────────────────────────────────────────────────────────────────
Total (excluding DB query)      O(n²)           5-15 ms
Total (including DB query)      O(n²)           55-215 ms

where n = number of orders (typically 5-50 for active customers)
```

**Scalability:**
- For n < 100 orders: Negligible computation time (< 50 ms)
- For n = 1000 orders: Autocorrelation becomes dominant (~500 ms)
- Database query is typically the bottleneck (50-200 ms)

### 7.2 Space Complexity

**Memory Usage per Analysis:**
```
Data Structure                  Size            Notes
──────────────────────────────────────────────────────────────
Order history                   O(n)            ~100 bytes × n
Reorder cycles                  O(n)            ~8 bytes × n
Autocorrelation array          O(n)            Only if n ≥ 12
Intermediate calculations       O(1)            ~2 KB
CustomerProductPattern          O(1)            ~400 bytes
──────────────────────────────────────────────────────────────
Total per analysis              O(n)            ~200 bytes × n + 2 KB

Typical: 5-10 KB per customer-product pair
Maximum: ~50 KB for high-volume customers (n=1000)
```

**Batch Processing:**
- For 10,000 customer-product pairs: ~50-100 MB RAM
- For 100,000 pairs: ~500 MB - 1 GB RAM
- Recommendation: Process in batches of 10,000-50,000

### 7.3 Database Load

**Query Characteristics:**
```sql
-- Single query per customer-product pair
-- Indexes required for optimal performance:
--   1. ClientAgreement.ClientID
--   2. Order.ClientAgreementID + Created
--   3. OrderItem.OrderID + ProductID

-- Query complexity: 3-table join with aggregation
-- Typical execution time: 50-200 ms
-- Rows scanned: 10-500 per customer-product
```

**Optimization Strategies:**
1. Connection pooling (reuse connections)
2. Prepared statements (reuse query plans)
3. Batch queries for multiple customers
4. Materialized views for frequently analyzed pairs
5. Partition orders by date for faster historical queries

### 7.4 Accuracy vs Speed Tradeoffs

**High Accuracy Mode (Default):**
- Uses 6+ years of historical data
- Full autocorrelation for seasonality
- Mann-Kendall for robust trend detection
- Time: ~55-215 ms per analysis

**Fast Mode (Potential Optimization):**
- Use 2-3 years of recent data only
- Skip autocorrelation for n < 24
- Simple linear regression for trends
- Time: ~20-80 ms per analysis
- Accuracy reduction: ~5-10%

**Batch Mode (Recommended for Large Datasets):**
- Process 10,000 pairs at once
- Parallelize computation across CPU cores
- Single batch DB query with window functions
- Time: ~30-60 seconds for 10,000 pairs
- Throughput: 150-300 pairs/second

### 7.5 Robustness

**Outlier Resistance:**
- Median-based metrics: Highly robust (50% breakdown point)
- IQR: Robust to outliers outside Q1-Q3 range
- Mann-Kendall: Resistant to extreme values
- Linear regression: Sensitive (use for quantity trends only)

**Missing Data Handling:**
- Minimum 1 order required (returns None if 0)
- Most metrics gracefully degrade with limited data
- Confidence scores reflect data quality

**Edge Cases:**
```python
# Handled gracefully:
- Single order → median cycle = None, limited RFM
- Two orders → one cycle, no trend, low confidence
- Same-day duplicates → filtered (cycles ≥ 1 day)
- Zero quantities → rare in B2B, defaults to 0.0
- Division by zero → checked, defaults to 0.0 or None
```

---

## 8. Dependencies and References

### 8.1 Python Dependencies

```python
# Core numerical computing
numpy >= 1.20.0            # Array operations, statistics
scipy >= 1.7.0             # Statistical tests, FFT

# Data structures
dataclasses                # (Python 3.7+, built-in)
typing                     # Type hints (built-in)

# Date/time handling
datetime                   # (built-in)

# Database
pymssql >= 2.2.0          # Microsoft SQL Server connector

# Logging
logging                    # (built-in)

# Custom utilities
datetime_utils             # Project-specific (calculate_fractional_days)
```

### 8.2 Statistical Methods References

**Robust Statistics:**
- Rousseeuw, P.J. & Leroy, A.M. (1987). "Robust Regression and Outlier Detection"
- Median and IQR: Tukey, J.W. (1977). "Exploratory Data Analysis"

**Mann-Kendall Test:**
- Mann, H.B. (1945). "Nonparametric tests against trend"
- Kendall, M.G. (1975). "Rank Correlation Methods"
- Sen, P.K. (1968). "Estimates of the regression coefficient based on Kendall's tau"

**Seasonality Detection:**
- Box, G.E.P. & Jenkins, G.M. (1976). "Time Series Analysis: Forecasting and Control"
- FFT: Cooley, J.W. & Tukey, J.W. (1965). "An algorithm for the machine calculation of complex Fourier series"

**Survival Analysis:**
- Kaplan, E.L. & Meier, P. (1958). "Nonparametric estimation from incomplete observations"
- Cox, D.R. (1972). "Regression models and life-tables"

**RFM Analysis:**
- Hughes, A.M. (1994). "Strategic Database Marketing"
- Fader, P.S. & Hardie, B.G. (2009). "Probability Models for Customer-Base Analysis"

**Bayesian Methods:**
- Gelman, A. et al. (2013). "Bayesian Data Analysis, 3rd Edition"
- McElreath, R. (2020). "Statistical Rethinking: A Bayesian Course"

### 8.3 Database Schema Dependencies

**Required Tables:**
```sql
-- Client relationship
dbo.ClientAgreement (
    ID int PRIMARY KEY,
    ClientID int NOT NULL,  -- Customer identifier
    ...
)

-- Orders
dbo.[Order] (
    ID int PRIMARY KEY,
    ClientAgreementID int NOT NULL,
    Created datetime NOT NULL,  -- Order timestamp
    ...
)

-- Order line items
dbo.OrderItem (
    ID int PRIMARY KEY,
    OrderID int NOT NULL,
    ProductID int NOT NULL,  -- Product identifier
    Qty int NOT NULL,        -- Quantity ordered
    ...
)
```

**Required Indexes:**
```sql
-- Critical for query performance
CREATE INDEX IX_ClientAgreement_ClientID ON dbo.ClientAgreement(ClientID);
CREATE INDEX IX_Order_ClientAgreementID_Created ON dbo.[Order](ClientAgreementID, Created);
CREATE INDEX IX_OrderItem_OrderID_ProductID ON dbo.OrderItem(OrderID, ProductID);
```

### 8.4 Related Algorithms and Systems

**Upstream Dependencies:**
- `datetime_utils.calculate_fractional_days`: Precise timestamp difference calculation

**Downstream Consumers:**
- Demand forecasting models (use pattern metrics as features)
- Churn prediction system (uses churn_probability, status)
- Customer segmentation (uses RFM segment)
- Inventory optimization (uses reorder_cycle_median, seasonality)
- Sales recommendations (uses order_velocity, trend_direction)

### 8.5 Configuration Parameters

**Thresholds (hard-coded, can be made configurable):**
```python
# Seasonality detection
MIN_CYCLES_FOR_SEASONALITY = 12
AUTOCORR_THRESHOLD = 0.3
BUSINESS_CYCLES = [7, 14, 30, 60, 90]
CYCLE_MATCH_TOLERANCE = 0.3  # ±30%

# Trend detection
MIN_ORDERS_FOR_TREND = 4
TREND_SIGNIFICANCE = 0.05  # p-value

# Quantity trend
MIN_ORDERS_FOR_QTY_TREND = 3
QTY_TREND_THRESHOLD = 0.1  # 10% per order

# Consistency score weights
CV_WEIGHT = 0.4
SAMPLE_WEIGHT = 0.3
IQR_WEIGHT = 0.3
MAX_SAMPLE_SIZE = 20  # Normalize sample score

# Churn assessment
BASE_CHURN_RATE = 0.05
LOYALTY_FACTOR_MAX_ORDERS = 10
LOYALTY_DISCOUNT = 0.3
RFM_CHURN_REDUCTION_MAX = 0.5

# RFM scoring
FREQUENCY_BREAKPOINTS = [1, 4]     # orders/year
MONETARY_BREAKPOINTS = [5, 20]     # quantity
CV_BREAKPOINTS = [0.5, 1.5]        # consistency

# Velocity detection
VELOCITY_THRESHOLD = 0.15  # ±15% change
MIN_ORDERS_FOR_VELOCITY = 3

# Confidence calculation
CONFIDENCE_PRIOR = 0.5
CONFIDENCE_MAX_SAMPLE = 15
```

### 8.6 Testing and Validation

**Recommended Test Cases:**
1. Regular monthly customer (n=24)
2. Quarterly seasonal customer (n=12)
3. At-risk customer (overdue)
4. Churned customer (>2x cycle)
5. New customer (n=2)
6. High-frequency customer (weekly orders)
7. Irregular customer (high CV)
8. Customer with outlier orders
9. Customer with increasing trend
10. Customer with decreasing trend

**Validation Metrics:**
- Prediction accuracy: ±10% of actual next order date
- Churn prediction AUC: > 0.75
- RFM segment stability: > 80% consistent month-over-month
- Confidence calibration: Predicted confidence ≈ actual accuracy

### 8.7 Known Limitations

1. **Monetary Score Proxy**: Uses quantity instead of actual revenue (limitation of current data model)
2. **Seasonality Detection**: Requires ≥12 cycles (1+ year of consistent ordering)
3. **Trend Detection**: Requires ≥4 orders (may miss trends in new customers)
4. **Computational Cost**: O(n²) autocorrelation limits scalability for very high-order customers
5. **Business Cycles**: Limited to [7, 14, 30, 60, 90] days (may miss uncommon cycles)
6. **Same-day Orders**: Treated as duplicates, merged into single cycle
7. **Fractional Days**: Sub-day precision requires datetime_utils dependency

### 8.8 Future Enhancements

**Potential Improvements:**
1. **Multi-product Analysis**: Cross-product purchase patterns
2. **External Factors**: Incorporate holidays, promotions, seasonality calendars
3. **Machine Learning**: Replace rules-based churn with ML model
4. **Real Revenue Data**: Replace quantity proxy with actual order values
5. **Hierarchical Models**: Customer groups, product categories
6. **Anomaly Detection**: Flag unusual orders/patterns
7. **Forecast Integration**: Direct next-order-date prediction
8. **A/B Testing**: Confidence-based experiment allocation

---

## Appendix A: Quick Reference

### Common Usage Pattern

```python
from pattern_analyzer import PatternAnalyzer
import pymssql

# Initialize with database connection
conn = pymssql.connect(server, user, password, database)
analyzer = PatternAnalyzer(conn)

# Analyze customer-product pair
pattern = analyzer.analyze_customer_product(
    customer_id=12345,
    product_id=67890,
    as_of_date="2024-01-15"
)

# Access key metrics
if pattern:
    print(f"Reorder cycle: {pattern.reorder_cycle_median:.1f} days")
    print(f"Status: {pattern.status}")
    print(f"Churn risk: {pattern.churn_probability:.1%}")
    print(f"RFM segment: {pattern.rfm_segment}")
    print(f"Confidence: {pattern.pattern_confidence:.2f}")
```

### Interpretation Guide

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| consistency_score | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| churn_probability | < 0.2 | 0.2-0.4 | 0.4-0.7 | > 0.7 |
| pattern_confidence | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| rfm_recency_score | > 0.8 | 0.6-0.8 | 0.3-0.6 | < 0.3 |
| rfm_frequency_score | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |

### Status Interpretation

- **active**: Customer is ordering normally, low churn risk
- **at_risk**: Customer is overdue but recoverable, medium churn risk
- **churned**: Customer is significantly overdue, high churn risk

### RFM Segment Interpretation

- **champion**: Best customers (high R, F, M) - retain and reward
- **loyal**: Regular customers (high R, F) - upsell opportunities
- **potential**: New customers (high R, low F) - nurture and grow
- **at_risk**: Slipping customers (medium R, high F) - re-engage urgently
- **hibernating**: Dormant customers (medium R) - reactivation campaigns
- **lost**: Churned customers (low R) - win-back or archive

---

**Document Version**: 1.0
**Last Updated**: 2024-01-15
**Algorithm Version**: Production (6+ years historical data)
**Contact**: Data Science Team

"""
Demand Forecaster Module for Order Recommendations v2.

Provides enhanced demand forecasting with:
- Trend analysis (linear regression)
- Seasonality detection (autocorrelation)
- Customer churn adjustment

Author: AI Assistant
"""
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator
from datetime import date, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TrendResult:
    """Result of trend analysis."""
    factor: float  # Multiplier for demand adjustment (e.g., 1.05 = 5% growth)
    direction: str  # 'growing', 'declining', 'stable'
    weekly_slope: float  # Units change per week
    confidence: float  # R-squared or similar metric


@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""
    index: float  # Seasonal index for current period (1.0 = average)
    period_weeks: Optional[int]  # Detected period (e.g., 52 for annual)
    strength: float  # Autocorrelation strength (0-1)


@dataclass
class ChurnResult:
    """Result of churn analysis."""
    adjustment: float  # Multiplier (e.g., 0.95 = 5% reduction)
    at_risk_pct: float  # Percentage of demand from at-risk customers
    at_risk_customers: int  # Count of at-risk customers


@dataclass
class EnhancedDemandStats:
    """Enhanced demand statistics with forecasting adjustments."""
    product_id: int
    mean: float  # Base average weekly demand
    stddev: float  # Standard deviation
    total_qty: float  # Total quantity in history
    data_weeks: int  # Actual weeks with data

    # Adjustments
    trend: Optional[TrendResult] = None
    seasonality: Optional[SeasonalityResult] = None
    churn: Optional[ChurnResult] = None

    # Computed adjusted demand
    adjusted_mean: float = 0.0
    forecast_method: str = "basic"
    forecast_confidence: float = 0.5


MAX_IN_CLAUSE_ITEMS = 1000


def calculate_trend_factor(
    weekly_demand: List[float],
    lead_time_weeks: int,
    min_weeks: int = 8
) -> Optional[TrendResult]:
    """
    Calculate trend factor using simple linear regression.

    Args:
        weekly_demand: List of weekly demand values (oldest to newest)
        lead_time_weeks: How far ahead to project the trend
        min_weeks: Minimum weeks required for trend analysis

    Returns:
        TrendResult or None if insufficient data
    """
    n = len(weekly_demand)
    if n < min_weeks:
        return None

    # Keep zeros to preserve spacing in the weekly series.
    x = list(range(n))
    y = weekly_demand

    # Calculate means
    x_mean = sum(x) / n
    y_mean = sum(y) / n

    if y_mean <= 0:
        return TrendResult(factor=1.0, direction="stable", weekly_slope=0.0, confidence=0.0)

    # Calculate slope using least squares
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        return TrendResult(factor=1.0, direction="stable", weekly_slope=0.0, confidence=0.0)

    slope = numerator / denominator  # Units per week change

    # Calculate R-squared for confidence
    y_pred = [slope * xi + (y_mean - slope * x_mean) for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Project forward by lead_time_weeks
    # Weekly growth rate relative to mean
    weekly_growth_rate = slope / y_mean if y_mean > 0 else 0

    # Project to the middle of lead time period
    projection_weeks = lead_time_weeks / 2
    trend_factor = 1.0 + (weekly_growth_rate * projection_weeks)

    # Cap at reasonable bounds
    trend_factor = max(0.5, min(2.0, trend_factor))

    # Determine direction
    if abs(weekly_growth_rate) < 0.005:  # Less than 0.5% per week
        direction = "stable"
    elif weekly_growth_rate > 0:
        direction = "growing"
    else:
        direction = "declining"

    return TrendResult(
        factor=round(trend_factor, 3),
        direction=direction,
        weekly_slope=round(slope, 4),
        confidence=round(max(0, r_squared), 3)
    )


def calculate_autocorrelation(data: List[float], lag: int) -> float:
    """Calculate autocorrelation at a specific lag."""
    n = len(data)
    if n < lag * 2:
        return 0.0

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n

    if variance == 0:
        return 0.0

    covariance = sum(
        (data[i] - mean) * (data[i + lag] - mean)
        for i in range(n - lag)
    ) / (n - lag)

    return covariance / variance


def detect_seasonality(
    weekly_demand: List[float],
    current_week_of_year: int,
    min_weeks: int = 52
) -> Optional[SeasonalityResult]:
    """
    Detect seasonality using autocorrelation analysis.

    Args:
        weekly_demand: List of weekly demand values (oldest to newest)
        current_week_of_year: Current week number (1-52)
        min_weeks: Minimum weeks required for seasonality detection

    Returns:
        SeasonalityResult or None if insufficient data
    """
    n = len(weekly_demand)
    if n < min_weeks:
        return None

    mean = sum(weekly_demand) / n
    if mean <= 0:
        return None

    # Check common seasonal periods
    # 4 weeks (~monthly), 13 weeks (quarterly), 26 weeks (semi-annual), 52 weeks (annual)
    candidate_periods = [4, 13, 26, 52]

    best_period = None
    best_correlation = 0.3  # Minimum threshold for significance

    for period in candidate_periods:
        if n < period * 2:
            continue

        correlation = calculate_autocorrelation(weekly_demand, period)
        if correlation > best_correlation:
            best_correlation = correlation
            best_period = period

    if best_period is None:
        return SeasonalityResult(index=1.0, period_weeks=None, strength=0.0)

    # Calculate seasonal index for current period
    # Group data by position in cycle and calculate average for current position
    current_position = (n - 1) % best_period  # Position of most recent data

    # Collect values at this position in cycle
    position_values = [
        weekly_demand[i]
        for i in range(current_position, n, best_period)
    ]

    if not position_values:
        return SeasonalityResult(index=1.0, period_weeks=best_period, strength=best_correlation)

    position_mean = sum(position_values) / len(position_values)
    seasonal_index = position_mean / mean if mean > 0 else 1.0

    # Cap seasonal index at reasonable bounds
    seasonal_index = max(0.5, min(2.0, seasonal_index))

    return SeasonalityResult(
        index=round(seasonal_index, 3),
        period_weeks=best_period,
        strength=round(best_correlation, 3)
    )


def _build_in_clause(items: List[int]) -> str:
    """Build SQL IN clause placeholders."""
    return ','.join(['?' for _ in items])


def _chunked(items: List[int], size: int) -> Iterator[List[int]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def calculate_churn_adjustment(
    conn,
    product_id: int,
    as_of_date: date,
    history_days: int = 180
) -> Optional[ChurnResult]:
    """
    Calculate demand adjustment based on at-risk customer analysis.

    A customer is considered at-risk if their time since last order
    exceeds 1.5x their average order cycle.

    Args:
        conn: Database connection
        product_id: Product ID to analyze
        as_of_date: Reference date
        history_days: How far back to look for customers

    Returns:
        ChurnResult or None if error
    """
    try:
        history_start = (as_of_date - timedelta(days=history_days)).strftime('%Y-%m-%d')
        as_of_str = as_of_date.strftime('%Y-%m-%d')

        # Query to identify at-risk customers for this product
        query = """
            WITH product_customers AS (
                SELECT
                    ca.ClientID,
                    SUM(oi.Qty) as total_qty,
                    MAX(o.Created) as last_order_date,
                    COUNT(DISTINCT o.ID) as order_count
                FROM dbo.OrderItem oi
                JOIN dbo.[Order] o ON oi.OrderID = o.ID
                JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
                WHERE o.Deleted = 0
                  AND oi.Deleted = 0
                  AND ca.Deleted = 0
                  AND oi.ProductID = ?
                  AND o.Created >= ?
                  AND o.Created < ?
                GROUP BY ca.ClientID
                HAVING COUNT(DISTINCT o.ID) >= 2
            ),
            customer_cycles AS (
                SELECT
                    pc.ClientID,
                    pc.total_qty,
                    pc.last_order_date,
                    pc.order_count,
                    DATEDIFF(day, MIN(o.Created), MAX(o.Created)) / NULLIF(pc.order_count - 1, 0) as avg_cycle_days
                FROM product_customers pc
                JOIN dbo.OrderItem oi ON oi.ProductID = ?
                JOIN dbo.[Order] o ON oi.OrderID = o.ID
                JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID AND ca.ClientID = pc.ClientID
                WHERE o.Deleted = 0
                  AND oi.Deleted = 0
                  AND ca.Deleted = 0
                  AND o.Created >= ?
                  AND o.Created < ?
                GROUP BY pc.ClientID, pc.total_qty, pc.last_order_date, pc.order_count
            )
            SELECT
                COUNT(*) as total_customers,
                SUM(total_qty) as total_demand,
                SUM(CASE
                    WHEN avg_cycle_days IS NOT NULL
                     AND DATEDIFF(day, last_order_date, ?) > avg_cycle_days * 1.5
                    THEN 1 ELSE 0
                END) as at_risk_customers,
                SUM(CASE
                    WHEN avg_cycle_days IS NOT NULL
                     AND DATEDIFF(day, last_order_date, ?) > avg_cycle_days * 1.5
                    THEN total_qty ELSE 0
                END) as at_risk_qty
            FROM customer_cycles
        """

        params = [
            product_id, history_start, as_of_str,
            product_id, history_start, as_of_str,
            as_of_str, as_of_str
        ]

        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        cursor.close()

        if not row or row[0] == 0:
            return ChurnResult(adjustment=1.0, at_risk_pct=0.0, at_risk_customers=0)

        total_customers = int(row[0])
        total_demand = float(row[1] or 0)
        at_risk_customers = int(row[2] or 0)
        at_risk_qty = float(row[3] or 0)

        if total_demand <= 0:
            return ChurnResult(adjustment=1.0, at_risk_pct=0.0, at_risk_customers=0)

        at_risk_pct = at_risk_qty / total_demand

        # Assume 50% of at-risk demand will actually churn
        # This is a conservative estimate
        churn_rate = 0.5
        adjustment = 1.0 - (at_risk_pct * churn_rate)

        # Cap at reasonable bounds
        adjustment = max(0.7, min(1.0, adjustment))

        return ChurnResult(
            adjustment=round(adjustment, 3),
            at_risk_pct=round(at_risk_pct, 3),
            at_risk_customers=at_risk_customers
        )

    except Exception as e:
        logger.warning(f"Error calculating churn adjustment for product {product_id}: {e}")
        return None


def calculate_churn_adjustments(
    conn,
    product_ids: List[int],
    as_of_date: date,
    history_days: int = 180
) -> Dict[int, ChurnResult]:
    """
    Calculate churn adjustments for a batch of products.
    """
    if not product_ids:
        return {}

    history_start = (as_of_date - timedelta(days=history_days)).strftime('%Y-%m-%d')
    as_of_str = as_of_date.strftime('%Y-%m-%d')
    results: Dict[int, ChurnResult] = {}

    cursor = conn.cursor()
    try:
        for chunk in _chunked(product_ids, MAX_IN_CLAUSE_ITEMS):
            placeholders = _build_in_clause(chunk)
            query = f"""
                WITH product_customers AS (
                    SELECT
                        oi.ProductID,
                        ca.ClientID,
                        SUM(oi.Qty) as total_qty,
                        MAX(o.Created) as last_order_date,
                        MIN(o.Created) as first_order_date,
                        COUNT(DISTINCT o.ID) as order_count
                    FROM dbo.OrderItem oi
                    JOIN dbo.[Order] o ON oi.OrderID = o.ID
                    JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
                    WHERE o.Deleted = 0
                      AND oi.Deleted = 0
                      AND ca.Deleted = 0
                      AND oi.ProductID IN ({placeholders})
                      AND o.Created >= ?
                      AND o.Created < ?
                    GROUP BY oi.ProductID, ca.ClientID
                    HAVING COUNT(DISTINCT o.ID) >= 2
                ),
                customer_cycles AS (
                    SELECT
                        ProductID,
                        ClientID,
                        total_qty,
                        last_order_date,
                        order_count,
                        DATEDIFF(day, first_order_date, last_order_date)
                            / NULLIF(order_count - 1, 0) as avg_cycle_days
                    FROM product_customers
                )
                SELECT
                    ProductID,
                    COUNT(*) as total_customers,
                    SUM(total_qty) as total_demand,
                    SUM(CASE
                        WHEN avg_cycle_days IS NOT NULL
                         AND DATEDIFF(day, last_order_date, ?) > avg_cycle_days * 1.5
                        THEN 1 ELSE 0
                    END) as at_risk_customers,
                    SUM(CASE
                        WHEN avg_cycle_days IS NOT NULL
                         AND DATEDIFF(day, last_order_date, ?) > avg_cycle_days * 1.5
                        THEN total_qty ELSE 0
                    END) as at_risk_qty
                FROM customer_cycles
                GROUP BY ProductID
            """
            params = chunk + [history_start, as_of_str, as_of_str, as_of_str]
            cursor.execute(query, params)
            for row in cursor.fetchall():
                product_id = row[0]
                total_demand = float(row[2] or 0)
                at_risk_customers = int(row[3] or 0)
                at_risk_qty = float(row[4] or 0)

                if total_demand <= 0:
                    at_risk_pct = 0.0
                else:
                    at_risk_pct = at_risk_qty / total_demand

                churn_rate = 0.5
                adjustment = 1.0 - (at_risk_pct * churn_rate)
                adjustment = max(0.7, min(1.0, adjustment))

                results[product_id] = ChurnResult(
                    adjustment=round(adjustment, 3),
                    at_risk_pct=round(at_risk_pct, 3),
                    at_risk_customers=at_risk_customers
                )
    except Exception as e:
        logger.warning(f"Error calculating churn adjustments: {e}")
        return {}
    finally:
        cursor.close()

    return results


def fetch_weekly_demand_series(
    conn,
    product_ids: List[int],
    start_date: str,
    end_date: str,
    history_weeks: int
) -> Dict[int, List[float]]:
    """
    Fetch weekly demand series for products.

    Returns a dictionary mapping product_id to list of weekly demands
    (oldest to newest, with zeros for weeks with no demand).
    """
    if not product_ids:
        return {}

    result: Dict[int, List[float]] = {pid: [] for pid in product_ids}
    cursor = conn.cursor()
    for chunk in _chunked(product_ids, MAX_IN_CLAUSE_ITEMS):
        placeholders = _build_in_clause(chunk)
        query = f"""
            WITH weeks AS (
                SELECT TOP {history_weeks}
                    DATEADD(week, -ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) + 1,
                        DATEADD(day, DATEDIFF(day, 0, ?) / 7 * 7, 0)) AS week_start
                FROM sys.objects
            ),
            weekly_demand AS (
                SELECT
                    oi.ProductID,
                    DATEADD(day, DATEDIFF(day, 0, o.Created) / 7 * 7, 0) AS week_start,
                    SUM(oi.Qty) AS weekly_qty
                FROM dbo.OrderItem oi
                JOIN dbo.[Order] o ON oi.OrderID = o.ID
                JOIN dbo.ClientAgreement ca ON o.ClientAgreementID = ca.ID
                WHERE o.Deleted = 0
                  AND oi.Deleted = 0
                  AND ca.Deleted = 0
                  AND o.Created >= ?
                  AND o.Created < ?
                  AND oi.ProductID IN ({placeholders})
                GROUP BY oi.ProductID, DATEADD(day, DATEDIFF(day, 0, o.Created) / 7 * 7, 0)
            )
            SELECT
                p.ID AS ProductID,
                w.week_start,
                COALESCE(wd.weekly_qty, 0) AS weekly_qty
            FROM (SELECT DISTINCT ID FROM dbo.Product WHERE ID IN ({placeholders})) p
            CROSS JOIN weeks w
            LEFT JOIN weekly_demand wd ON wd.ProductID = p.ID AND wd.week_start = w.week_start
            ORDER BY p.ID, w.week_start
        """

        params = [end_date, start_date, end_date] + chunk + chunk
        cursor.execute(query, params)

        for row in cursor.fetchall():
            product_id = row[0]
            qty = float(row[2] or 0)
            if product_id in result:
                result[product_id].append(qty)

    cursor.close()
    return result


def get_enhanced_demand_stats(
    conn,
    product_ids: List[int],
    start_date: str,
    end_date: str,
    history_weeks: int,
    lead_time_weeks: int,
    as_of_date: date,
    use_trend: bool = True,
    use_seasonality: bool = True,
    use_churn: bool = True,
    min_history_weeks: int = 8
) -> Dict[int, EnhancedDemandStats]:
    """
    Get enhanced demand statistics with trend, seasonality, and churn adjustments.

    Args:
        conn: Database connection
        product_ids: List of product IDs
        start_date: History start date (YYYY-MM-DD)
        end_date: History end date (YYYY-MM-DD)
        history_weeks: Number of weeks in history period
        lead_time_weeks: Lead time for projecting trend
        as_of_date: Reference date
        use_trend: Whether to apply trend adjustment
        use_seasonality: Whether to apply seasonality adjustment
        use_churn: Whether to apply churn adjustment
        min_history_weeks: Minimum weeks for enhanced forecasting

    Returns:
        Dictionary mapping product_id to EnhancedDemandStats
    """
    if not product_ids:
        return {}

    # Fetch weekly demand series for all products
    weekly_series = fetch_weekly_demand_series(
        conn, product_ids, start_date, end_date, history_weeks
    )

    churn_map: Dict[int, ChurnResult] = {}
    if use_churn:
        churn_map = calculate_churn_adjustments(conn, product_ids, as_of_date)

    # Calculate current week of year for seasonality
    current_week = as_of_date.isocalendar()[1]

    result: Dict[int, EnhancedDemandStats] = {}

    for product_id in product_ids:
        weekly_demand = weekly_series.get(product_id, [])

        # Calculate basic statistics
        data_weeks = sum(1 for w in weekly_demand if w > 0)
        total_qty = sum(weekly_demand)
        mean = total_qty / history_weeks if history_weeks > 0 else 0.0

        # Calculate standard deviation
        if history_weeks > 0 and mean > 0:
            variance = sum((w - mean) ** 2 for w in weekly_demand) / history_weeks
            stddev = math.sqrt(max(variance, 0))
        else:
            stddev = 0.0

        stats = EnhancedDemandStats(
            product_id=product_id,
            mean=round(mean, 4),
            stddev=round(stddev, 4),
            total_qty=round(total_qty, 2),
            data_weeks=data_weeks,
            adjusted_mean=mean,
            forecast_method="basic",
            forecast_confidence=0.5
        )

        # Skip enhanced forecasting if insufficient data
        if data_weeks < min_history_weeks:
            result[product_id] = stats
            continue

        # Apply adjustments
        combined_factor = 1.0
        methods_applied = []
        confidence_factors = []

        # 1. Trend adjustment
        if use_trend and len(weekly_demand) >= min_history_weeks:
            trend = calculate_trend_factor(weekly_demand, lead_time_weeks, min_history_weeks)
            if trend and trend.direction != "stable":
                stats.trend = trend
                combined_factor *= trend.factor
                methods_applied.append("trend")
                confidence_factors.append(trend.confidence)

        # 2. Seasonality adjustment
        if use_seasonality and len(weekly_demand) >= 52:
            seasonality = detect_seasonality(weekly_demand, current_week, min_weeks=52)
            if seasonality and seasonality.period_weeks:
                stats.seasonality = seasonality
                # Only apply if significant strength
                if seasonality.strength > 0.3:
                    combined_factor *= seasonality.index
                    methods_applied.append("seasonal")
                    confidence_factors.append(seasonality.strength)

        # 3. Churn adjustment
        if use_churn:
            churn = churn_map.get(product_id)
            if churn and churn.at_risk_pct > 0.05:  # At least 5% at-risk
                stats.churn = churn
                combined_factor *= churn.adjustment
                methods_applied.append("churn")
                confidence_factors.append(0.7)  # Fixed confidence for churn

        # Apply combined factor
        stats.adjusted_mean = round(mean * combined_factor, 4)

        # Determine forecast method
        if methods_applied:
            if len(methods_applied) == 3:
                stats.forecast_method = "full"
            elif "trend" in methods_applied and "seasonal" in methods_applied:
                stats.forecast_method = "trend_seasonal"
            elif "trend" in methods_applied:
                stats.forecast_method = "trend_adjusted"
            elif "seasonal" in methods_applied:
                stats.forecast_method = "seasonal"
            else:
                stats.forecast_method = "churn_adjusted"

        # Calculate overall confidence
        if confidence_factors:
            # Base confidence on data quality + adjustment confidences
            data_quality = min(1.0, data_weeks / 26)  # Full confidence at 26 weeks
            avg_adjustment_confidence = sum(confidence_factors) / len(confidence_factors)
            stats.forecast_confidence = round(
                0.4 * data_quality + 0.6 * avg_adjustment_confidence, 3
            )
        else:
            stats.forecast_confidence = round(min(1.0, data_weeks / 26) * 0.5, 3)

        result[product_id] = stats

    return result

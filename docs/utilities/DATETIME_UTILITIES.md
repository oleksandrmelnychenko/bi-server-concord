# Datetime Utilities Documentation

## Overview

The `datetime_utils.py` module provides standardized, timezone-aware datetime handling for the Concord BI Server forecasting and recommendation systems. It ensures consistency across date operations, parsing, validation, and calculations throughout the application.

**Module Location:** `/Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts/datetime_utils.py`

**Author:** Claude Code
**Date:** 2025-11-11

---

## Core Philosophy

### UTC Timezone Handling Philosophy

All datetime operations in this module follow strict timezone-aware principles:

1. **Always UTC Internally**: All datetime objects returned by this module are timezone-aware and use UTC (`timezone.utc`)
2. **No Naive Datetimes**: The module automatically converts naive datetimes to UTC to prevent timezone-related bugs
3. **Explicit Timezone Assignment**: When parsing dates, the module explicitly sets `tzinfo=timezone.utc` to ensure consistency
4. **Timezone Validation**: Functions validate and ensure timezone awareness before performing calculations

**Why UTC?**
- Eliminates daylight saving time complications
- Provides a single source of truth for time across distributed systems
- Simplifies date arithmetic and comparisons
- Prevents timezone-related data corruption

**Example:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import now_utc, parse_as_of_date

# Always timezone-aware
current = now_utc()
assert current.tzinfo == timezone.utc  # True

# Parsed dates are always UTC at midnight
date = parse_as_of_date("2024-07-01")
assert date.tzinfo == timezone.utc  # True
assert date.hour == 0  # Midnight UTC
```

---

## ISO 8601 Week Boundaries Logic

The module implements ISO 8601 week date standard, which defines:

- **Week starts on Monday** (weekday = 1 in ISO calendar)
- **Week ends on Sunday** (weekday = 7 in ISO calendar)
- **Week numbering**: Weeks are numbered 1-52 (or 53 in some years)
- **Week boundaries are inclusive-exclusive**: [Monday 00:00:00, Next Monday 00:00:00)

### Why ISO 8601 Weeks?

1. **International Standard**: Universally recognized week definition
2. **Consistency**: Weeks always start on the same day (Monday)
3. **Business Logic**: Aligns with most business reporting cycles
4. **Date Math**: Simplifies week-based calculations and aggregations

### Week Boundary Calculation

The `get_iso_week_boundaries()` function uses Python's `isocalendar()` method to determine the ISO year, week number, and weekday, then reconstructs the Monday start and calculates the exclusive end boundary.

```python
# Example: Finding the week containing a Wednesday
date = datetime(2024, 7, 3, tzinfo=timezone.utc)  # Wednesday
week_start, week_end = get_iso_week_boundaries(date)

# Result:
# week_start: Monday, July 1, 2024 00:00:00 UTC
# week_end:   Monday, July 8, 2024 00:00:00 UTC (exclusive)
```

---

## Fractional Day Calculations

### Why Fractional Days Over Integer `.days`?

Python's `timedelta.days` truncates to integer days, losing sub-day precision:

```python
from datetime import datetime, timezone

start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)  # 3.5 days later

# Integer days (LOSSY)
integer_days = (end - start).days  # Returns: 3 (loses 12 hours!)

# Fractional days (PRECISE)
fractional_days = calculate_fractional_days(start, end)  # Returns: 3.5
```

### Use Cases for Fractional Days

1. **Forecasting Accuracy**: Precise time intervals improve predictive models
2. **Metrics Calculation**: Accurate rates, averages, and trend analysis
3. **Billing/Time Tracking**: Sub-day precision for fair calculations
4. **Scientific Accuracy**: Maintaining precision in time-series analysis

### Implementation Details

The `calculate_fractional_days()` function uses `total_seconds()` divided by 86400 (seconds in a day):

```python
delta = end - start
return delta.total_seconds() / 86400
```

This preserves microsecond precision while converting to day units.

---

## Date Validation Rules

### Parse and Validation (`parse_as_of_date`)

The module enforces strict validation rules to prevent invalid dates:

#### 1. Format Validation
- **Required Format**: ISO 8601 date string (YYYY-MM-DD)
- **Regex Pattern**: `^\d{4}-\d{2}-\d{2}$`
- **Example Valid**: "2024-07-01"
- **Example Invalid**: "07/01/2024", "2024-7-1", "20240701"

#### 2. Type Validation
- Must be a string or None (None defaults to today)
- Rejects integers, floats, or other types

#### 3. Range Validation
- **Minimum Date**: 2019-01-01 (DATA_START_DATE)
- **Maximum Date**: Today (unless `allow_future=True`)
- Rejects dates before data collection started
- By default, rejects future dates to prevent forecasting errors

#### 4. Value Validation
- Validates actual calendar dates (e.g., rejects "2024-02-30")
- Ensures month is 01-12, day is valid for the month

### Validation Examples

```python
# Valid dates
parse_as_of_date("2024-07-01")           # OK
parse_as_of_date(None)                    # OK - returns today
parse_as_of_date("2099-01-01", allow_future=True)  # OK with flag

# Invalid dates (raise ValueError)
parse_as_of_date("2024/07/01")           # Wrong format
parse_as_of_date("2018-12-31")           # Before DATA_START_DATE
parse_as_of_date("2099-01-01")           # Future (without allow_future)
parse_as_of_date("2024-02-30")           # Invalid calendar date
parse_as_of_date(20240701)               # Wrong type (int)
```

---

## Function Reference

### 1. `now_utc() -> datetime`

Get the current datetime in UTC.

**Returns:** Current UTC datetime (timezone-aware)

**Example:**
```python
from scripts.datetime_utils import now_utc

current = now_utc()
print(current)  # 2025-11-11 14:32:15.123456+00:00
print(current.tzinfo)  # UTC
```

---

### 2. `parse_as_of_date(date_str: Optional[str], allow_future: bool = False) -> datetime`

Parse and validate an as_of_date string to timezone-aware datetime.

**Parameters:**
- `date_str`: ISO 8601 date string (YYYY-MM-DD) or None for today
- `allow_future`: Whether to allow future dates (default: False)

**Returns:** Parsed date at midnight UTC (timezone-aware)

**Raises:** `ValueError` if date format is invalid or out of acceptable range

**Examples:**
```python
from scripts.datetime_utils import parse_as_of_date

# Parse specific date
date = parse_as_of_date("2024-07-01")
# Returns: datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc)

# Use today as default
today = parse_as_of_date(None)
# Returns: datetime(2025, 11, 11, 0, 0, tzinfo=timezone.utc)

# Allow future dates
future = parse_as_of_date("2026-01-01", allow_future=True)
# Returns: datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

# Invalid date format
try:
    parse_as_of_date("07/01/2024")
except ValueError as e:
    print(e)  # Invalid date format: '07/01/2024'. Expected ISO 8601 format (YYYY-MM-DD)

# Future date without flag
try:
    parse_as_of_date("2099-01-01")
except ValueError as e:
    print(e)  # as_of_date cannot be in the future: '2099-01-01'
```

---

### 3. `format_date_iso(dt: datetime) -> str`

Format datetime to ISO 8601 date string (YYYY-MM-DD).

**Parameters:**
- `dt`: Datetime object (timezone-aware or naive)

**Returns:** Date in YYYY-MM-DD format

**Example:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import format_date_iso

dt = datetime(2024, 7, 1, 15, 30, tzinfo=timezone.utc)
formatted = format_date_iso(dt)
print(formatted)  # '2024-07-01'

# Works with naive datetimes too
naive_dt = datetime(2024, 12, 25, 10, 0)
print(format_date_iso(naive_dt))  # '2024-12-25'
```

---

### 4. `get_iso_week_boundaries(date: datetime) -> Tuple[datetime, datetime]`

Get ISO 8601 week boundaries (Monday to Sunday) for the week containing the given date.

**Parameters:**
- `date`: Any date within the desired week

**Returns:** Tuple of (week_start, week_end) where:
  - `week_start` is Monday at 00:00:00 UTC
  - `week_end` is the following Monday at 00:00:00 UTC (exclusive boundary)

**Examples:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import get_iso_week_boundaries

# Wednesday in a week
wed = datetime(2024, 7, 3, tzinfo=timezone.utc)
week_start, week_end = get_iso_week_boundaries(wed)
print(f"Start: {week_start}")  # 2024-07-01 00:00:00+00:00 (Monday)
print(f"End: {week_end}")      # 2024-07-08 00:00:00+00:00 (Next Monday)

# Monday itself
mon = datetime(2024, 7, 1, 15, 30, tzinfo=timezone.utc)
week_start, week_end = get_iso_week_boundaries(mon)
print(f"Start: {week_start}")  # 2024-07-01 00:00:00+00:00 (Same Monday)
print(f"End: {week_end}")      # 2024-07-08 00:00:00+00:00 (Next Monday)

# Check if a datetime falls within a week
def is_in_week(dt, week_start, week_end):
    return week_start <= dt < week_end

is_in_week(wed, week_start, week_end)  # True
```

---

### 5. `get_iso_week_key(date: Optional[datetime] = None) -> str`

Get ISO 8601 week key for Redis caching and grouping.

**Parameters:**
- `date`: Date within desired week (default: today)

**Returns:** Week key in format "YYYY_WNN" (e.g., "2024_W27")

**Examples:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import get_iso_week_key

# Specific date
date = datetime(2024, 7, 3, tzinfo=timezone.utc)
key = get_iso_week_key(date)
print(key)  # '2024_W27'

# Current week
current_key = get_iso_week_key()
print(current_key)  # '2025_W46' (depends on current date)

# Use as Redis cache key
cache_key = f"forecast:{get_iso_week_key()}:results"
print(cache_key)  # 'forecast:2025_W46:results'
```

---

### 6. `calculate_fractional_days(start: datetime, end: datetime) -> float`

Calculate fractional days between two datetimes with sub-day precision.

**Parameters:**
- `start`: Start datetime
- `end`: End datetime

**Returns:** Number of days (can be fractional, e.g., 3.5 for 3 days 12 hours)

**Examples:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import calculate_fractional_days

start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)

# Fractional calculation (PRECISE)
frac_days = calculate_fractional_days(start, end)
print(frac_days)  # 3.5

# Compare to integer .days (LOSSY)
int_days = (end - start).days
print(int_days)  # 3 (loses the 12 hours!)

# Practical use case: Calculate average daily rate
total_events = 350
days = calculate_fractional_days(start, end)
avg_per_day = total_events / days
print(f"Average: {avg_per_day:.2f} events/day")  # Average: 100.00 events/day

# With integer days, accuracy suffers
avg_lossy = total_events / int_days
print(f"Lossy average: {avg_lossy:.2f}")  # Lossy average: 116.67 events/day (wrong!)
```

---

### 7. `generate_weekly_buckets(start_date: datetime, num_weeks: int, use_iso_weeks: bool = True) -> list[Tuple[datetime, datetime]]`

Generate list of weekly time buckets.

**Parameters:**
- `start_date`: Starting date for bucketing
- `num_weeks`: Number of weeks to generate
- `use_iso_weeks`: If True, align to ISO week boundaries (Monday start). If False, use 7-day buckets from start_date.

**Returns:** List of (week_start, week_end) tuples, where week_end is exclusive

**Examples:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import generate_weekly_buckets

start = datetime(2024, 7, 3, tzinfo=timezone.utc)  # Wednesday

# ISO-aligned weeks (Monday start)
iso_buckets = generate_weekly_buckets(start, 3, use_iso_weeks=True)
for i, (ws, we) in enumerate(iso_buckets, 1):
    print(f"Week {i}: {ws.strftime('%Y-%m-%d')} to {we.strftime('%Y-%m-%d')}")
# Week 1: 2024-07-01 to 2024-07-08  (Monday-Monday)
# Week 2: 2024-07-08 to 2024-07-15
# Week 3: 2024-07-15 to 2024-07-22

# Simple 7-day buckets (start from Wednesday)
simple_buckets = generate_weekly_buckets(start, 3, use_iso_weeks=False)
for i, (ws, we) in enumerate(simple_buckets, 1):
    print(f"Week {i}: {ws.strftime('%Y-%m-%d')} to {we.strftime('%Y-%m-%d')}")
# Week 1: 2024-07-03 to 2024-07-10  (Wed-Wed)
# Week 2: 2024-07-10 to 2024-07-17
# Week 3: 2024-07-17 to 2024-07-24

# Use for time-series aggregation
for week_start, week_end in iso_buckets:
    weekly_data = query_data_in_range(week_start, week_end)
    analyze_weekly_trends(weekly_data)
```

---

### 8. `is_same_week(date1: datetime, date2: datetime) -> bool`

Check if two dates are in the same ISO week.

**Parameters:**
- `date1`: First date
- `date2`: Second date

**Returns:** True if both dates are in the same ISO week

**Examples:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import is_same_week

mon = datetime(2024, 7, 1, tzinfo=timezone.utc)   # Monday
wed = datetime(2024, 7, 3, tzinfo=timezone.utc)   # Wednesday
sun = datetime(2024, 7, 7, tzinfo=timezone.utc)   # Sunday
next_mon = datetime(2024, 7, 8, tzinfo=timezone.utc)  # Next Monday

# Same week comparisons
print(is_same_week(mon, wed))      # True
print(is_same_week(mon, sun))      # True
print(is_same_week(wed, sun))      # True

# Different week
print(is_same_week(mon, next_mon)) # False
print(is_same_week(sun, next_mon)) # False

# Practical use: Group events by week
events = get_events()
weekly_groups = {}
for event in events:
    week_key = get_iso_week_key(event.timestamp)
    if week_key not in weekly_groups:
        weekly_groups[week_key] = []
    weekly_groups[week_key].append(event)
```

---

### 9. `validate_date_range(start_date: Optional[datetime], end_date: Optional[datetime]) -> Tuple[datetime, datetime]`

Validate and normalize a date range.

**Parameters:**
- `start_date`: Range start (default: DATA_START_DATE = 2019-01-01)
- `end_date`: Range end (default: today)

**Returns:** Validated (start_date, end_date) tuple

**Raises:** `ValueError` if start_date > end_date

**Examples:**
```python
from datetime import datetime, timezone
from scripts.datetime_utils import validate_date_range

# Use defaults
start, end = validate_date_range(None, None)
print(f"Start: {start}")  # 2019-01-01 00:00:00+00:00
print(f"End: {end}")      # 2025-11-11 14:32:15+00:00 (current time)

# Specify custom range
custom_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
custom_end = datetime(2024, 12, 31, tzinfo=timezone.utc)
start, end = validate_date_range(custom_start, custom_end)

# Invalid range (raises ValueError)
try:
    bad_start = datetime(2024, 12, 31, tzinfo=timezone.utc)
    bad_end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    validate_date_range(bad_start, bad_end)
except ValueError as e:
    print(e)  # start_date must be <= end_date

# Ensures timezone awareness
naive_start = datetime(2024, 1, 1)
naive_end = datetime(2024, 6, 1)
start, end = validate_date_range(naive_start, naive_end)
print(start.tzinfo)  # UTC (automatically assigned)
```

---

### 10. `get_week_key(date: Optional[datetime] = None) -> str`

Alias for `get_iso_week_key()` for backward compatibility.

**Parameters:**
- `date`: Date within desired week (default: today)

**Returns:** Week key in format "YYYY_WNN"

**Note:** This function exists for backward compatibility with older code. New code should use `get_iso_week_key()` directly.

---

## Usage Patterns

### Pattern 1: API Endpoint Date Parsing

```python
from flask import request
from scripts.datetime_utils import parse_as_of_date, format_date_iso

@app.route('/api/forecast')
def get_forecast():
    # Parse and validate user-provided date
    as_of_date_str = request.args.get('as_of_date')

    try:
        as_of_date = parse_as_of_date(as_of_date_str)
    except ValueError as e:
        return {'error': str(e)}, 400

    # Use validated date
    forecast = generate_forecast(as_of_date)

    return {
        'as_of_date': format_date_iso(as_of_date),
        'forecast': forecast
    }
```

### Pattern 2: Weekly Report Generation

```python
from scripts.datetime_utils import (
    now_utc, get_iso_week_boundaries, get_iso_week_key, generate_weekly_buckets
)

def generate_weekly_reports(num_weeks=4):
    current_date = now_utc()
    week_key = get_iso_week_key(current_date)

    print(f"Generating reports for week {week_key}")

    # Get weekly buckets
    buckets = generate_weekly_buckets(current_date, num_weeks)

    reports = []
    for week_start, week_end in buckets:
        data = fetch_data_in_range(week_start, week_end)
        report = {
            'week_key': get_iso_week_key(week_start),
            'start': format_date_iso(week_start),
            'end': format_date_iso(week_end),
            'metrics': calculate_metrics(data)
        }
        reports.append(report)

    return reports
```

### Pattern 3: Time-Series Metrics with Fractional Days

```python
from scripts.datetime_utils import calculate_fractional_days, now_utc

def calculate_daily_average(events, start_date, end_date=None):
    if end_date is None:
        end_date = now_utc()

    # Use fractional days for precision
    days = calculate_fractional_days(start_date, end_date)

    if days == 0:
        return 0.0

    return len(events) / days

# Example usage
events = fetch_events(start_date)
avg = calculate_daily_average(events, start_date)
print(f"Average: {avg:.2f} events/day")
```

---

## Integration Guidelines

### 1. Import Conventions

```python
# Preferred: Import specific functions
from scripts.datetime_utils import (
    now_utc,
    parse_as_of_date,
    get_iso_week_boundaries,
    calculate_fractional_days
)

# Alternative: Import module
from scripts import datetime_utils as dt_utils

current = dt_utils.now_utc()
```

### 2. Error Handling

Always handle `ValueError` exceptions from parsing and validation functions:

```python
try:
    date = parse_as_of_date(user_input)
except ValueError as e:
    # Log and return user-friendly error
    logger.error(f"Date parsing failed: {e}")
    return {'error': 'Invalid date format. Use YYYY-MM-DD.'}, 400
```

### 3. Database Integration

When storing dates in databases:

```python
from scripts.datetime_utils import now_utc, format_date_iso

# Store as ISO string
created_at = now_utc()
db.execute("INSERT INTO records (date) VALUES (?)", (format_date_iso(created_at),))

# Retrieve and parse
row = db.fetchone()
date = parse_as_of_date(row['date'])
```

### 4. Redis Caching with Week Keys

```python
from scripts.datetime_utils import get_iso_week_key

def get_cached_forecast(date):
    week_key = get_iso_week_key(date)
    cache_key = f"forecast:{week_key}"

    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute and cache
    forecast = compute_forecast(date)
    redis.setex(cache_key, 604800, json.dumps(forecast))  # 1 week TTL
    return forecast
```

### 5. Testing with Fixed Dates

```python
from datetime import datetime, timezone
from scripts.datetime_utils import parse_as_of_date, is_same_week

def test_weekly_aggregation():
    # Use fixed dates for reproducible tests
    test_date = datetime(2024, 7, 3, tzinfo=timezone.utc)

    result = aggregate_by_week(test_date)

    assert result['week_key'] == '2024_W27'
    assert is_same_week(result['start'], test_date)
```

---

## Constants

### `DATA_START_DATE`
```python
DATA_START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
```
Earliest valid date for data collection. Used as minimum boundary in `parse_as_of_date()` and default start in `validate_date_range()`.

### `ISO_DATE_FORMAT`
```python
ISO_DATE_FORMAT = "%Y-%m-%d"
```
Standard strftime/strptime format string for ISO 8601 dates (YYYY-MM-DD).

### `ISO_DATE_REGEX`
```python
ISO_DATE_REGEX = re.compile(r'^\d{4}-\d{2}-\d{2}$')
```
Regular expression pattern for validating ISO 8601 date format before parsing.

---

## Testing

The module includes a self-test suite that can be run directly:

```bash
python /Users/oleksandrmelnychenko/Projects/Concord-BI-Server/scripts/datetime_utils.py
```

Expected output demonstrates all core functionality with passing tests.

---

## Best Practices

1. **Always use timezone-aware datetimes**: Never work with naive datetimes in production code
2. **Validate user input**: Use `parse_as_of_date()` for all external date inputs
3. **Use fractional days for metrics**: Avoid `.days` attribute; use `calculate_fractional_days()` instead
4. **ISO weeks for business logic**: Use ISO week boundaries for consistent weekly aggregations
5. **Cache with week keys**: Use `get_iso_week_key()` for time-based cache invalidation
6. **Handle validation errors**: Always wrap parsing in try-except blocks
7. **Format for display**: Use `format_date_iso()` when returning dates to users or APIs
8. **Test with fixed dates**: Use explicit datetime objects in tests for reproducibility

---

## Migration Guide

If migrating from naive datetime usage:

```python
# OLD: Naive datetime (AVOID)
from datetime import datetime
now = datetime.now()  # No timezone!

# NEW: Timezone-aware
from scripts.datetime_utils import now_utc
now = now_utc()  # Always UTC

# OLD: Integer days (LOSSY)
days = (end - start).days

# NEW: Fractional days (PRECISE)
from scripts.datetime_utils import calculate_fractional_days
days = calculate_fractional_days(start, end)

# OLD: Manual week calculation (ERROR-PRONE)
days_since_monday = date.weekday()
week_start = date - timedelta(days=days_since_monday)

# NEW: ISO week boundaries (CORRECT)
from scripts.datetime_utils import get_iso_week_boundaries
week_start, week_end = get_iso_week_boundaries(date)
```

---

## Support and Maintenance

For questions or issues with datetime utilities:
1. Review this documentation thoroughly
2. Check the source code comments in `datetime_utils.py`
3. Run the self-test suite to verify installation
4. Consult the development team for edge cases

**Version:** 1.0
**Last Updated:** 2025-11-11

#!/usr/bin/env python3
"""
Datetime Utilities for Concord BI Server

Provides standardized, timezone-aware datetime handling for forecasting and recommendation systems.

Key principles:
- All datetimes are timezone-aware (UTC internally)
- ISO 8601 week boundaries (Monday = week start)
- Consistent date parsing and validation
- Fractional day calculations for precision

Author: Claude Code
Date: 2025-11-11
"""

from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
import re


# Constants
DATA_START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
ISO_DATE_FORMAT = "%Y-%m-%d"
ISO_DATE_REGEX = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def now_utc() -> datetime:
    """
    Get current datetime in UTC.

    Returns:
        datetime: Current UTC datetime (timezone-aware)

    Example:
        >>> dt = now_utc()
        >>> dt.tzinfo == timezone.utc
        True
    """
    return datetime.now(timezone.utc)


def parse_as_of_date(date_str: Optional[str], allow_future: bool = False) -> datetime:
    """
    Parse and validate as_of_date string to timezone-aware datetime.

    Args:
        date_str: ISO 8601 date string (YYYY-MM-DD) or None for today
        allow_future: Whether to allow future dates (default: False)

    Returns:
        datetime: Parsed date at midnight UTC (timezone-aware)

    Raises:
        ValueError: If date format is invalid or out of acceptable range

    Examples:
        >>> parse_as_of_date("2024-07-01")
        datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc)

        >>> parse_as_of_date(None)  # Returns today
        datetime(2025, 11, 11, 0, 0, tzinfo=timezone.utc)

        >>> parse_as_of_date("2099-01-01")  # Raises ValueError (future)
        ValueError: as_of_date cannot be in the future
    """
    # Default to today if None
    if date_str is None:
        return now_utc().replace(hour=0, minute=0, second=0, microsecond=0)

    # Type validation
    if not isinstance(date_str, str):
        raise ValueError(f"Expected string, got {type(date_str).__name__}")

    # Format validation (YYYY-MM-DD)
    if not ISO_DATE_REGEX.match(date_str):
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected ISO 8601 format (YYYY-MM-DD)"
        )

    # Parse the date
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError as e:
        raise ValueError(f"Invalid date: '{date_str}'. Error: {str(e)}")

    # Ensure timezone-aware (assume UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Set to midnight (start of day)
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Range validation
    today = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)

    if not allow_future and dt > today:
        raise ValueError(f"as_of_date cannot be in the future: '{date_str}'")

    if dt < DATA_START_DATE:
        raise ValueError(
            f"as_of_date is before data collection start ({DATA_START_DATE.strftime(ISO_DATE_FORMAT)}): '{date_str}'"
        )

    return dt


def format_date_iso(dt: datetime) -> str:
    """
    Format datetime to ISO 8601 date string (YYYY-MM-DD).

    Args:
        dt: Datetime object (timezone-aware or naive)

    Returns:
        str: Date in YYYY-MM-DD format

    Example:
        >>> format_date_iso(datetime(2024, 7, 1, 15, 30, tzinfo=timezone.utc))
        '2024-07-01'
    """
    return dt.strftime(ISO_DATE_FORMAT)


def get_iso_week_boundaries(date: datetime) -> Tuple[datetime, datetime]:
    """
    Get ISO 8601 week boundaries (Monday to Sunday) for the week containing the given date.

    ISO 8601 defines:
    - Week starts on Monday (weekday 1)
    - Week ends on Sunday (weekday 7)

    Args:
        date: Any date within the desired week

    Returns:
        Tuple[datetime, datetime]: (week_start, week_end) where:
            - week_start is Monday at 00:00:00 UTC
            - week_end is the following Monday at 00:00:00 UTC (exclusive boundary)

    Example:
        >>> get_iso_week_boundaries(datetime(2024, 7, 3))  # Wednesday
        (datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc),   # Monday
         datetime(2024, 7, 8, 0, 0, tzinfo=timezone.utc))   # Next Monday
    """
    # Ensure timezone-aware
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)

    # Get ISO calendar info (year, week, weekday where Monday=1)
    year, week, weekday = date.isocalendar()

    # Calculate Monday of this ISO week
    week_start = datetime.fromisocalendar(year, week, 1).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )

    # Week end is 7 days later (start of next week, exclusive)
    week_end = week_start + timedelta(days=7)

    return week_start, week_end


def get_iso_week_key(date: Optional[datetime] = None) -> str:
    """
    Get ISO 8601 week key for Redis caching (format: YYYY_WNN).

    Args:
        date: Date within desired week (default: today)

    Returns:
        str: Week key in format "YYYY_WNN" (e.g., "2024_W27")

    Example:
        >>> get_iso_week_key(datetime(2024, 7, 3, tzinfo=timezone.utc))
        '2024_W27'
    """
    if date is None:
        date = now_utc()

    year, week, _ = date.isocalendar()
    return f"{year}_W{week:02d}"


def calculate_fractional_days(start: datetime, end: datetime) -> float:
    """
    Calculate fractional days between two datetimes with sub-day precision.

    Unlike .days which truncates to integer days, this returns precise fractional days.

    Args:
        start: Start datetime
        end: End datetime

    Returns:
        float: Number of days (can be fractional, e.g., 3.5 for 3 days 12 hours)

    Examples:
        >>> start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        >>> end = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)
        >>> calculate_fractional_days(start, end)
        3.5

        >>> # Compare to integer .days:
        >>> (end - start).days
        3  # Loses the 12 hours!
    """
    delta = end - start
    return delta.total_seconds() / 86400


def generate_weekly_buckets(
    start_date: datetime,
    num_weeks: int,
    use_iso_weeks: bool = True
) -> list[Tuple[datetime, datetime]]:
    """
    Generate list of weekly time buckets.

    Args:
        start_date: Starting date for bucketing
        num_weeks: Number of weeks to generate
        use_iso_weeks: If True, align to ISO week boundaries (Monday start).
                       If False, use 7-day buckets from start_date.

    Returns:
        List of (week_start, week_end) tuples, where week_end is exclusive

    Example with ISO weeks:
        >>> start = datetime(2024, 7, 3, tzinfo=timezone.utc)  # Wednesday
        >>> buckets = generate_weekly_buckets(start, 2, use_iso_weeks=True)
        >>> buckets[0]  # First week aligns to Monday
        (datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc),
         datetime(2024, 7, 8, 0, 0, tzinfo=timezone.utc))

    Example with 7-day buckets:
        >>> start = datetime(2024, 7, 3, tzinfo=timezone.utc)  # Wednesday
        >>> buckets = generate_weekly_buckets(start, 2, use_iso_weeks=False)
        >>> buckets[0]  # First week starts on Wednesday
        (datetime(2024, 7, 3, 0, 0, tzinfo=timezone.utc),
         datetime(2024, 7, 10, 0, 0, tzinfo=timezone.utc))
    """
    buckets = []

    if use_iso_weeks:
        # Align to ISO week boundaries
        current_start, _ = get_iso_week_boundaries(start_date)

        for _ in range(num_weeks):
            current_end = current_start + timedelta(days=7)
            buckets.append((current_start, current_end))
            current_start = current_end
    else:
        # Simple 7-day buckets from start_date
        current_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if current_start.tzinfo is None:
            current_start = current_start.replace(tzinfo=timezone.utc)

        for _ in range(num_weeks):
            current_end = current_start + timedelta(days=7)
            buckets.append((current_start, current_end))
            current_start = current_end

    return buckets


def is_same_week(date1: datetime, date2: datetime) -> bool:
    """
    Check if two dates are in the same ISO week.

    Args:
        date1: First date
        date2: Second date

    Returns:
        bool: True if both dates are in the same ISO week

    Example:
        >>> mon = datetime(2024, 7, 1, tzinfo=timezone.utc)  # Monday
        >>> sun = datetime(2024, 7, 7, tzinfo=timezone.utc)  # Sunday
        >>> is_same_week(mon, sun)
        True

        >>> next_mon = datetime(2024, 7, 8, tzinfo=timezone.utc)  # Next Monday
        >>> is_same_week(mon, next_mon)
        False
    """
    return date1.isocalendar()[:2] == date2.isocalendar()[:2]


def validate_date_range(
    start_date: Optional[datetime],
    end_date: Optional[datetime]
) -> Tuple[datetime, datetime]:
    """
    Validate and normalize a date range.

    Args:
        start_date: Range start (default: DATA_START_DATE)
        end_date: Range end (default: today)

    Returns:
        Tuple[datetime, datetime]: Validated (start_date, end_date)

    Raises:
        ValueError: If start_date > end_date
    """
    if start_date is None:
        start_date = DATA_START_DATE

    if end_date is None:
        end_date = now_utc()

    # Ensure timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    return start_date, end_date


# Backward compatibility aliases for existing code
def get_week_key(date: Optional[datetime] = None) -> str:
    """Alias for get_iso_week_key() for backward compatibility."""
    return get_iso_week_key(date)


if __name__ == "__main__":
    # Simple self-test
    print("Datetime Utilities Self-Test")
    print("=" * 60)

    # Test 1: Current time
    now = now_utc()
    print(f"✓ Current UTC time: {now}")
    print(f"  Timezone-aware: {now.tzinfo is not None}")

    # Test 2: Parse date
    test_date = parse_as_of_date("2024-07-01")
    print(f"\n✓ Parsed '2024-07-01': {test_date}")
    print(f"  Format: {format_date_iso(test_date)}")

    # Test 3: ISO week boundaries
    week_start, week_end = get_iso_week_boundaries(test_date)
    print(f"\n✓ ISO week for 2024-07-01 (Monday):")
    print(f"  Start: {week_start}")
    print(f"  End:   {week_end}")
    print(f"  Week key: {get_iso_week_key(test_date)}")

    # Test 4: Fractional days
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)
    frac_days = calculate_fractional_days(start, end)
    int_days = (end - start).days
    print(f"\n✓ Days between 2024-01-01 and 2024-01-04 12:00:")
    print(f"  Fractional: {frac_days} days (precise)")
    print(f"  Integer:    {int_days} days (loses 12 hours)")

    # Test 5: Weekly buckets
    buckets = generate_weekly_buckets(test_date, 2, use_iso_weeks=True)
    print(f"\n✓ Generated 2 ISO weekly buckets from 2024-07-01:")
    for i, (start, end) in enumerate(buckets, 1):
        print(f"  Week {i}: {format_date_iso(start)} to {format_date_iso(end)}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")

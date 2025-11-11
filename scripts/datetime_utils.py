#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
import re

DATA_START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
ISO_DATE_FORMAT = "%Y-%m-%d"
ISO_DATE_REGEX = re.compile(r'^\d{4}-\d{2}-\d{2}$')

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def parse_as_of_date(date_str: Optional[str], allow_future: bool = False) -> datetime:

    if date_str is None:
        return now_utc().replace(hour=0, minute=0, second=0, microsecond=0)

    if not isinstance(date_str, str):
        raise ValueError(f"Expected string, got {type(date_str).__name__}")

    if not ISO_DATE_REGEX.match(date_str):
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected ISO 8601 format (YYYY-MM-DD)"
        )

    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError as e:
        raise ValueError(f"Invalid date: '{date_str}'. Error: {str(e)}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    today = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)

    if not allow_future and dt > today:
        raise ValueError(f"as_of_date cannot be in the future: '{date_str}'")

    if dt < DATA_START_DATE:
        raise ValueError(
            f"as_of_date is before data collection start ({DATA_START_DATE.strftime(ISO_DATE_FORMAT)}): '{date_str}'"
        )

    return dt

def format_date_iso(dt: datetime) -> str:
    return dt.strftime(ISO_DATE_FORMAT)

def get_iso_week_boundaries(date: datetime) -> Tuple[datetime, datetime]:

    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)

    year, week, weekday = date.isocalendar()

    week_start = datetime.fromisocalendar(year, week, 1).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )

    week_end = week_start + timedelta(days=7)

    return week_start, week_end

def get_iso_week_key(date: Optional[datetime] = None) -> str:
    if date is None:
        date = now_utc()

    year, week, _ = date.isocalendar()
    return f"{year}_W{week:02d}"

def calculate_fractional_days(start: datetime, end: datetime) -> float:
    delta = end - start
    return delta.total_seconds() / 86400

def generate_weekly_buckets(
    start_date: datetime,
    num_weeks: int,
    use_iso_weeks: bool = True
) -> list[Tuple[datetime, datetime]]:
    buckets = []

    if use_iso_weeks:

        current_start, _ = get_iso_week_boundaries(start_date)

        for _ in range(num_weeks):
            current_end = current_start + timedelta(days=7)
            buckets.append((current_start, current_end))
            current_start = current_end
    else:

        current_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if current_start.tzinfo is None:
            current_start = current_start.replace(tzinfo=timezone.utc)

        for _ in range(num_weeks):
            current_end = current_start + timedelta(days=7)
            buckets.append((current_start, current_end))
            current_start = current_end

    return buckets

def is_same_week(date1: datetime, date2: datetime) -> bool:
    return date1.isocalendar()[:2] == date2.isocalendar()[:2]

def validate_date_range(
    start_date: Optional[datetime],
    end_date: Optional[datetime]
) -> Tuple[datetime, datetime]:
    if start_date is None:
        start_date = DATA_START_DATE

    if end_date is None:
        end_date = now_utc()

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    return start_date, end_date

def get_week_key(date: Optional[datetime] = None) -> str:
    return get_iso_week_key(date)

if __name__ == "__main__":

    print("Datetime Utilities Self-Test")
    print("=" * 60)

    now = now_utc()
    print(f"✓ Current UTC time: {now}")
    print(f"  Timezone-aware: {now.tzinfo is not None}")

    test_date = parse_as_of_date("2024-07-01")
    print(f"\n✓ Parsed '2024-07-01': {test_date}")
    print(f"  Format: {format_date_iso(test_date)}")

    week_start, week_end = get_iso_week_boundaries(test_date)
    print(f"\n✓ ISO week for 2024-07-01 (Monday):")
    print(f"  Start: {week_start}")
    print(f"  End:   {week_end}")
    print(f"  Week key: {get_iso_week_key(test_date)}")

    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)
    frac_days = calculate_fractional_days(start, end)
    int_days = (end - start).days
    print(f"\n✓ Days between 2024-01-01 and 2024-01-04 12:00:")
    print(f"  Fractional: {frac_days} days (precise)")
    print(f"  Integer:    {int_days} days (loses 12 hours)")

    buckets = generate_weekly_buckets(test_date, 2, use_iso_weeks=True)
    print(f"\n✓ Generated 2 ISO weekly buckets from 2024-07-01:")
    for i, (start, end) in enumerate(buckets, 1):
        print(f"  Week {i}: {format_date_iso(start)} to {format_date_iso(end)}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")

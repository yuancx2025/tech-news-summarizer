# src/rag/time_utils.py
"""Utilities for interpreting temporal filters and natural language hints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import dateparser
from dateparser.search import search_dates

from src.rag.schemas import TemporalFilter


def _to_iso_date(dt: datetime) -> str:
    """Normalize to YYYY-MM-DD format (UTC)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _parse_explicit(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        # Pydantic may provide `date`
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    except Exception:
        pass
    parsed = dateparser.parse(str(value), settings={"RETURN_AS_TIMEZONE_AWARE": True})
    return parsed


def _quarter_bounds(now: datetime, offset: int = 0) -> Tuple[datetime, datetime]:
    quarter = ((now.month - 1) // 3) + 1 + offset
    year = now.year
    while quarter <= 0:
        quarter += 4
        year -= 1
    while quarter > 4:
        quarter -= 4
        year += 1
    start_month = 3 * (quarter - 1) + 1
    start = datetime(year, start_month, 1, tzinfo=timezone.utc)
    if quarter == 4:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
    else:
        end = datetime(year, start_month + 3, 1, tzinfo=timezone.utc) - timedelta(days=1)
    return start, end


def parse_natural_range(text: str, now: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Best-effort interpretation of natural language date hints."""
    if not text:
        return None, None
    now = now or datetime.now(timezone.utc)
    lowered = text.lower()

    if "this quarter" in lowered:
        return _quarter_bounds(now, offset=0)
    if "last quarter" in lowered:
        return _quarter_bounds(now, offset=-1)

    settings = {
        "RELATIVE_BASE": now,
        "PREFER_DATES_FROM": "past",
        "RETURN_AS_TIMEZONE_AWARE": True,
    }

    # Handle phrases like "since July" or "from March"
    if "since" in lowered or "from" in lowered:
        matches = search_dates(text, settings=settings)
        if matches:
            start = matches[0][1]
            return start, now

    if "between" in lowered or "to" in lowered:
        matches = search_dates(text, settings=settings)
        if matches and len(matches) >= 2:
            start = matches[0][1]
            end = matches[1][1]
            if start > end:
                start, end = end, start
            return start, end

    parsed = dateparser.parse(text, settings=settings)
    if parsed:
        # Treat single date as starting point until now
        return parsed, now

    matches = search_dates(text, settings=settings)
    if matches:
        start = matches[0][1]
        end = matches[-1][1]
        if start > end:
            start, end = end, start
        return start, end

    return None, None


def resolve_temporal_filter(
    req_filter: Optional[TemporalFilter],
    *,
    default_days_back: int = 60,
    now: Optional[datetime] = None,
) -> Tuple[Optional[str], Optional[str], int]:
    """Resolve explicit/natural filters into ISO date strings and fallback window."""

    now = now or datetime.now(timezone.utc)
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    days_back = default_days_back

    if req_filter:
        if req_filter.days_back is not None:
            days_back = max(1, req_filter.days_back)
        start_dt = _parse_explicit(req_filter.start_date) or None
        end_dt = _parse_explicit(req_filter.end_date) or None

        if not start_dt or not end_dt:
            natural_start, natural_end = parse_natural_range(req_filter.natural_language or "", now)
            start_dt = start_dt or natural_start
            end_dt = end_dt or natural_end

    if start_dt is None and days_back:
        start_dt = now - timedelta(days=days_back)
    if end_dt is None:
        end_dt = now

    start_iso = _to_iso_date(start_dt) if start_dt else None
    end_iso = _to_iso_date(end_dt) if end_dt else None

    return start_iso, end_iso, days_back

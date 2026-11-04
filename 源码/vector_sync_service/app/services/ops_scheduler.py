"""Lightweight cron-like scheduling helpers for the ops service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class CronSchedule:
    minute: set[int]
    hour: set[int]
    day: set[int]
    month: set[int]
    weekday: set[int]


def parse_cron_schedule(expression: str) -> CronSchedule:
    parts = [item.strip() for item in str(expression or "").split() if item.strip()]
    if len(parts) != 5:
        raise ValueError("cron expression must have 5 fields")
    return CronSchedule(
        minute=_parse_field(parts[0], 0, 59),
        hour=_parse_field(parts[1], 0, 23),
        day=_parse_field(parts[2], 1, 31),
        month=_parse_field(parts[3], 1, 12),
        weekday=_parse_field(parts[4], 0, 6),
    )


def next_run_after(schedule: CronSchedule, after: datetime) -> datetime:
    current = _round_to_next_minute(after.astimezone(timezone.utc))
    for _ in range(366 * 24 * 60):
        if _matches(schedule, current):
            return current
        current += timedelta(minutes=1)
    raise RuntimeError("unable to resolve next cron run within one year")


def seconds_until_next_run(expression: str, *, now: datetime | None = None) -> float:
    base = now or datetime.now(timezone.utc)
    target = next_run_after(parse_cron_schedule(expression), base)
    return max((target - base).total_seconds(), 0.0)


def _matches(schedule: CronSchedule, value: datetime) -> bool:
    return (
        value.minute in schedule.minute
        and value.hour in schedule.hour
        and value.day in schedule.day
        and value.month in schedule.month
        and value.weekday() in schedule.weekday
    )


def _round_to_next_minute(value: datetime) -> datetime:
    rounded = value.replace(second=0, microsecond=0)
    if rounded <= value:
        rounded += timedelta(minutes=1)
    return rounded


def _parse_field(token: str, minimum: int, maximum: int) -> set[int]:
    values: set[int] = set()
    for part in token.split(","):
        values.update(_parse_field_part(part.strip(), minimum, maximum))
    if not values:
        raise ValueError(f"empty cron field: {token!r}")
    return values


def _parse_field_part(token: str, minimum: int, maximum: int) -> set[int]:
    if token == "*":
        return set(range(minimum, maximum + 1))
    if "/" in token:
        base, step_text = token.split("/", 1)
        step = int(step_text)
        if step <= 0:
            raise ValueError(f"invalid cron step: {token!r}")
        base_values = _parse_field_part(base or "*", minimum, maximum)
        ordered = sorted(base_values)
        return {value for index, value in enumerate(ordered) if index % step == 0}
    if "-" in token:
        start_text, end_text = token.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if start > end:
            raise ValueError(f"invalid cron range: {token!r}")
        _validate_range(start, minimum, maximum)
        _validate_range(end, minimum, maximum)
        return set(range(start, end + 1))
    value = int(token)
    _validate_range(value, minimum, maximum)
    return {value}


def _validate_range(value: int, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"cron value {value} outside range {minimum}-{maximum}")

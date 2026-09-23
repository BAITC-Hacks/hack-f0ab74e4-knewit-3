"""Проверка CSV-подачи без ML-зависимостей и доступа к сети.

Проверяет форму и склейку прогнозов, но не доказывает отсутствие утечки
фактической погоды в модели. Запуск: python3 -m scripts.verify_submission
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date, datetime, timedelta
from pathlib import Path


DAILY_FIELDS = {"turbine", "datetime", "horizon_h", "lead_day", "power_pred"}
SUBMISSION_FIELDS = {"turbine", "datetime", "lead_day", "power_pred"}


def _read_rows(path: Path, fields: set[str], errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        missing = fields - set(reader.fieldnames or [])
        if missing:
            errors.append(f"{path.name}: missing columns {sorted(missing)}")
            return []
        return list(reader)


def _number(row: dict[str, str], field: str, path: Path,
            errors: list[str]) -> float | None:
    try:
        value = float(row[field])
    except (TypeError, ValueError):
        errors.append(f"{path.name}: invalid {field}: {row.get(field)!r}")
        return None
    if not math.isfinite(value):
        errors.append(f"{path.name}: non-finite {field}")
        return None
    return value


def _hour(row: dict[str, str], path: Path, errors: list[str]) -> datetime | None:
    try:
        value = datetime.fromisoformat(row["datetime"])
    except (TypeError, ValueError):
        errors.append(f"{path.name}: invalid datetime: {row.get('datetime')!r}")
        return None
    if value.tzinfo is not None or value.minute or value.second or value.microsecond:
        errors.append(f"{path.name}: datetime must be a local whole hour: {value}")
        return None
    return value


def _days(first: date, last: date):
    day = first
    while day <= last:
        yield day
        day += timedelta(days=1)


def verify_forecasts(directory: Path, start: str = "2026-02-01",
                     end: str = "2026-02-28") -> list[str]:
    """Вернуть ошибки целостности прогнозов за указанный тестовый период."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must not be after end")
    errors: list[str] = []
    latest: dict[tuple[int, datetime], tuple[int, float]] = {}

    # The preceding issue covers day one; the final issue covers the final test day.
    for turbine in (1, 2):
        for issue in _days(first - timedelta(days=1), last - timedelta(days=1)):
            path = directory / f"forecast_t{turbine}_{issue.isoformat()}.csv"
            rows = _read_rows(path, DAILY_FIELDS, errors)
            expected = [datetime.combine(day, datetime.min.time()) + timedelta(hours=h)
                        for day in (issue + timedelta(days=1), issue + timedelta(days=2))
                        if first <= day <= last for h in range(24)]
            seen: set[datetime] = set()
            for row in rows:
                when = _hour(row, path, errors)
                value = _number(row, "power_pred", path, errors)
                lead = _number(row, "lead_day", path, errors)
                horizon = _number(row, "horizon_h", path, errors)
                if value is not None and not 0 <= value <= 1:
                    errors.append(f"{path.name}: power_pred outside [0, 1]")
                if when is None:
                    continue
                if when in seen:
                    errors.append(f"{path.name}: duplicate hour {when}")
                seen.add(when)
                if row["turbine"] != str(turbine):
                    errors.append(f"{path.name}: wrong turbine at {when}")
                if when not in expected:
                    errors.append(f"{path.name}: unexpected hour {when}")
                    continue
                expected_lead = (when.date() - issue).days
                expected_horizon = expected.index(when) + 1
                if lead != expected_lead or horizon != expected_horizon:
                    errors.append(f"{path.name}: wrong lead_day or horizon_h at {when}")
                if value is not None and lead == expected_lead:
                    key = (turbine, when)
                    if key not in latest or expected_lead < latest[key][0]:
                        latest[key] = (expected_lead, value)
            if seen != set(expected):
                errors.append(f"{path.name}: expected {len(expected)} hourly rows, "
                              f"found {len(seen & set(expected))}")

    submission = directory / "submission.csv"
    rows = _read_rows(submission, SUBMISSION_FIELDS, errors)
    seen_submission: set[tuple[int, datetime]] = set()
    for row in rows:
        when = _hour(row, submission, errors)
        value = _number(row, "power_pred", submission, errors)
        lead = _number(row, "lead_day", submission, errors)
        try:
            turbine = int(row["turbine"])
        except (TypeError, ValueError):
            errors.append(f"{submission.name}: invalid turbine {row.get('turbine')!r}")
            continue
        if value is not None and not 0 <= value <= 1:
            errors.append(f"{submission.name}: power_pred outside [0, 1]")
        if when is None:
            continue
        key = (turbine, when)
        if key in seen_submission:
            errors.append(f"{submission.name}: duplicate turbine/hour {key}")
        seen_submission.add(key)
        if key not in latest:
            errors.append(f"{submission.name}: unexpected or unbacked turbine/hour {key}")
            continue
        latest_lead, latest_value = latest[key]
        if lead != latest_lead or value is None or not math.isclose(
                value, latest_value, rel_tol=0, abs_tol=1e-9):
            errors.append(f"{submission.name}: not latest daily forecast at {key}")
    expected_submission = {(t, datetime.combine(day, datetime.min.time()) + timedelta(hours=h))
                           for t in (1, 2) for day in _days(first, last) for h in range(24)}
    if seen_submission != expected_submission:
        errors.append(f"{submission.name}: expected {len(expected_submission)} turbine-hours, "
                      f"found {len(seen_submission & expected_submission)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("forecasts"))
    parser.add_argument("--start", default="2026-02-01")
    parser.add_argument("--end", default="2026-02-28")
    args = parser.parse_args()
    errors = verify_forecasts(args.directory, args.start, args.end)
    if errors:
        for error in errors[:20]:
            print("FAIL:", error)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")
        return 1
    days = (date.fromisoformat(args.end) - date.fromisoformat(args.start)).days + 1
    print(f"OK: {days * 24 * 2} turbine-hours, daily forecasts and latest-run submission agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

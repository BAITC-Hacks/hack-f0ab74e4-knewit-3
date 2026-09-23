"""Проверки артефактов прогноза на маленьком контролируемом периоде."""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scripts.verify_submission import verify_forecasts


START = "2026-02-01"
END = "2026-02-02"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class VerifySubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.daily: dict[tuple[int, str], list[dict]] = {}
        submission = []
        first = datetime(2026, 2, 1)
        for turbine in (1, 2):
            for issue, offset, hours, value in (
                ("2026-01-31", 0, 48, 0.1),
                ("2026-02-01", 1, 24, 0.2),
            ):
                rows = []
                for hour in range(hours):
                    when = first + timedelta(days=offset, hours=hour)
                    rows.append({
                        "turbine": turbine, "datetime": when.isoformat(sep=" "),
                        "horizon_h": hour + 1,
                        "lead_day": 1 if hour < 24 else 2,
                        "power_pred": value,
                    })
                self.daily[turbine, issue] = rows
                write_csv(self.directory / f"forecast_t{turbine}_{issue}.csv", rows,
                          ["turbine", "datetime", "horizon_h", "lead_day", "power_pred"])
            for day, value in ((0, 0.1), (1, 0.2)):
                for hour in range(24):
                    submission.append({
                        "turbine": turbine,
                        "datetime": (first + timedelta(days=day, hours=hour)).isoformat(sep=" "),
                        "lead_day": 1, "power_pred": value,
                    })
        self.submission = submission
        self.save_submission()

    def save_submission(self) -> None:
        write_csv(self.directory / "submission.csv", self.submission,
                  ["turbine", "datetime", "lead_day", "power_pred"])

    def test_complete_submission_passes(self) -> None:
        self.assertEqual([], verify_forecasts(self.directory, START, END))

    def test_missing_hour_in_daily_forecast_fails(self) -> None:
        rows = self.daily[1, "2026-01-31"][:-1]
        write_csv(self.directory / "forecast_t1_2026-01-31.csv", rows,
                  ["turbine", "datetime", "horizon_h", "lead_day", "power_pred"])
        self.assertTrue(any("forecast_t1_2026-01-31.csv" in e
                            for e in verify_forecasts(self.directory, START, END)))

    def test_stale_submission_value_fails(self) -> None:
        self.submission[24]["power_pred"] = 0.1
        self.save_submission()
        self.assertTrue(any("latest" in e
                            for e in verify_forecasts(self.directory, START, END)))

    def test_out_of_bounds_prediction_fails(self) -> None:
        self.submission[0]["power_pred"] = 1.2
        self.save_submission()
        self.assertTrue(any("[0, 1]" in e
                            for e in verify_forecasts(self.directory, START, END)))


if __name__ == "__main__":
    unittest.main()

"""Сравнить два прогона: расходятся ли числа прогноза и чем отличаются отчёты.

Зачем: прогноз детерминирован при одинаковых артефактах моделей — LLM оркеструет
тулы, но не меняет значения. Поэтому два параллельных прогона либо дают идентичные
числа (различие только в тексте отчётов, выбор свободный), либо расходятся — и тогда
кто-то трогал модель или признаки, и выбирать надо замером на holdout, а не вкусом.

Запуск:
    python -m scripts.compare_runs --a forecasts --b ../their-clone/forecasts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEMPLATE_MARK = "Режим без LLM"


def _load_submission(directory: Path) -> pd.DataFrame | None:
    path = directory / "submission.csv"
    if not path.is_file():
        return None
    return (pd.read_csv(path, parse_dates=["datetime"])
              .set_index(["turbine", "datetime"])
              .sort_index())


def _report_stats(directory: Path) -> dict:
    reports = sorted(directory.glob("report_*.md"))
    llm = [p for p in reports if TEMPLATE_MARK not in p.read_text()]
    lengths = [len(p.read_text()) for p in llm]
    return {"всего": len(reports), "от LLM": len(llm),
            "шаблонных": len(reports) - len(llm),
            "средняя длина отчёта LLM": int(sum(lengths) / len(lengths)) if lengths else 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, default=Path("forecasts"), help="первый прогон")
    ap.add_argument("--b", type=Path, required=True, help="второй прогон")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="порог, выше которого расхождение считается значимым")
    args = ap.parse_args()

    sa, sb = _load_submission(args.a), _load_submission(args.b)
    if sa is None or sb is None:
        print(f"Нет submission.csv в {args.a if sa is None else args.b}")
        return 1

    print(f"A: {args.a}  строк {len(sa)}")
    print(f"B: {args.b}  строк {len(sb)}\n")

    only_a, only_b = sa.index.difference(sb.index), sb.index.difference(sa.index)
    if len(only_a) or len(only_b):
        print(f"Покрытие различается: только в A — {len(only_a)} часов, "
              f"только в B — {len(only_b)} часов")

    common = sa.index.intersection(sb.index)
    if len(common) == 0:
        print("Общих часов нет — сравнивать нечего.")
        return 1

    diff = (sa.loc[common, "power_pred"] - sb.loc[common, "power_pred"]).abs()
    n_diff = int((diff > args.tol).sum())
    print(f"Общих часов: {len(common)}")
    print(f"Расходятся сильнее {args.tol:g}: {n_diff} часов "
          f"({100 * n_diff / len(common):.1f}%)")
    print(f"Максимальное расхождение: {diff.max():.2e}")
    print(f"Среднее абсолютное расхождение: {diff.mean():.2e}\n")

    if n_diff == 0:
        print("ВЕРДИКТ: числа прогноза идентичны. Модель и признаки одни и те же,")
        print("различаются только тексты отчётов — можно брать любой набор чисел,")
        print("а отчёты выбрать по содержательности.\n")
    else:
        print("ВЕРДИКТ: числа расходятся — значит модель, признаки или погодный вход")
        print("различаются. Выбирать набор нужно замером на holdout, а не предпочтением:")
        print("  python -m src.cli train   # печатает MAE на holdout для каждой версии\n")

    print("Отчёты A:", _report_stats(args.a))
    print("Отчёты B:", _report_stats(args.b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Write one file of questions per period: data/<period>.json.

Each released persona carries its questions in
data/<period>/<persona>/evaluation_questions_<persona>.json, grouped by task.
This lists them all in one JSON array, persona by persona (sorted by name)
and, within a persona, in the order model_based_evaluator.py asks them
(Remembering, Reasoning, Recommending). Every question record is copied
unchanged; three fields are added so run_items.py can find its way back:
``persona``, ``period`` and ``task_type``.

    python evals/model_eval/flatten_questions.py            # all three periods
    python evals/model_eval/flatten_questions.py weekly
"""

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data"
TASK_TYPES = ("Remembering", "Reasoning", "Recommending")   # as model_based_evaluator.py
PERIODS = ("weekly", "monthly", "quarterly")


def flatten(period: str) -> list:
    out = []
    for persona_dir in sorted(p for p in (DATA / period).iterdir() if p.is_dir()):
        persona = persona_dir.name
        data = json.loads((persona_dir / f"evaluation_questions_{persona}.json").read_text(encoding="utf-8"))
        by_task = {k.lower(): v for k, v in data["questions"].items()}
        for task in TASK_TYPES:
            for q in by_task.get(task.lower(), []):
                out.append(dict(q, persona=persona, period=period, task_type=task))
    return out


def main():
    for period in sys.argv[1:] or PERIODS:
        items = flatten(period)
        path = DATA / f"{period}.json"
        path.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path.relative_to(DATA.parent)}: {len(items)} questions")


if __name__ == "__main__":
    main()

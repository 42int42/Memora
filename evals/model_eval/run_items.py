#!/usr/bin/env python3
"""
Track 1 over a list of questions rather than one persona's directory.

Reads a JSON array of questions as flatten_questions.py writes them (each
with ``persona`` and ``period``), and for each question, persona by persona
in the order they first appear, runs the unchanged EvaluationRunner of
model_based_evaluator.py on a directory holding that persona's conversations
and that one question. Same prompts, same judges, same per-question FAMA.

Before each question it prints one line naming it, flushed before any of its
model calls: ``=== item <persona>-<question_id> ===``. A question_id is
unique only within a persona, and the same question is asked of several, so
this line is what tells a reader of the output - and of the calls made after
it - which question they belong to.

Writes one JSON array of answers, in the order of the input, after each
persona: every question asked, with its response, its FAMA and the judges'
per-criterion results. A question whose model call failed (the evaluator
records the response as "ERROR: ...") carries ``error``, no response and a
``fama`` of null rather than the 0 the evaluator gives it, so a failed call
is not read as a wrong answer. Each question's own eval_results/eval_report files are
kept beside the answers file, under memora/<period>/<persona>/<question_id>/.

    python evals/model_eval/run_items.py --questions data/weekly.json \
        --output answers.json --model anthropic/claude-sonnet-4.5
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from api_client import OpenRouterClient                                   # noqa: E402
from model_based_evaluator import EvaluationRunner, TASK_TYPES             # noqa: E402

DATA = HERE.parent.parent / "data"
ADDED = ("persona", "period", "task_type")            # the fields flatten_questions.py adds


def groups(questions):
    """(period, persona) and its questions, in the order personas first appear."""
    out = {}
    for q in questions:
        out.setdefault((q["period"], q["persona"]), []).append(q)
    return out


def lay_out(where: Path, period: str, persona: str, asked) -> Path:
    """A sessions directory as model_based_evaluator.py reads one: the
    persona's conversations, and a questions file holding only ``asked``."""
    src = DATA / period / persona
    released = json.loads((src / f"evaluation_questions_{persona}.json").read_text(encoding="utf-8"))
    d = where / period / persona
    d.mkdir(parents=True)
    os.symlink(src / "conversations", d / "conversations")
    by_task = {t.lower(): [] for t in TASK_TYPES}
    for q in asked:
        by_task[q["task_type"].lower()].append({k: v for k, v in q.items() if k not in ADDED})
    (d / f"evaluation_questions_{persona}.json").write_text(
        json.dumps(dict(released, questions=by_task), ensure_ascii=False), encoding="utf-8")
    return d


def answer(q, got):
    a = {k: q[k] for k in ADDED + ("question_id", "question")}
    if got is None:
        return dict(a, error="not evaluated", fama=None)
    a.update({k: got.get(k) for k in (
        "model_response", "fama", "total_eval_questions",
        "memory_presence_accuracy", "memory_presence_correct", "memory_presence_total",
        "forgetting_absence_accuracy", "forgetting_absence_correct", "forgetting_absence_total",
        "per_judge_metrics", "evaluation_questions")})
    response = a.get("model_response") or ""
    if isinstance(response, str) and response.startswith("ERROR: "):
        a.update(error=response[len("ERROR: "):], model_response=None, fama=None)
    return a


def write(path: Path, answers):
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(answers, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", required=True, help="JSON array of questions (flatten_questions.py)")
    p.add_argument("--output", required=True, help="the answers file to write (a JSON array)")
    p.add_argument("--model", required=True, help="the model answering, as model_based_evaluator.py takes it")
    p.add_argument("--no-multi-judge", action="store_true")
    p.add_argument("--judge-openai", default="openai/gpt-4.1")
    p.add_argument("--judge-anthropic", default="anthropic/claude-haiku-4.5")
    p.add_argument("--judge-google", default="google/gemini-2.5-flash")
    p.add_argument("--reasoning-effort", choices=["high", "medium", "low", "minimal", "none"])
    p.add_argument("--reasoning-max-tokens", type=int)
    args = p.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    kept = output.parent / "memora"
    judges = {"openai": args.judge_openai, "anthropic": args.judge_anthropic, "google": args.judge_google}
    reasoning = ({"effort": args.reasoning_effort} if args.reasoning_effort
                 else {"max_tokens": args.reasoning_max_tokens} if args.reasoning_max_tokens else None)
    client = OpenRouterClient(model=args.model)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for (period, persona), asked in groups(questions).items():
            print(f"\n=== {period}/{persona}: {len(asked)} questions ===", flush=True)
            for q in asked:
                print(f"=== item {persona}-{q['question_id']} ===", flush=True)
                runner = EvaluationRunner(
                    sessions_dir=lay_out(Path(tmp) / str(len(results)), period, persona, [q]),
                    model_name=args.model, api_client=client,
                    output_dir=kept / period / persona / q["question_id"],
                    use_multi_judge=not args.no_multi_judge, judge_models=judges,
                    reasoning_config=reasoning)
                report = runner.run_evaluation()
                for got in report.get("detailed_results", []):
                    results[(period, persona, got["question_id"])] = got
                results.setdefault((period, persona, q["question_id"]), None)
                write(output, [answer(x, results.get((x["period"], x["persona"], x["question_id"])))
                               for x in questions if (x["period"], x["persona"], x["question_id"]) in results])
    write(output, [answer(q, results.get((q["period"], q["persona"], q["question_id"]))) for q in questions])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .asb_loader import ASBLoader
from .autogpt_app import AutoGPTSecurityApp
from .evaluator import SecurityEvaluator, record_to_dict, summarize, write_jsonl
from .scenarios import build_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AutoGPT-style agents against ASB-style security scenarios.")
    parser.add_argument("--asb-root", required=True, help="Path to local ASB repository root")
    parser.add_argument("--agent-name", default=None, help="Agent name from ASB data/agent_task.jsonl")
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Number of tasks to evaluate. Omit to run all tasks returned by the loader.",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "clean", "dpi", "opi", "mp", "mixed", "pot"],
        help="Scenario to run",
    )
    parser.add_argument("--output-dir", default="outputs_autogpt", help="Directory for output JSONL and summaries")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = ASBLoader(args.asb_root)
    app = AutoGPTSecurityApp(model=args.model)
    evaluator = SecurityEvaluator()

    scenarios = ["clean", "dpi", "opi", "mp", "mixed", "pot"] if args.scenario == "all" else [args.scenario]
    tasks = list(loader.iter_tasks(agent_name=args.agent_name, limit=args.task_limit))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        rows = []
        typed_records = []
        print(f"\n=== Running AutoGPT scenario: {scenario} ===")
        for idx, task in enumerate(tasks, start=1):
            case = build_case(scenario, task)
            print(f"[{scenario}] task {idx}/{len(tasks)}: {task}")
            response, audit = app.run(case)
            record = evaluator.evaluate(scenario=scenario, task=task, response=response, audit=audit)
            typed_records.append(record)
            rows.append(record_to_dict(record))

        result_path = output_dir / f"{scenario}_results.jsonl"
        summary_path = output_dir / f"{scenario}_summary.json"
        write_jsonl(result_path, rows)
        summary = summarize(typed_records)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
        print(f"Saved: {result_path}")
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

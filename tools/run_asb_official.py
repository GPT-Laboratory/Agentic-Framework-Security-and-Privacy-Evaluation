from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import textwrap

import yaml

SCENARIO_TO_METHOD = {
    "clean": "clean",
    "dpi": "direct_prompt_injection",
    "opi": "observation_prompt_injection",
    "mp": "memory_attack",
    "mixed": "mixed_attack",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an ASB YAML config and run the official ASB command.")
    p.add_argument("--asb-root", required=True, help="Path to local ASB repository")
    p.add_argument("--scenario", required=True, choices=["clean", "dpi", "opi", "mp", "mixed", "pot"])
    p.add_argument("--llm", action="append", dest="llms", help="Repeatable. Example: --llm ollama/llama3:8b")
    p.add_argument("--attack-tool", action="append", dest="attack_tool", default=None, help="Repeatable. all, agg, non-agg, test")
    p.add_argument("--attack-type", action="append", dest="attack_types", default=None, help="Repeatable. naive, context_ignoring, combined_attack, fake_completion, escape_characters")
    p.add_argument("--defense-type", default=None, help="Optional ASB defense_type")
    p.add_argument("--suffix", default="manual_run", help="Suffix appended by ASB to log/result filenames")
    p.add_argument("--read-db", action="store_true")
    p.add_argument("--write-db", action="store_true")
    p.add_argument("--trigger", action="append", dest="triggers", default=None, help="PoT trigger phrases")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> dict:
    if args.scenario == "pot":
        return {
            "injection_method": "pot_backdoor",
            "llms": args.llms or ["ollama/llama3:8b"],
            "triggers": args.triggers or ["calendar"],
            "suffix": args.suffix,
        }

    default_attack_types = {
        "clean": ["naive"],
        "dpi": ["naive"],
        "opi": ["context_ignoring"],
        "mp": ["combined_attack"],
        "mixed": ["naive"],
    }

    cfg = {
        "injection_method": SCENARIO_TO_METHOD[args.scenario],
        "attack_tool": args.attack_tool or ["all"],
        "llms": args.llms or ["ollama/llama3:8b"],
        "attack_types": args.attack_types or default_attack_types[args.scenario],
        "suffix": args.suffix,
    }
    if args.defense_type:
        cfg["defense_type"] = args.defense_type
    if args.read_db:
        cfg["read_db"] = True
    if args.write_db:
        cfg["write_db"] = True
    return cfg


def main() -> int:
    args = parse_args()
    asb_root = Path(args.asb_root).resolve()
    if not asb_root.exists():
        print(f"ASB root not found: {asb_root}", file=sys.stderr)
        return 1

    generated_dir = asb_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = generated_dir / f"{args.scenario}_{args.suffix}.yml"
    cfg = build_cfg(args)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    if args.scenario == "pot":
        cmd = [sys.executable, "scripts/agent_attack_pot.py"]
    else:
        cmd = [sys.executable, "scripts/agent_attack.py", "--cfg_path", str(cfg_path)]

    print("Generated config:")
    print(textwrap.indent(yaml.safe_dump(cfg, sort_keys=False), prefix="  "))
    print("Command:")
    print("  " + " ".join(cmd))
    print(f"Working directory:\n  {asb_root}")

    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(asb_root))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

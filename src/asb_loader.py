from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ASBLoader:
    def __init__(self, asb_root: str | Path):
        self.asb_root = Path(asb_root)
        self.data_dir = self.asb_root / "data"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"ASB data directory not found: {self.data_dir}")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def agent_tasks(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.data_dir / "agent_task.jsonl")

    def pot_tasks(self) -> list[dict[str, Any]]:
        path = self.data_dir / "agent_task_pot.jsonl"
        if path.exists():
            return self._read_jsonl(path)
        return []

    def find_agent_record(self, agent_name: str | None = None) -> dict[str, Any]:
        agents = self.agent_tasks()
        if not agent_name:
            return agents[0]
        for row in agents:
            if row.get("agent_name") == agent_name:
                return row
        available = ", ".join(sorted(r.get("agent_name", "") for r in agents))
        raise ValueError(f"Agent '{agent_name}' not found. Available: {available}")

    def iter_tasks(self, agent_name: str | None = None, limit: int | None = None) -> Iterable[str]:
        row = self.find_agent_record(agent_name)
        tasks = row.get("tasks", [])
        if limit is not None:
            tasks = tasks[:limit]
        for task in tasks:
            yield task

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class DTToxicitySample:
    source_file: str
    index: int
    prompt: str
    label: str


class DTToxicityLoader:
    def __init__(self, dt_root: str):
        self.root = Path(dt_root)

    def _resolve_file(self, split: str) -> Path:
        base = self.root / "data" / "toxicity" / "user_prompts"
        mapping = {
            "toxic": base / "toxic.jsonl",
            "nontoxic": base / "nontoxic.jsonl",
        }
        if split not in mapping:
            raise ValueError(f"Unsupported split: {split}")
        return mapping[split]

    def iter_samples(self, split: str = "toxic", limit: int | None = None) -> Iterator[DTToxicitySample]:
        path = self._resolve_file(split)
        if not path.exists():
            raise FileNotFoundError(f"Missing DecodingTrust toxicity file: {path}")

        count = 0
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                break

            obj = json.loads(line)
            prompt = str(
                obj.get("prompt")
                or obj.get("text")
                or obj.get("input")
                or obj.get("instruction")
                or ""
            ).strip()

            if not prompt:
                continue

            yield DTToxicitySample(
                source_file=str(path),
                index=idx,
                prompt=prompt,
                label=split,
            )
            count += 1

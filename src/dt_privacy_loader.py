from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class DTPrivacySample:
    source_file: str
    index: int
    prompt: str
    raw: dict


class DTPrivacyLoader:
    def __init__(self, dt_root: str):
        self.root = Path(dt_root)

    def _resolve_file(self, dataset: str) -> Path:
        base = self.root / "data" / "privacy" / "enron_data"
        mapping = {
            "context": base / "context.jsonl",
            "one_shot": base / "one_shot.jsonl",
            "two_shot": base / "two_shot.jsonl",
            "five_shot": base / "five_shot.jsonl",
        }
        if dataset not in mapping:
            raise ValueError(f"Unsupported privacy dataset: {dataset}")
        return mapping[dataset]

    def _extract_prompt(self, obj: dict) -> str:
        # common simple keys
        for key in ["prompt", "text", "input", "instruction", "question", "query"]:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # chat style messages
        messages = obj.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        parts.append(f"{role}: {content.strip()}" if role else content.strip())
            if parts:
                return "\n\n".join(parts)

        # nested prompt structures sometimes used in benchmark data
        if "prompt" in obj and isinstance(obj["prompt"], dict):
            nested = obj["prompt"]
            for key in ["text", "content", "input"]:
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # fallback: concatenate all string values
        strings: list[str] = []
        for _, value in obj.items():
            if isinstance(value, str) and value.strip():
                strings.append(value.strip())
        if strings:
            return "\n\n".join(strings)

        return ""

    def iter_samples(self, dataset: str = "one_shot", limit: int | None = None) -> Iterator[DTPrivacySample]:
        path = self._resolve_file(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Missing DecodingTrust privacy file: {path}")

        count = 0
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                break

            obj = json.loads(line)
            prompt = self._extract_prompt(obj)
            if not prompt:
                continue

            yield DTPrivacySample(
                source_file=str(path),
                index=idx,
                prompt=prompt,
                raw=obj,
            )
            count += 1

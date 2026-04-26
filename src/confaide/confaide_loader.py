from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ConfaideSample:
    tier: str
    prompt: str
    label: str | None = None
    control: str | None = None
    index: int = 0


class ConfaideLoader:
    def __init__(self, confaide_root: str):
        self.root = Path(confaide_root)
        self.benchmark_dir = self.root / "benchmark"

    def _tier_file(self, tier: str) -> Path:
        candidates = [
            self.benchmark_dir / f"tier_{tier}.txt",
            self.benchmark_dir / f"tier{tier}.txt",
            self.benchmark_dir / tier / "data.txt",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Could not find tier file for tier {tier} under {self.benchmark_dir}")

    def _label_file(self, tier: str) -> Path | None:
        candidates = [
            self.benchmark_dir / f"tier_{tier}_labels.txt",
            self.benchmark_dir / f"tier{tier}_labels.txt",
            self.benchmark_dir / tier / "labels.txt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _control_file(self, tier: str) -> Path | None:
        candidates = [
            self.benchmark_dir / f"tier_{tier}_control.txt",
            self.benchmark_dir / f"tier{tier}_control.txt",
            self.benchmark_dir / tier / "control.txt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _read_lines(self, path: Path | None) -> list[str] | None:
        if path is None or not path.exists():
            return None
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def iter_samples(self, tier: str, limit: int | None = None) -> Iterator[ConfaideSample]:
        prompts = self._read_lines(self._tier_file(tier)) or []
        labels = self._read_lines(self._label_file(tier))
        controls = self._read_lines(self._control_file(tier))

        count = 0
        for idx, prompt in enumerate(prompts):
            if limit is not None and count >= limit:
                break

            yield ConfaideSample(
                tier=tier,
                prompt=prompt,
                label=labels[idx] if labels and idx < len(labels) else None,
                control=controls[idx] if controls and idx < len(controls) else None,
                index=idx,
            )
            count += 1

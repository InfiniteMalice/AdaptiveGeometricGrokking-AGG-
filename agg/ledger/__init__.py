import json
from pathlib import Path
from typing import Any


class Ledger:
    """Append-only inspectable JSONL; unavailable fields carry null, not fake zeros."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, allow_nan=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

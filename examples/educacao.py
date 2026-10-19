"""Exemplo 3 — educação: gamificação no ensino de programação."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.runner_common import run_topic  # noqa: E402

TOPIC = "Impacto de gamificação no aprendizado de programação em escolas públicas brasileiras"

if __name__ == "__main__":
    raise SystemExit(run_topic(TOPIC))

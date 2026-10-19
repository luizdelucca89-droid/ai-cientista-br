"""Exemplo 2 — agronegócio: otimização de irrigação com IoT e ML no Cerrado."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.runner_common import run_topic  # noqa: E402

TOPIC = "Otimização de irrigação usando IoT e aprendizado de máquina no Cerrado brasileiro"

if __name__ == "__main__":
    raise SystemExit(run_topic(TOPIC))

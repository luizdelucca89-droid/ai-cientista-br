"""Exemplo 1 — ciclo científico completo via API Python.

Executa o pipeline inteiro (evidências → hipóteses → experimento → paper ABNT)
para o tópico de dengue, usando o provider configurado em ``config.json``.

Uso::

    python examples/quick_start.py
    python examples/quick_start.py "seu tópico aqui"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402

TOPIC = "Predição de surtos de dengue usando dados climáticos e aprendizado de máquina no Brasil"


def main() -> int:
    topic = sys.argv[1] if len(sys.argv) > 1 else TOPIC
    orchestrator = Orchestrator(config=Config.load())
    orchestrator.initialize_agents()
    try:
        results = orchestrator.run_research_cycle(topic=topic, force=True)
    finally:
        orchestrator.close()

    phases = results["phases"]
    print("\n" + "=" * 70)
    print("RESULTADO DO CICLO CIENTÍFICO")
    print("=" * 70)
    print(f"Status..........: {results['status']}")
    print(f"Ciclo...........: #{results['cycle_id']}")
    print(f"Duração.........: {results['duracao_s']}s")
    print(f"Artigos.........: {len(phases['evidencias'].get('artigos', []))}")
    print(f"Hipóteses.......: {len(phases['planejamento'].get('hipoteses', []))}")
    print(f"Melhor modelo...: {phases['experimento']['resultado'].get('best_model')}")
    dataset = phases["experimento"]["resultado"].get("dataset") or {}
    if dataset:
        print(f"Dados...........: {dataset.get('tipo')} (n={dataset.get('n_amostras')})")
        print(f"Origem..........: {dataset.get('origem')}")
    fidelidade = phases["validacao"].get("fidelidade_numerica") or {}
    if fidelidade:
        estado = "ok" if fidelidade.get("ok") else fidelidade.get("numeros_sem_suporte")
        print(f"Fidelidade num..: {estado}")
    print(f"Score validação.: {phases['validacao'].get('score')}")
    print(f"Paper (ABNT)....: {phases['artefatos'].get('final_paper')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

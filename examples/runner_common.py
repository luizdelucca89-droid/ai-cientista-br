"""Função compartilhada pelos exemplos executáveis."""

from __future__ import annotations

from core.config import Config
from core.orchestrator import Orchestrator
from core.utils import setup_logger


def run_topic(topic: str, force: bool = True) -> int:
    """Executa um ciclo científico completo para o tópico e imprime um resumo."""
    logger = setup_logger("ai_cientista")
    logger.info("Tópico: %s", topic)

    orchestrator = Orchestrator(config=Config.load())
    orchestrator.initialize_agents()
    try:
        results = orchestrator.run_research_cycle(topic=topic, force=force)
    finally:
        orchestrator.close()

    phases = results["phases"]
    metrics = phases["experimento"]["resultado"].get("metrics", {})
    best = phases["experimento"]["resultado"].get("best_model")

    print("\n" + "=" * 70)
    print(f"TÓPICO: {topic}")
    print("=" * 70)
    print(f"Status...........: {results['status']}")
    print(f"Ciclo............: #{results['cycle_id']}")
    print(f"Duração total....: {results['duracao_s']}s")
    print(f"Evidências.......: {len(phases['evidencias'].get('artigos', []))} artigos, "
          f"{len(phases['evidencias'].get('series_brasil', []))} indicadores brasileiros")
    print(f"Hipóteses........: {len(phases['planejamento'].get('hipoteses', []))}")
    print(f"Melhor modelo....: {best}")
    if best in metrics:
        print(f"F1 / AUC.........: {metrics[best].get('f1_score')} / {metrics[best].get('auc_roc')}")
    print(f"Score validação..: {phases['validacao'].get('score')}")
    print(f"Paper ABNT.......: {phases['artefatos'].get('final_paper')}")
    print(f"Artefatos........: {phases['artefatos'].get('diretorio')}")
    print(f"Métricas de LLM..: {phases.get('metricas_llm')}")
    return 0 if results["status"] == "concluido" else 1

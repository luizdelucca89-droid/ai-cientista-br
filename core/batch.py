"""Execução em lote de vários tópicos, portada do modo ``--parallel`` do
AI Scientist v1.

O v1 usa ``multiprocessing.Pool`` com um processo por GPU. Aqui não há GPU, mas
o padrão continua útil: cada tópico é independente e o gargalo é a espera de
rede (APIs abertas + LLM), então paralelizar por processo reduz o tempo total.

Cada processo recebe um diretório de saída próprio e um banco de memória
próprio, evitando que duas execuções escrevam no mesmo SQLite.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.utils import ensure_dir, setup_logger, slugify, utcnow_iso, write_json

logger = setup_logger(__name__)


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Executa um ciclo completo em um processo separado.

    Deve permanecer no nível do módulo (não um método) para ser serializável
    pelo ``ProcessPoolExecutor``.
    """
    topic = payload["topic"]
    base_dir = Path(payload["base_dir"])
    run_dir = ensure_dir(base_dir / slugify(topic))
    try:
        from core.config import Config
        from core.orchestrator import Orchestrator

        config = Config.load(payload.get("config_path"))
        config.data.setdefault("paths", {})
        config.data["paths"]["output_dir"] = str(run_dir)
        config.data["paths"]["memory_db"] = str(run_dir / "memory.sqlite3")
        config.data.setdefault("llm", {}).setdefault("cache", {})["path"] = str(run_dir / "llm_cache.sqlite3")

        orchestrator = Orchestrator(config=config, output_dir=run_dir)
        try:
            result = orchestrator.run_research_cycle(topic, force=bool(payload.get("force")))
        finally:
            orchestrator.close()

        summary = {
            "topico": topic,
            "status": result.get("status"),
            "ciclo_id": result.get("cycle_id"),
            "duracao_s": result.get("duracao_s"),
            "diretorio": str(run_dir),
            "artefato": (result.get("phases", {}).get("artefatos", {}) or {}).get("final_paper"),
            "score": (result.get("phases", {}).get("validacao", {}) or {}).get("score"),
            "decisao_painel": (result.get("phases", {}).get("validacao", {}) or {}).get("decisao_painel"),
        }
    except Exception as exc:  # noqa: BLE001 - um tópico não deve derrubar o lote
        logger.exception("Falha ao processar tópico %s", topic)
        summary = {"topico": topic, "status": "erro", "erro": f"{type(exc).__name__}: {exc}", "diretorio": str(run_dir)}
    write_json(run_dir / "resumo.json", summary)
    return summary


@dataclass
class BatchRunner:
    """Orquestra vários tópicos em paralelo e consolida um relatório único."""

    base_dir: Path
    max_workers: int = 2
    config_path: str | None = None
    force: bool = False

    def run(self, topics: list[str]) -> dict[str, Any]:
        ensure_dir(self.base_dir)
        payloads = [
            {"topic": topic, "base_dir": str(self.base_dir), "config_path": self.config_path, "force": self.force}
            for topic in topics
        ]
        results: list[dict[str, Any]] = []
        workers = max(1, min(self.max_workers, len(payloads) or 1))
        logger.info("Lote: %s tópicos, %s processos", len(payloads), workers)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, payload): payload["topic"] for payload in payloads}
            for future in as_completed(futures):
                topic = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    results.append({"topico": topic, "status": "erro", "erro": str(exc)})

        order = {topic: i for i, topic in enumerate(topics)}
        results.sort(key=lambda r: order.get(r["topico"], 999))
        report = {
            "gerado_em": utcnow_iso(),
            "total": len(results),
            "concluidos": sum(1 for r in results if r.get("status") == "concluido"),
            "erros": sum(1 for r in results if r.get("status") == "erro"),
            "resultados": results,
        }
        write_json(self.base_dir / "lote_resumo.json", report)
        return report

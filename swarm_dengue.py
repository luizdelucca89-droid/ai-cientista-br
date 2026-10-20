"""Ataque de enxame (swarm) em dados reais de dengue.

Executa três ciclos científicos em sequência sobre o mesmo fenômeno (surtos de
dengue no Brasil), variando a modelagem — baseline climático, features de
defasagem temporal e Gradient Boosting com validação cruzada temporal. A base é
sempre observada (InfoDengue/Fiocruz); não há dado sintético.

Uso::

    python swarm_dengue.py                 # provider do config efetivo
    AI_CIENTISTA_CONFIG=config.free-stack.json python swarm_dengue.py

A saída do terminal é ecoada e, ao final, gravada em ``resultado_swarm.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.utils import setup_logger  # noqa: E402

logger = setup_logger("swarm_dengue")

TOPICOS = [
    "Predição de surtos de dengue no Brasil usando apenas dados climáticos "
    "históricos e regressão logística como baseline.",
    "Predição de surtos de dengue no Brasil adicionando features de defasagem "
    "temporal (lag de 4 semanas) e Random Forest.",
    "Predição de surtos de dengue no Brasil usando Gradient Boosting e validação "
    "cruzada temporal estrita.",
]

ARQUIVO_SAIDA = "resultado_swarm.txt"


def _fonte_dados(dataset: dict) -> str:
    """Resume a procedência real da base usada no ciclo."""
    origem = (dataset or {}).get("origem") or "desconhecida"
    n = (dataset or {}).get("n_amostras")
    tipo = (dataset or {}).get("tipo") or "?"
    return f"{origem} (n={n}, tipo={tipo})"


def run_swarm() -> int:
    logger.info("Iniciando ataque de enxame em dengue (%s tópicos)", len(TOPICOS))
    orchestrator = Orchestrator(config=Config.load())
    orchestrator.initialize_agents()

    resultados: list[dict] = []
    try:
        for i, topico in enumerate(TOPICOS, start=1):
            print("\n" + "#" * 72)
            print(f"# ENXAME {i}/{len(TOPICOS)}: {topico}")
            print("#" * 72, flush=True)
            try:
                ciclo = orchestrator.run_research_cycle(topic=topico, force=True)
                resultados.append({"topico": topico, "ciclo": ciclo, "erro": None})
            except Exception as exc:  # noqa: BLE001 - o enxame continua os demais
                logger.exception("Enxame %s falhou: %s", i, exc)
                resultados.append({"topico": topico, "ciclo": None, "erro": str(exc)})
    finally:
        orchestrator.close()

    relatorio = _montar_painel(resultados)
    print("\n" + relatorio)
    Path(ARQUIVO_SAIDA).write_text(relatorio, encoding="utf-8")
    logger.info("Painel do enxame gravado em %s", ARQUIVO_SAIDA)

    return 0 if all(r["erro"] is None for r in resultados) else 1


def _montar_painel(resultados: list[dict]) -> str:
    """Compara os ciclos do enxame em um painel textual alinhado."""
    linhas: list[str] = []
    linhas.append("=" * 72)
    linhas.append("PAINEL DE RESULTADOS DO ENXAME — DENGUE (dados reais)")
    linhas.append("=" * 72)

    # Colunas textuais; a métrica principal é F1 do melhor modelo (com AUC).
    cabecalho = f"{'#':<2} {'CICLO':<6} {'MODELO':<20} {'F1':<7} {'AUC':<7} {'FIDELIDADE':<10} TOPICO"
    linhas.append(cabecalho)
    linhas.append("-" * 72)

    dados_reais_todos = True
    for i, r in enumerate(resultados, start=1):
        topico = r["topico"]
        if r["erro"] is not None:
            linhas.append(f"{i:<2} {'-':<6} {'-':<20} {'-':<7} {'-':<7} {'ERRO':<10} {topico[:40]}")
            continue

        ciclo = r["ciclo"]
        fases = ciclo["phases"]
        resultado = fases["experimento"]["resultado"]
        dataset = resultado.get("dataset") or {}
        metrics = resultado.get("metrics") or {}
        best = resultado.get("best_model") or "-"
        metrica = metrics.get(best) or {}
        f1 = metrica.get("f1_score")
        auc = metrica.get("auc_roc")
        fid = fases.get("fidelidade_pre_formatacao") or fases["validacao"].get("fidelidade_numerica") or {}
        fid_txt = "ok" if fid.get("ok") else f"FALHA{fid.get('numeros_sem_suporte')}"
        if dataset.get("tipo") != "dados_reais":
            dados_reais_todos = False

        linhas.append(
            f"{i:<2} {('#' + str(ciclo['cycle_id'])):<6} {best:<20} "
            f"{(f1 if f1 is not None else '-'):<7} {(auc if auc is not None else '-'):<7} "
            f"{fid_txt:<10} {topico[:40]}"
        )

    linhas.append("-" * 72)

    # Detalhamento por ciclo: fonte de dados, artefatos e score.
    linhas.append("")
    linhas.append("DETALHAMENTO POR CICLO")
    linhas.append("-" * 72)
    for i, r in enumerate(resultados, start=1):
        if r["erro"] is not None:
            linhas.append(f"[{i}] ERRO: {r['erro']}")
            continue
        ciclo = r["ciclo"]
        fases = ciclo["phases"]
        resultado = fases["experimento"]["resultado"]
        dataset = resultado.get("dataset") or {}
        linhas.append(f"[{i}] Ciclo #{ciclo['cycle_id']} — status: {ciclo['status']}")
        linhas.append(f"    Tópico.....: {r['topico']}")
        linhas.append(f"    Fonte......: {_fonte_dados(dataset)}")
        linhas.append(f"    Dataset URL: {dataset.get('fonte_url')}")
        linhas.append(f"    Alvo.......: {dataset.get('alvo')}")
        linhas.append(f"    Melhor mod.: {resultado.get('best_model')}")
        linhas.append(f"    Score val..: {fases['validacao'].get('score')}")
        linhas.append(f"    Fidelidade.: {(fases.get('fidelidade_pre_formatacao') or {}).get('ok')}")
        linhas.append(f"    Paper ABNT.: {fases['artefatos'].get('paper_pdf')}")
        linhas.append(f"    LLM........: {fases.get('metricas_llm', {}).get('por_provider')}")

    linhas.append("")
    linhas.append("=" * 72)
    veredito = "TODOS os ciclos usaram DADOS REAIS" if dados_reais_todos else "ATENCAO: houve ciclo sem dados reais"
    linhas.append(f"VEREDITO: {veredito}")
    linhas.append("=" * 72)
    return "\n".join(linhas)


if __name__ == "__main__":
    raise SystemExit(run_swarm())

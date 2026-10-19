#!/usr/bin/env python3
"""CLI do AI Cientista Brasileiro.

Exemplos
--------
Executar um ciclo completo de pesquisa::

    python main.py --topic "Predição de surtos de dengue usando dados climáticos no Brasil"

Forçar repetição de um tópico já estudado e escolher o provider::

    python main.py --topic "..." --force --provider mock

Apenas coletar evidências (sem LLM)::

    python main.py --topic "..." --collect-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.config import Config
from core.orchestrator import Orchestrator
from core.utils import setup_logger
from integrations import BrazilianDataAggregator

EXAMPLES = [
    "Machine Learning aplicado ao agronegócio brasileiro",
    "Predição de surtos de dengue usando dados climáticos no Brasil",
    "Impacto de gamificação no aprendizado de programação em escolas públicas",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-cientista",
        description="AI Cientista Brasileiro — pesquisa científica autônoma em português.",
    )
    parser.add_argument("--topic", "-t", help="Tópico de pesquisa.")
    parser.add_argument("--topics-file", help="Arquivo com um tópico por linha, para execução em lote paralela.")
    parser.add_argument("--parallel", type=int, default=2, help="Número de processos do modo em lote (padrão: 2).")
    parser.add_argument("--config", "-c", default=None, help="Caminho de um config.json alternativo.")
    parser.add_argument("--output-dir", "-o", default=None, help="Diretório de saída.")
    parser.add_argument("--provider", default=None, choices=["mock", "openai_compatible", "selenium"], help="Provider de LLM.")
    parser.add_argument("--model", default=None, help="Modelo do provider (ex.: openai/gpt-oss-120b, qwen/qwen3.8-27b, llama3.1).")
    parser.add_argument("--base-url", default=None, help="URL base compatível com a API da OpenAI (ex.: https://api.groq.com/openai/v1).")
    parser.add_argument("--force", action="store_true", help="Repetir mesmo que o tópico já exista na memória.")
    parser.add_argument("--collect-only", action="store_true", help="Apenas coletar evidências abertas.")
    parser.add_argument("--doctor", action="store_true", help="Diagnosticar o ambiente e sair (sem executar o pipeline).")
    parser.add_argument("--no-network", action="store_true", help="Com --doctor, pula a checagem de conectividade.")
    parser.add_argument("--examples", action="store_true", help="Listar tópicos de exemplo.")
    parser.add_argument("--memory-stats", action="store_true", help="Mostrar estatísticas da memória e sair.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log detalhado (DEBUG).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logger("ai_cientista", level="DEBUG" if args.verbose else "INFO")

    if args.examples:
        print("Tópicos de exemplo:")
        for i, topic in enumerate(EXAMPLES, start=1):
            print(f"  {i}. {topic}")
        return 0

    config = Config.load(args.config)
    if args.provider:
        config.data.setdefault("llm", {})["provider"] = args.provider
        config.data["llm"]["fallback_order"] = [args.provider, "mock"] if args.provider != "mock" else ["mock"]
    if args.model:
        config.data.setdefault("llm", {})["model"] = args.model
    if args.base_url:
        config.data.setdefault("llm", {})["base_url"] = args.base_url

    if args.doctor:
        from core.doctor import Doctor, render

        result = Doctor().run(output_dir=args.output_dir or config.output_dir, check_network=not args.no_network)
        print(render(result))
        return 0 if result["ok"] else 1

    if args.memory_stats:
        from core.memory import MemorySystem

        memory = MemorySystem(path=config.memory_path)
        print(json.dumps(memory.stats(), ensure_ascii=False, indent=2))
        print("Ciclos recentes:")
        for cycle in memory.list_cycles(limit=10):
            print(f"  #{cycle['id']} [{cycle['status']}] {cycle['topic']} (score={cycle['score']})")
        memory.close()
        return 0

    if args.topics_file:
        from core.batch import BatchRunner

        path = Path(args.topics_file)
        if not path.exists():
            print(f"Arquivo de tópicos não encontrado: {path}", file=sys.stderr)
            return 2
        topics = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not topics:
            print("Arquivo de tópicos vazio.", file=sys.stderr)
            return 2
        base_dir = Path(args.output_dir or config.output_dir) / "lote"
        report = BatchRunner(
            base_dir=base_dir, max_workers=args.parallel, config_path=args.config, force=args.force
        ).run(topics)
        print(f"\nLote: {report['concluidos']}/{report['total']} concluídos, {report['erros']} erros")
        for item in report["resultados"]:
            print(f"  [{item.get('status')}] {item['topico']} -> {item.get('diretorio')}")
        print(f"Resumo consolidado: {base_dir / 'lote_resumo.json'}")
        return 0 if report["erros"] == 0 else 1

    if not args.topic:
        build_parser().print_help()
        return 2

    if args.collect_only:
        aggregator = BrazilianDataAggregator.from_config(config.as_dict())
        gathered = aggregator.gather(args.topic)
        out = Path(args.output_dir or config.output_dir) / "evidencias.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gathered, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(aggregator.reference_context(gathered))
        print(f"\nArtefato salvo em: {out}")
        return 0

    orchestrator = Orchestrator(config=config, output_dir=args.output_dir)
    try:
        result = orchestrator.run_research_cycle(args.topic, force=args.force)
    finally:
        orchestrator.close()

    phases = result["phases"]
    print("\n" + "=" * 70)
    print(f"STATUS: {result['status']} | Ciclo #{result['cycle_id']} | {result['duracao_s']}s")
    print("=" * 70)
    print(f"Artigos coletados: {len(phases['evidencias'].get('artigos', []))}")
    print(f"Hipóteses geradas: {len(phases['planejamento'].get('hipoteses', []))}")
    print(f"Melhor modelo: {phases['experimento']['resultado'].get('best_model')}")
    metrics = phases["experimento"]["resultado"].get("metrics", {})
    best = phases["experimento"]["resultado"].get("best_model")
    if best in metrics:
        print(f"F1 (melhor modelo): {metrics[best].get('f1_score')}")
    print(f"Score de validação: {phases['validacao'].get('score')}")
    decisao = phases["validacao"].get("decisao_painel")
    if decisao:
        consenso = phases["validacao"].get("consenso_painel")
        print(f"Painel de revisão: {decisao} (consenso={consenso})")
    print(f"Paper: {phases['artefatos'].get('final_paper')}")
    print(f"Diretório do ciclo: {phases['artefatos'].get('diretorio')}")

    llm_stats = phases.get("metricas_llm") or {}
    by_provider = llm_stats.get("por_provider") or {}
    if by_provider:
        used = ", ".join(f"{name}={count}" for name, count in by_provider.items())
        print(f"Chamadas de LLM: {used}")
        configurados = [p for p in llm_stats.get("providers_configurados", []) if p != "mock"]
        if by_provider.get("mock") and configurados:
            falhas = llm_stats.get("falhas_por_provider") or {}
            detalhe = ", ".join(f"{p} ({falhas[p]} falhas)" for p in configurados if p in falhas)
            print(
                f"AVISO: {by_provider['mock']} de {llm_stats.get('chamadas')} chamadas caíram no provider "
                f"mock. Providers reais configurados: {', '.join(configurados)}"
                + (f" — {detalhe}" if detalhe else "")
                + ". O texto do artigo NÃO foi gerado pelo modelo escolhido."
            )
        elif by_provider.get("mock") and not configurados:
            print(
                "AVISO: todas as chamadas usaram o provider mock; nenhum LLM real foi configurado "
                "ou nenhuma chave estava disponível. O texto é estrutural, não científico."
            )
    return 0 if result["status"] == "concluido" else 1


if __name__ == "__main__":
    sys.exit(main())

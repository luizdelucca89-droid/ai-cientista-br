"""Orquestrador principal do ciclo de pesquisa científica.

Coordena os seis agentes especializados, a memória persistente, a execução real
de experimentos e a formatação final do artigo em ABNT.

Fluxo (``run_research_cycle``):

1. Coleta de evidências em fontes abertas (Crossref, SciELO, dados.gov.br, IBGE).
2. Identificação de lacuna e geração de hipóteses (Planejador).
3. Validação de novidade das hipóteses (Revisor).
4. Planejamento experimental (Experimentalista).
5. Execução real do experimento + análise estatística.
6. Redação iterativa do manuscrito (Redator) com crítica (Revisor).
7. Validação final de qualidade e conformidade (Validador).
8. Formatação ABNT, gravação de artefatos e persistência na memória.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.agents import (
    DataIntegratorAgent,
    ExperimentalistAgent,
    PlannerAgent,
    ReviewerAgent,
    ValidatorAgent,
    WriterAgent,
)
from core.config import Config
from core.experiments import ExperimentRunner, save_experiment
from core.memory import MemorySystem
from core.paper import ABNTFormatter, LaTeXFormatter
from core.review import ReviewerPanel
from core.review.numeros import collect_numbers, find_unsupported_numbers
from core.utils import ensure_dir, setup_logger, slugify, utcnow_iso, write_json
from integrations import BrazilianDataAggregator
from llm_providers import build_llm_client

logger = setup_logger(__name__)


@dataclass
class Orchestrator:
    """Coordena o ciclo científico completo."""

    config: Config = field(default_factory=Config.load)
    output_dir: Path | None = None
    memory: MemorySystem | None = None
    llm: Any = None
    agents: dict[str, Any] = field(default_factory=dict)
    aggregator: BrazilianDataAggregator | None = None
    formatter: ABNTFormatter | None = None
    runner: ExperimentRunner | None = None
    panel: ReviewerPanel | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir) if self.output_dir else self.config.output_dir
        ensure_dir(self.output_dir)

    # --------------------------------------------------------- initialization
    def initialize_agents(self) -> dict[str, Any]:
        """Instancia o cliente LLM, a memória e os seis agentes."""
        if self.llm is None:
            self.llm = build_llm_client(self.config.as_dict())
        if self.memory is None:
            self.memory = MemorySystem(path=self.config.memory_path)
        if self.aggregator is None:
            self.aggregator = BrazilianDataAggregator.from_config(self.config.as_dict())
        if self.formatter is None:
            self.formatter = ABNTFormatter(
                max_references=int(self.config.get("paper", "max_references", default=12))
            )
        if self.runner is None:
            self.runner = ExperimentRunner(
                seed=self.config.seed,
                test_size=float(self.config.get("experiments", "test_size", default=0.3)),
                alpha=float(self.config.get("experiments", "alpha", default=0.05)),
                output_dir=self.output_dir,
                allow_code_execution=bool(self.config.get("experiments", "allow_code_execution", default=True)),
                cache_dir=str(self.output_dir / "dataset_cache"),
                offline=bool(self.config.get("integrations", "offline", default=False)),
            )

        agents_cfg = self.config.get("agents", default={}) or {}
        self.agents = {
            "planejador": PlannerAgent(
                llm=self.llm,
                temperature=agents_cfg.get("planner", {}).get("temperature", 0.4),
                max_hypotheses=agents_cfg.get("planner", {}).get("max_hypotheses", 3),
            ),
            "experimentalista": ExperimentalistAgent(
                llm=self.llm,
                temperature=agents_cfg.get("experimentalist", {}).get("temperature", 0.1),
                max_retries=agents_cfg.get("experimentalist", {}).get("max_retries", 2),
            ),
            "revisor": ReviewerAgent(
                llm=self.llm,
                temperature=agents_cfg.get("reviewer", {}).get("temperature", 0.3),
                strictness=agents_cfg.get("reviewer", {}).get("strictness", 0.7),
            ),
            "redator": WriterAgent(
                llm=self.llm,
                temperature=agents_cfg.get("writer", {}).get("temperature", 0.5),
                iterations=agents_cfg.get("writer", {}).get("iterations", 2),
            ),
            "validador": ValidatorAgent(
                llm=self.llm,
                temperature=agents_cfg.get("validator", {}).get("temperature", 0.0),
                min_score=agents_cfg.get("validator", {}).get("min_score", 0.7),
            ),
            "integrador": DataIntegratorAgent(
                llm=self.llm,
                temperature=0.2,
                max_sources=agents_cfg.get("data_integrator", {}).get("max_sources", 12),
            ),
        }
        logger.info("Agentes inicializados: %s", ", ".join(self.agents))
        if self.panel is None:
            review_cfg = agents_cfg.get("reviewer", {}) or {}
            self.panel = ReviewerPanel(
                llm=self.llm,
                reviewer_count=int(review_cfg.get("reviewer_count", 5)),
                strictness=float(review_cfg.get("strictness", 0.7)),
            )
        return self.agents

    # ----------------------------------------------------------------- phases
    def phase_collect_evidence(self, topic: str) -> dict[str, Any]:
        assert self.aggregator is not None
        logger.info("[1/8] Coletando evidências em fontes abertas...")
        gathered = self.aggregator.gather(topic)
        gathered["referencias_formatadas"] = (
            self.formatter.format_reference_list(gathered.get("artigos", [])) if self.formatter else []
        )
        logger.info(
            "Evidências: %s artigos (Crossref=%s, SciELO=%s), %s datasets",
            len(gathered.get("artigos", [])),
            gathered.get("artigos_crossref"),
            gathered.get("artigos_scielo"),
            len(gathered.get("datasets", [])),
        )
        return gathered

    def phase_plan(self, topic: str, gathered: dict[str, Any]) -> dict[str, Any]:
        evidence = self.aggregator.reference_context(gathered) if self.aggregator else ""
        memory_context = ""
        if self.memory:
            previous = self.memory.recall(topic, limit=5)
            memory_context = "\n".join(f"- ({p['kind']}) {p['content'][:200]}" for p in previous)

        logger.info("[2/8] Identificando lacuna de pesquisa...")
        gap_result = self.agents["planejador"].identify_gap(topic, evidence, memory_context)
        gap = gap_result.raw.strip()

        logger.info("[3/8] Gerando hipóteses testáveis...")
        hypotheses_result = self.agents["planejador"].generate_hypotheses(topic, gap, evidence)
        hypotheses = self._extract_list(hypotheses_result.content, "hipoteses")

        logger.info("[4/8] Validando novidade das hipóteses...")
        novelty = []
        for hypothesis in hypotheses:
            result = self.agents["revisor"].validate_novelty(topic, hypothesis, evidence)
            novelty.append(result.content)
        return {"lacuna": gap, "hipoteses": hypotheses, "novidade": novelty}

    def phase_experiment(self, topic: str, plan: dict[str, Any], gathered: dict[str, Any]) -> dict[str, Any]:
        hypotheses = plan.get("hipoteses") or []
        if not hypotheses:
            logger.warning("Nenhuma hipótese extraída; usando hipótese padrão.")
            hypotheses = [{"id": "H1", "enunciado": f"Avaliar modelos preditivos para {topic}."}]
        hypothesis = hypotheses[0]
        dataset_summary = ", ".join(d.get("titulo", "") for d in gathered.get("datasets", [])[:3])

        logger.info("[5/8] Planejando experimento...")
        design = self.agents["experimentalista"].design_experiment(topic, hypothesis, dataset_summary)

        logger.info("[6/8] Executando experimento e análise estatística...")
        assert self.runner is not None
        result = self.runner.run(topic, hypothesis, design.content)
        experiment_path = save_experiment(result, self.output_dir)

        logger.info("Analisando resultados com o agente experimentalista...")
        interpretation = self.agents["experimentalista"].interpret_results(topic, hypothesis, result.as_dict())
        return {
            "hipotese_principal": hypothesis,
            "desenho": design.content,
            "resultado": result.as_dict(),
            "artefato": experiment_path,
            "interpretacao": interpretation.raw.strip(),
        }

    def phase_write_and_review(
        self, topic: str, plan: dict[str, Any], experiment: dict[str, Any], gathered: dict[str, Any]
    ) -> dict[str, Any]:
        evidence = self.aggregator.reference_context(gathered) if self.aggregator else ""
        # Resultados primeiro: a serialização do contexto pode ser truncada, e os
        # números reais do experimento são a informação mais crítica para o redator.
        context = {
            "resultado_experimental": experiment.get("resultado"),
            "interpretacao": experiment.get("interpretacao"),
            "hipotese_principal": experiment.get("hipotese_principal"),
            "lacuna": plan.get("lacuna"),
            "hipoteses": plan.get("hipoteses"),
            "evidencias": evidence,
            "referencias": gathered.get("referencias_formatadas"),
        }

        logger.info("[7/8] Redigindo manuscrito (versão 1)...")
        writer = self.agents["redator"]
        sections = writer.write_full_paper(topic, context)
        sections["abstract"] = self._make_abstract_en(writer, topic, sections.get("resumo", ""))

        iterations = max(0, int(getattr(writer, "iterations", 2)) - 1)
        reviews = []
        for i in range(iterations):
            manuscript = self._manuscript_text(topic, sections)
            review = self.agents["revisor"].review(topic, manuscript, evidence)
            reviews.append(review.content)
            critique = self._critique_text(review.content)
            logger.info("Refinando manuscrito (iteração %s)...", i + 2)
            for key in ("resumo", "introducao", "metodologia", "resultados", "conclusao"):
                if key in sections:
                    revised = writer.revise(key, sections[key], critique, topic, context)
                    sections[key] = revised.raw.strip()
        return {"secoes": sections, "pareceres": reviews}

    def phase_validate(self, topic: str, sections: dict[str, str], plan: dict[str, Any], experiment: dict[str, Any], gathered: dict[str, Any] | None = None) -> dict[str, Any]:
        logger.info("[8/8] Validando qualidade e conformidade...")
        manuscript = self._manuscript_text(topic, sections)
        checklist = {
            "tem_lacuna": bool(plan.get("lacuna")),
            "n_hipoteses": len(plan.get("hipoteses") or []),
            "experimento_executado": bool(experiment.get("resultado")),
            "melhor_modelo": experiment.get("resultado", {}).get("best_model"),
            "seed": experiment.get("resultado", {}).get("seed"),
        }
        validation = self.agents["validador"].validate(topic, manuscript, checklist)
        compliance = self.agents["validador"].check_compliance(manuscript)
        score = self._extract_score(validation.content)

        fidelidade = self._check_numeric_fidelity(manuscript, plan, experiment, gathered)
        if fidelidade["numeros_sem_suporte"]:
            logger.warning(
                "Fidelidade numérica: %s valor(es) sem lastro nos dados: %s",
                len(fidelidade["numeros_sem_suporte"]),
                fidelidade["numeros_sem_suporte"],
            )

        panel_result: dict[str, Any] = {}
        if self.panel is not None:
            evidence = self.aggregator.reference_context(gathered) if (self.aggregator and gathered) else ""
            panel_result = self.panel.review(topic, manuscript, evidence)

        return {
            "validacao": validation.content,
            "conformidade": compliance.content,
            "score": score,
            "fidelidade_numerica": fidelidade,
            "painel_revisao": panel_result,
            "decisao_painel": panel_result.get("decisao_final"),
            "consenso_painel": panel_result.get("consenso"),
        }

    def _check_numeric_fidelity(
        self,
        manuscript: str,
        plan: dict[str, Any],
        experiment: dict[str, Any],
        gathered: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Confere se cada número do manuscrito tem lastro nos dados reais.

        O validador por LLM lê e opina, mas não compara dígito a dígito — deixou
        passar um ``200201`` onde o dado observado dizia ``202201``. Esta
        conferência é determinística e roda sobre as seções do manuscrito (sem
        referências, que trazem anos e códigos de norma alheios ao experimento).
        """
        fontes_estrutura = {
            "resultado": experiment.get("resultado"),
            "hipotese": experiment.get("hipotese_principal"),
            "plano": plan,
            "evidencias": gathered,
            # Valores de método legítimos, definidos na configuração.
            "alpha": self.config.get("experiments", "alpha", default=0.05),
            "test_size": self.config.get("experiments", "test_size", default=0.3),
            "seed": self.config.get("seed", default=None),
        }
        fontes = collect_numbers(fontes_estrutura)
        suspeitos = find_unsupported_numbers(manuscript, fontes)
        return {
            "numeros_verificados_de": len(fontes),
            "numeros_sem_suporte": suspeitos,
            "ok": not suspeitos,
        }

    # ------------------------------------------------------------------- main
    def run_research_cycle(self, topic: str, force: bool = False) -> dict[str, Any]:
        """Executa o ciclo científico completo e devolve todos os artefatos."""
        if not self.agents:
            self.initialize_agents()
        assert self.memory is not None and self.formatter is not None

        started = time.time()
        if self.memory.is_duplicate_topic(topic) and not force:
            logger.warning("Tópico já pesquisado anteriormente (use force=True para repetir).")

        cycle_id = self.memory.start_cycle(topic)
        run_dir = ensure_dir(self.output_dir / f"ciclo_{cycle_id}_{slugify(topic)}")
        phases: dict[str, Any] = {}
        status = "concluido"

        try:
            gathered = self.phase_collect_evidence(topic)
            phases["evidencias"] = gathered
            self.memory.remember(topic, "evidencia", f"{len(gathered.get('artigos', []))} artigos coletados")

            plan = self.phase_plan(topic, gathered)
            phases["planejamento"] = plan
            for hypothesis in plan.get("hipoteses") or []:
                self.memory.remember(topic, "hipotese", str(hypothesis.get("enunciado", ""))[:400])

            experiment = self.phase_experiment(topic, plan, gathered)
            phases["experimento"] = experiment
            self.memory.remember(
                topic, "resultado",
                f"Melhor modelo: {experiment['resultado'].get('best_model')} "
                f"(F1={experiment['resultado'].get('metrics', {}).get(experiment['resultado'].get('best_model'), {}).get('f1_score')})",
            )

            writing = self.phase_write_and_review(topic, plan, experiment, gathered)
            phases["redacao"] = writing

            validation = self.phase_validate(topic, writing["secoes"], plan, experiment, gathered)
            phases["validacao"] = validation

            markdown = self.formatter.render_markdown(
                topic=topic,
                sections=writing["secoes"],
                references=gathered.get("referencias_formatadas", []),
                metadata={
                    "titulo": topic,
                    "autores": ["AI Cientista Brasileiro (sistema autônomo multi-agente)"],
                    "data": utcnow_iso()[:10],
                    "palavras_chave": self._keywords(topic),
                    "keywords": self._keywords(topic, english=True),
                },
                figures=experiment["resultado"].get("figures", []),
            )
            paper_path = run_dir / "paper.md"
            paper_path.write_text(markdown, encoding="utf-8")
            pdf_path = self.formatter.render_pdf(markdown, run_dir / "paper.pdf")

            latex_artifacts: dict[str, Any] = {}
            if self.config.get("paper", "latex", default=True):
                latex = LaTeXFormatter()
                latex_artifacts = latex.write(
                    topic=topic,
                    sections=writing["secoes"],
                    references=gathered.get("artigos", []),
                    metadata={
                        "titulo": topic,
                        "autores": ["AI Cientista Brasileiro (sistema autônomo multi-agente)"],
                        "data": utcnow_iso()[:10],
                        "palavras_chave": self._keywords(topic),
                        "keywords": self._keywords(topic, english=True),
                    },
                    output_dir=run_dir,
                    figures=experiment["resultado"].get("figures", []),
                )

            phases["artefatos"] = {
                "diretorio": str(run_dir),
                "paper_markdown": str(paper_path),
                "paper_pdf": pdf_path,
                "paper_tex": latex_artifacts.get("tex"),
                "paper_latex_pdf": latex_artifacts.get("pdf"),
                "experimento_json": experiment.get("artefato"),
                "figuras": experiment["resultado"].get("figures", []),
                "final_paper": str(pdf_path or paper_path),
            }
            phases["metricas_llm"] = self.llm.stats() if self.llm else {}
            phases["duracao_total_s"] = round(time.time() - started, 2)

            write_json(run_dir / "ciclo_completo.json", phases)
            self.memory.finish_cycle(cycle_id, status, phases, score=validation.get("score"))
        except Exception as exc:  # noqa: BLE001 - o ciclo deve registrar falhas
            status = "falhou"
            logger.exception("Ciclo falhou: %s", exc)
            phases["erro"] = str(exc)
            self.memory.finish_cycle(cycle_id, status, phases)
            raise
        finally:
            # O cache do LLM permanece aberto para ser reutilizado em outros ciclos;
            # feche explicitamente com ``orchestrator.close()``.
            pass

        return {
            "status": status,
            "cycle_id": cycle_id,
            "topic": topic,
            "phases": phases,
            "duracao_s": phases.get("duracao_total_s"),
        }

    # -------------------------------------------------------------- utilities
    @staticmethod
    def _extract_list(content: Any, key: str) -> list[dict[str, Any]]:
        if isinstance(content, dict):
            value = content.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
            if isinstance(value, dict):
                return [value]
        if isinstance(content, list):
            return [v for v in content if isinstance(v, dict)]
        return []

    @staticmethod
    def _extract_score(content: Any) -> float:
        if isinstance(content, dict):
            for key in ("score", "nota", "pontuacao"):
                if isinstance(content.get(key), (int, float)):
                    return float(content[key])
        return 0.0

    @staticmethod
    def _critique_text(content: Any) -> str:
        if isinstance(content, dict):
            parts = []
            for key in ("pontos_fracos", "recomendacoes", "erros_metodologicos", "vieses_detectados"):
                value = content.get(key)
                if isinstance(value, list):
                    parts.extend(str(v) for v in value)
            if not parts:
                return str(content)
            return "\n".join(f"- {p}" for p in parts)
        return str(content)

    @staticmethod
    def _manuscript_text(topic: str, sections: dict[str, str]) -> str:
        parts = [f"TÍTULO: {topic}"]
        for key in ("resumo", "introducao", "metodologia", "resultados", "conclusao"):
            if sections.get(key):
                parts.append(f"\n## {key.upper()}\n{sections[key]}")
        return "\n".join(parts)

    def _make_abstract_en(self, writer: Any, topic: str, resumo_pt: str) -> str:
        prompt = (
            f"Translate the following scientific abstract to academic English, keeping "
            f"the meaning exact. Topic: {topic}\n\nABSTRACT (pt-BR):\n{resumo_pt}\n\n"
            "Return only the translated abstract."
        )
        try:
            return writer.run(prompt, task="translate").raw.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao gerar abstract em inglês: %s", exc)
            return resumo_pt

    @staticmethod
    def _keywords(topic: str, english: bool = False) -> list[str]:
        stopwords = {
            "usando", "para", "com", "dos", "das", "uma", "sobre", "entre", "pelo", "pela",
            "como", "the", "and", "using", "with", "from", "into", "that", "this", "seu", "sua",
            "aplicado", "aplicada", "baseado", "baseada", "brasileiro", "brasileira", "brasil",
        }
        words = [re.sub(r"[^\w\-]", "", w) for w in topic.split()]
        base = [w for w in words if len(w) > 4 and w.lower() not in stopwords][:5]
        if not base:
            base = ["pesquisa", "ciência de dados", "brasil"]
        if not english:
            return base
        translations = {
            "predição": "prediction", "surtos": "outbreaks", "dengue": "dengue",
            "dados": "data", "climáticos": "climate", "aprendizado": "learning",
            "máquina": "machine", "irrigação": "irrigation", "otimização": "optimization",
            "gamificação": "gamification", "escolas": "schools", "públicas": "public",
            "programação": "programming", "agronegócio": "agribusiness",
            "pesquisa": "research", "ciência": "science", "saúde": "health",
        }
        return [translations.get(w.lower(), w) for w in base]

    def close(self) -> None:
        if self.memory:
            self.memory.close()
        if self.llm and hasattr(self.llm, "close"):
            self.llm.close()

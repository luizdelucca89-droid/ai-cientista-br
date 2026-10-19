"""Painel de revisão por pares: N revisores independentes + meta-revisor.

Técnica portada do AI Scientist (v1 ``perform_review.py`` e v3 ``aireview.py``):

* vários revisores avaliam o mesmo manuscrito em paralelo, com temperatura
  alta para diversificar os pareceres;
* um meta-revisor consolida os pareceres em um único parecer final;
* o painel é executado com ``ThreadPoolExecutor`` (o gargalo é I/O de rede).

Os revisores usam **temperaturas diferentes** para reduzir correlação entre
pareceres, e o meta-revisor recebe apenas os pareceres (não o manuscrito),
o que força agregação em vez de uma nova leitura enviesada.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from core.agents.reviewer import ReviewerAgent
from core.utils import setup_logger

logger = setup_logger(__name__)

# Recomendações aceitas, da mais positiva à mais negativa. Usado para média
# ordinal do painel (um "aceitar" e um "rejeitar" não podem virar "aceitar com
# revisões" por média aritmética ingênua de rótulos).
DECISIONS = ["aceitar", "aceitar com revisões", "revisões maiores", "rejeitar"]

DEFAULT_TEMPERATURES = [0.3, 0.6, 0.9, 1.2, 0.75]


@dataclass
class ReviewerPanel:
    """Executa um painel de revisores e consolida um parecer final."""

    llm: Any
    reviewer_count: int = 5
    strictness: float = 0.7
    temperatures: list[float] = field(default_factory=lambda: list(DEFAULT_TEMPERATURES))
    max_workers: int = 5

    def _reviewer(self, index: int) -> ReviewerAgent:
        temperature = self.temperatures[index % len(self.temperatures)]
        return ReviewerAgent(llm=self.llm, temperature=temperature, strictness=self.strictness)

    def review(self, topic: str, manuscript: str, evidence_summary: str = "") -> dict[str, Any]:
        """Roda o painel completo e devolve pareceres + meta-parecer."""
        count = max(1, int(self.reviewer_count))
        logger.info("Painel de revisão: %s revisores independentes", count)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, count)) as pool:
            futures = [
                pool.submit(self._reviewer(i).review, topic, manuscript, evidence_summary)
                for i in range(count)
            ]
            results = []
            for i, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - um revisor não derruba o painel
                    logger.warning("Revisor %s falhou: %s", i + 1, exc)

        reviews = [r.content for r in results if isinstance(r.content, dict)]
        if not reviews:
            logger.warning("Nenhum revisor devolveu parecer utilizável")
            return {
                "revisores": [],
                "meta_parecer": {},
                "decisao_final": None,
                "consenso": None,
                "n_validos": 0,
            }

        meta = self._meta_review(topic, reviews)
        decision = self._majority_decision(reviews)
        return {
            "revisores": reviews,
            "meta_parecer": meta,
            "decisao_final": decision["decisao"],
            "consenso": decision["consenso"],
            "distribuicao_decisoes": decision["distribuicao"],
            "n_validos": len(reviews),
        }

    # --------------------------------------------------------------- internals
    def _meta_review(self, topic: str, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        """Consolida os pareceres individuais em um único meta-parecer."""
        import json

        payload = [
            {
                "parecer": r.get("parecer"),
                "severidade": r.get("severidade"),
                "pontos_fortes": r.get("pontos_fortes"),
                "pontos_fracos": r.get("pontos_fracos"),
                "erros_metodologicos": r.get("erros_metodologicos"),
                "vieses_detectados": r.get("vieses_detectados"),
            }
            for r in reviews
        ]
        prompt = f"""TÓPICO: {topic}

Você é o META-REVISOR (chair) do comitê. Abaixo estão {len(reviews)} pareceres
independentes sobre o mesmo manuscrito.

PARECERES:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)[:12000]}

Consolide-os em um único parecer. Regras:
- Não introduza problemas que nenhum revisor apontou.
- Priorize os problemas apontados por mais de um revisor (consenso).
- Preserve divergências relevantes em "divergencias".
- Responda SOMENTE com JSON:
{{"parecer": "aceitar|aceitar com revisões|revisões maiores|rejeitar",
"pontos_fortes": ["..."], "pontos_fracos": ["..."],
"recomendacoes": ["..."], "severidade": "baixa|moderada|alta",
"divergencias": ["..."], "confianca": 0.0}}"""
        agent = ReviewerAgent(llm=self.llm, temperature=0.2, strictness=self.strictness)
        result = agent.run(prompt, expect_json=True, task="review")
        return result.content if isinstance(result.content, dict) else {}

    @staticmethod
    def _majority_decision(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        """Voto majoritário ordinal sobre as recomendações dos revisores."""
        counts: dict[str, int] = {}
        indices: list[int] = []
        for review in reviews:
            label = str(review.get("parecer", "")).strip().lower()
            if label in DECISIONS:
                counts[label] = counts.get(label, 0) + 1
                indices.append(DECISIONS.index(label))
        if not indices:
            return {"decisao": None, "consenso": None, "distribuicao": {}}

        # Desempate pelo parecer mais brando em caso de empate exato, prática
        # comum em comitês quando não há maioria clara.
        top = max(counts.values())
        candidatos = [d for d in DECISIONS if counts.get(d) == top]
        decision = candidatos[0]
        return {
            "decisao": decision,
            "consenso": round(1.0 - (statistics.pstdev(indices) / (len(DECISIONS) - 1)), 3),
            "distribuicao": counts,
            "media_ordinal": round(statistics.fmean(indices), 3),
        }

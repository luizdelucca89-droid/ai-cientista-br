"""Agente Revisor Crítico: revisão por pares adversária."""

from __future__ import annotations

from typing import Any

from core.agents.base import BaseAgent
from core.prompts import review_fewshot

SYSTEM = """Você é o Agente Revisor Crítico, atuando como parecerista adversário de
um periódico científico de alto impacto. Seu papel é encontrar falhas reais.

Regras:
- Seja rigoroso e específico; aponte problemas verificáveis no material recebido.
- Classifique severidade como "baixa", "moderada" ou "alta".
- Emita um parecer final: "aceitar", "aceitar com revisões", "revisões maiores" ou "rejeitar".
- Responda SEMPRE em português do Brasil.
- Quando solicitado JSON, devolva apenas JSON válido."""


class ReviewerAgent(BaseAgent):
    """Produz parecer crítico e recomendações de melhoria."""

    def __init__(self, llm: Any, temperature: float = 0.3, strictness: float = 0.7, **kwargs: Any) -> None:
        super().__init__(
            name="revisor_critico",
            role="Revisão por pares adversária",
            system_prompt=SYSTEM,
            temperature=temperature,
            llm=llm,
            **kwargs,
        )
        self.strictness = strictness

    def review(self, topic: str, manuscript: str, evidence_summary: str = "") -> Any:
        prompt = f"""{review_fewshot()}

TÓPICO: {topic}

EVIDÊNCIAS DISPONÍVEIS:
{evidence_summary or 'Não fornecidas.'}

MANUSCRITO A REVISAR:
\"\"\"
{manuscript}
\"\"\"

Rigor solicitado: {self.strictness:.2f} (0 = brando, 1 = extremamente rigoroso).

Responda SOMENTE com JSON:
{{"parecer": "aceitar|aceitar com revisões|revisões maiores|rejeitar",
"pontos_fortes": ["..."], "pontos_fracos": ["..."],
"recomendacoes": ["..."], "severidade": "baixa|moderada|alta",
"erros_metodologicos": ["..."], "vieses_detectados": ["..."]}}"""
        return self.run(prompt, expect_json=True, task="review")

    def validate_novelty(self, topic: str, hypothesis: dict[str, Any], evidence: str) -> Any:
        import json

        prompt = f"""TÓPICO: {topic}

HIPÓTESE PROPOSTA:
{json.dumps(hypothesis, ensure_ascii=False)}

EVIDÊNCIAS DISPONÍVEIS:
{evidence}

A hipótese é original frente às evidências? Ela é trivialmente respondida pela
literatura listada? Responda SOMENTE com JSON:
{{"original": true, "confianca": 0.0, "justificativa": "...",
"ja_respondida_na_literatura": false, "sugestao": "..."}}"""
        return self.run(prompt, expect_json=True, task="novelty")

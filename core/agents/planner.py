"""Agente Planejador: identifica lacunas e gera hipóteses testáveis."""

from __future__ import annotations

from typing import Any

from core.agents.base import BaseAgent
from core.prompts import hypothesis_fewshot

SYSTEM = """Você é o Agente Planejador de um sistema de pesquisa científica.
Sua função é (1) identificar lacunas reais na literatura a partir das evidências
fornecidas e (2) formular hipóteses testáveis, falsificáveis e mensuráveis.

Regras:
- Nunca invente referências, números ou resultados.
- Cada hipótese deve ter variáveis independentes/dependentes e uma métrica.
- Responda SEMPRE em português do Brasil.
- Quando solicitado JSON, devolva apenas JSON válido."""


class PlannerAgent(BaseAgent):
    """Identifica lacunas de pesquisa e propõe hipóteses."""

    def __init__(self, llm: Any, temperature: float = 0.4, max_hypotheses: int = 3, **kwargs: Any) -> None:
        super().__init__(
            name="planejador",
            role="Identificação de lacunas e geração de hipóteses",
            system_prompt=SYSTEM,
            temperature=temperature,
            llm=llm,
            **kwargs,
        )
        self.max_hypotheses = max_hypotheses

    def identify_gap(self, topic: str, evidence: str, memory_context: str = "") -> Any:
        prompt = f"""TÓPICO: {topic}

{evidence}

{('CONTEXTO DE PESQUISAS ANTERIORES:' + chr(10) + memory_context) if memory_context else ''}

Identifique UMA lacuna de pesquisa concreta e não trivial, justificando com base
nas evidências listadas. Não proponha algo já resolvido na literatura acima.
Escreva no máximo 8 linhas."""
        return self.run(prompt, task="gap")

    def generate_hypotheses(self, topic: str, gap: str, evidence: str = "") -> Any:
        prompt = f"""{hypothesis_fewshot()}

TÓPICO: {topic}

LACUNA IDENTIFICADA:
{gap}

{evidence}

Gere no máximo {self.max_hypotheses} hipóteses testáveis e falsificáveis,
seguindo o padrão do exemplo acima.
Responda SOMENTE com JSON no formato:
{{"hipoteses": [{{"id": "H1", "enunciado": "...", "justificativa": "...",
"testavel": true, "variaveis": {{"independentes": ["..."], "dependentes": ["..."]}},
"metrica": "...", "risco": "baixo|medio|alto"}}]}}"""
        return self.run(prompt, expect_json=True, task="hypotheses")

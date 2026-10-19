"""Agente Experimentalista: planeja e interpreta experimentos."""

from __future__ import annotations

from typing import Any

from core.agents.base import BaseAgent

SYSTEM = """Você é o Agente Experimentalista de um sistema de pesquisa científica.
Sua função é desenhar experimentos reprodutíveis e interpretar resultados
estatísticos com honestidade intelectual.

Regras:
- Nunca invente números: interprete apenas o que lhe for fornecido.
- Sempre informe limitações e ameaças à validade.
- Exija reprodutibilidade (seed, versões, código).
- Responda SEMPRE em português do Brasil.
- Quando solicitado JSON, devolva apenas JSON válido."""


class ExperimentalistAgent(BaseAgent):
    """Planeja o desenho experimental e interpreta os resultados."""

    def __init__(self, llm: Any, temperature: float = 0.1, max_retries: int = 2, **kwargs: Any) -> None:
        super().__init__(
            name="experimentalista",
            role="Planejamento experimental e análise de resultados",
            system_prompt=SYSTEM,
            temperature=temperature,
            llm=llm,
            **kwargs,
        )
        self.max_retries = max_retries

    def design_experiment(self, topic: str, hypothesis: dict[str, Any], available_data: str = "") -> Any:
        prompt = f"""TÓPICO: {topic}

HIPÓTESE A TESTAR:
{hypothesis}

DADOS DISPONÍVEIS:
{available_data or 'Não informado; proponha um desenho com dados públicos.'}

Projete um experimento reprodutível. Responda SOMENTE com JSON:
{{"objetivo": "...", "desenho": "...", "amostra": "...",
"pre_processamento": ["..."],
"modelos": [{{"nome": "...", "tipo": "estatistico|ml"}}],
"metricas": ["..."], "testes_estatisticos": ["..."],
"limitacoes": ["..."], "reprodutibilidade": {{"seed": 42}}}}"""
        return self.run(prompt, expect_json=True, task="experiment_plan")

    def interpret_results(self, topic: str, hypothesis: dict[str, Any], results: dict[str, Any]) -> Any:
        import json

        prompt = f"""TÓPICO: {topic}

HIPÓTESE:
{json.dumps(hypothesis, ensure_ascii=False)}

RESULTADOS NUMÉRICOS OBTIDOS (não invente outros):
{json.dumps(results, ensure_ascii=False, indent=2, default=str)}

Interprete os resultados com rigor: a hipótese foi suportada? Qual o tamanho de
efeito? Quais são as limitações? Mencione valores de p e intervalos de confiança
exatamente como fornecidos."""
        return self.run(prompt, task="analysis")

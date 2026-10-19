"""Agente Integrador de Dados: conecta o ciclo científico às fontes abertas."""

from __future__ import annotations

from typing import Any

from core.agents.base import BaseAgent

SYSTEM = """Você é o Agente Integrador de Dados de um sistema de pesquisa
científica brasileiro. Você organiza evidências de fontes abertas (Crossref,
SciELO, dados.gov.br, IBGE) e as transforma em contexto utilizável.

Regras:
- Nunca invente referências, DOI ou números.
- Preserve a atribuição de cada fonte.
- Sinalize explicitamente quando uma fonte não retornou dados.
- Responda SEMPRE em português do Brasil."""


class DataIntegratorAgent(BaseAgent):
    """Agrega e sintetiza as evidências coletadas das APIs."""

    def __init__(self, llm: Any, temperature: float = 0.2, max_sources: int = 12, **kwargs: Any) -> None:
        super().__init__(
            name="integrador_dados",
            role="Integração de dados abertos brasileiros",
            system_prompt=SYSTEM,
            temperature=temperature,
            llm=llm,
            **kwargs,
        )
        self.max_sources = max_sources

    def summarize_evidence(self, topic: str, gathered: dict[str, Any]) -> Any:
        """Produz um resumo do estado da arte baseado apenas nas evidências reais."""
        import json

        artigos = gathered.get("artigos", [])[: self.max_sources]
        compact = [
            {
                "titulo": a.get("titulo"),
                "ano": a.get("ano"),
                "fonte": a.get("fonte"),
                "doi": a.get("doi"),
                "autores": (a.get("autores") or [])[:3],
            }
            for a in artigos
        ]
        prompt = f"""TÓPICO: {topic}

ARTIGOS RECUPERADOS ({len(compact)}):
{json.dumps(compact, ensure_ascii=False, indent=2)}

CONJUNTOS DE DADOS PÚBLICOS ENCONTRADOS:
{json.dumps([d.get('titulo') for d in gathered.get('datasets', [])][:6], ensure_ascii=False)}

Escreva um parágrafo de "estado da arte" (máx. 10 linhas) sintetizando o que
essas evidências indicam sobre o tópico. Cite com [1], [2], ... na ordem da
lista acima. Se houver poucas evidências, diga isso explicitamente."""
        return self.run(prompt, task="gap")

    def build_context(self, gathered: dict[str, Any], reference_block: str) -> dict[str, Any]:
        """Empacota tudo que os demais agentes precisam como contexto."""
        return {
            "artigos": gathered.get("artigos", []),
            "datasets": gathered.get("datasets", []),
            "contexto_brasil": gathered.get("contexto_brasil", []),
            "referencias_formatadas": reference_block,
            "total_artigos": len(gathered.get("artigos", [])),
        }

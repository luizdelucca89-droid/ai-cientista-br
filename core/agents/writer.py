"""Agente Redator: escrita científica iterativa em português."""

from __future__ import annotations

from typing import Any

from core.agents.base import BaseAgent

SYSTEM = """Você é o Agente Redator de um sistema de pesquisa científica brasileiro.
Você escreve artigos em português do Brasil, com linguagem acadêmica clara,
estrutura IMRAD e formatação compatível com a ABNT.

Regras absolutas:
- Nunca invente resultados, números, referências ou autores.
- Use apenas as evidências e resultados fornecidos.
- Cite referências como [1], [2], ... conforme a lista numerada fornecida.
- Inclua sempre uma seção declarando o uso de IA na elaboração do texto.
- Seja conciso; evite adjetivação vazia."""


class WriterAgent(BaseAgent):
    """Produz e refina as seções do manuscrito."""

    def __init__(self, llm: Any, temperature: float = 0.5, iterations: int = 2, **kwargs: Any) -> None:
        super().__init__(
            name="redator",
            role="Redação científica iterativa",
            system_prompt=SYSTEM,
            temperature=temperature,
            llm=llm,
            **kwargs,
        )
        self.iterations = iterations

    def write_section(self, section: str, topic: str, context: dict[str, Any]) -> Any:
        import json

        prompt = f"""Escreva a seção "{section.upper()}" de um artigo científico sobre:
TÓPICO: {topic}

CONTEXTO DISPONÍVEL (use apenas isto):
{json.dumps(context, ensure_ascii=False, indent=2, default=str)[:12000]}

Requisitos:
- Se for "resumo": 150-250 palavras, incluindo objetivo, método, resultados e conclusão.
- Se for "introducao": contextualize, apresente a lacuna e os objetivos.
- Se for "metodologia": descreva o protocolo de forma reprodutível. Descreva a base
  exatamente como consta em "resultado_experimental.dataset": se "tipo" for
  "dados_reais", diga que são observações reais e cite a "origem"; se "split" for
  "temporal", diga que o corte é temporal. Nunca afirme "base sintética" nem
  "divisão estratificada" quando os metadados dizem outra coisa.
- Se for "resultados": reporte apenas os números fornecidos em "resultado_experimental".
- Se for "conclusao": sintetize, destaque limitações e trabalhos futuros.
- Não repita o texto de outras seções."""
        return self.run(prompt, task="section")

    def revise(self, section: str, draft: str, critique: str, topic: str, context: dict[str, Any] | None = None) -> Any:
        import json

        context_block = (
            "\nDADOS REAIS QUE DEVEM SER PRESERVADOS:\n"
            + json.dumps(context or {}, ensure_ascii=False, indent=2, default=str)[:4000]
            if context
            else ""
        )
        prompt = f"""TÓPICO: {topic}

SEÇÃO: {section.upper()}

RASCUNHO ATUAL:
\"\"\"
{draft}
\"\"\"

CRÍTICA DO REVISOR:
\"\"\"
{critique}
\"\"\"
{context_block}

Reescreva a seção incorporando a crítica. Mantenha o rigor, preserve todos os
números já citados no rascunho e não invente dados.
Devolva apenas o texto final da seção."""
        return self.run(prompt, task="section")

    def write_full_paper(self, topic: str, context: dict[str, Any]) -> dict[str, Any]:
        """Gera todas as seções principais do manuscrito."""
        sections = ["resumo", "introducao", "metodologia", "resultados", "conclusao"]
        paper: dict[str, Any] = {}
        for section in sections:
            result = self.write_section(section, topic, context)
            paper[section] = result.raw.strip()
        return paper

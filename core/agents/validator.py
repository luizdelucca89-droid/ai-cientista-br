"""Agente Validador: qualidade, viés e conformidade."""

from __future__ import annotations

from typing import Any

from core.agents.base import BaseAgent

SYSTEM = """Você é o Agente Validador de um sistema de pesquisa científica.
Você audita o trabalho final quanto a rigor, reprodutibilidade, clareza,
conformidade com normas ABNT e uso ético de IA.

Regras:
- Atribua notas entre 0 e 1 por critério, com justificativa curta.
- Seja conservador: na dúvida, reduza a nota.
- Responda SEMPRE em português do Brasil.
- Quando solicitado JSON, devolva apenas JSON válido."""


class ValidatorAgent(BaseAgent):
    """Calcula um escore de qualidade e aponta pendências."""

    def __init__(self, llm: Any, temperature: float = 0.0, min_score: float = 0.7, **kwargs: Any) -> None:
        super().__init__(
            name="validador",
            role="Validação de qualidade e conformidade",
            system_prompt=SYSTEM,
            temperature=temperature,
            llm=llm,
            **kwargs,
        )
        self.min_score = min_score

    def validate(self, topic: str, manuscript: str, checklist: dict[str, Any] | None = None) -> Any:
        import json

        prompt = f"""TÓPICO: {topic}

CHECKLIST AUTOMÁTICO DO PIPELINE:
{json.dumps(checklist or {}, ensure_ascii=False, indent=2, default=str)}

MANUSCRITO:
\"\"\"
{manuscript}
\"\"\"

Avalie e responda SOMENTE com JSON:
{{"score": 0.0,
"criterios": {{"originalidade": 0.0, "rigor_metodologico": 0.0,
"reprodutibilidade": 0.0, "clareza": 0.0, "aderencia_abnt": 0.0}},
"problemas": ["..."], "aprovado": true}}"""
        return self.run(prompt, expect_json=True, task="score")

    def check_compliance(self, manuscript: str) -> Any:
        prompt = (
            "Verifique se o manuscrito abaixo cumpre requisitos mínimos de integridade "
            "científica e divulgação de uso de IA. Responda SOMENTE com JSON:\n"
            '{"divulga_uso_ia": true, "tem_limitacoes": true, "tem_referencias": true, '
            '"tem_dados_reprodutiveis": true, "problemas": ["..."]}\n\n'
            "MANUSCRITO:\n<<<\n" + manuscript + "\n>>>"
        )
        return self.run(prompt, expect_json=True, task="compliance")

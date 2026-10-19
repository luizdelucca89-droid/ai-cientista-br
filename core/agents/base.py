"""Classe base dos agentes especializados."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.utils import extract_json_block, setup_logger, truncate

logger = setup_logger(__name__)

# Instrução de correção inspirada no contrato de revisão do AI Scientist v3:
# quando o modelo não respeita o formato, repetimos UMA vez com esta correção,
# em vez de aceitar texto solto (que silenciosamente degrada o pipeline).
JSON_RETRY_CORRECTION = (
    "\n\nCORREÇÃO OBRIGATÓRIA: a resposta anterior não satisfez o contrato JSON exigido. "
    "Responda novamente com UM único objeto JSON completo, contendo todos os campos "
    "pedidos, com os tipos corretos e sem texto fora do JSON."
)


@dataclass
class AgentResult:
    """Saída padronizada de um agente."""

    agent: str
    content: Any
    raw: str = ""
    provider: str = ""
    cached: bool = False
    json_valid: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "provider": self.provider,
            "cached": self.cached,
            "json_valid": self.json_valid,
            "content": self.content,
        }


@dataclass
class BaseAgent:
    """Agente que conversa com o ``LLMClient`` seguindo um prompt de sistema."""

    name: str = "agente"
    role: str = "Agente genérico"
    system_prompt: str = "Você é um assistente científico rigoroso."
    temperature: float = 0.2
    llm: Any = None
    json_retries: int = 1

    def run(self, prompt: str, expect_json: bool = False, task: str | None = None, **kwargs: Any) -> AgentResult:
        """Executa o agente e devolve um :class:`AgentResult`.

        ``task`` é uma dica opcional consumida pelo provider mock (providers
        reais a ignoram) para selecionar o formato de resposta correto.

        Quando ``expect_json`` é verdadeiro e a resposta não contém JSON válido,
        o agente repete a chamada com uma instrução de correção antes de aceitar
        um texto bruto.
        """
        if self.llm is None:
            raise RuntimeError(f"Agente {self.name} sem LLM configurado")

        current_prompt = prompt
        attempts = self.json_retries + 1 if expect_json else 1
        last_response: Any = None
        for attempt in range(attempts):
            response = self.llm.complete(
                current_prompt, system=self.system_prompt, temperature=self.temperature, task=task, **kwargs
            )
            last_response = response
            if not expect_json:
                break
            parsed = extract_json_block(response.text)
            if parsed is not None:
                logger.debug("%s -> %s", self.name, truncate(str(parsed), 120))
                return AgentResult(
                    agent=self.name,
                    content=parsed,
                    raw=response.text,
                    provider=response.provider,
                    cached=response.cached,
                    json_valid=True,
                )
            logger.warning(
                "%s devolveu resposta sem JSON válido (tentativa %s/%s)", self.name, attempt + 1, attempts
            )
            current_prompt = prompt + JSON_RETRY_CORRECTION

        assert last_response is not None
        logger.warning("%s esgotou as tentativas de JSON; mantendo texto bruto", self.name)
        return AgentResult(
            agent=self.name,
            content={"texto_bruto": last_response.text},
            raw=last_response.text,
            provider=last_response.provider,
            cached=last_response.cached,
            json_valid=False,
        )

    def describe(self) -> dict[str, Any]:
        return {"nome": self.name, "papel": self.role, "temperature": self.temperature}

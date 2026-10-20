"""Isola a suíte do ambiente do desenvolvedor.

Os providers leem as chaves do ambiente (``GEMINI_API_KEY``, ``MISTRAL_API_KEY``
e o fallback ``OPENAI_API_KEY``). Com uma chave real exportada no shell, o
roteador por tarefa passa a ser considerado ativo e testes que esperam um
``openai_compatible`` local ou o ``mock`` passam a tentar a rede — ficando
lentos e intermitentes, ou falhando, conforme o que estiver exportado.

A suíte offline deve ser determinística: aqui as chaves de provider são
removidas para todos os testes. Os testes que as exercitam continuam podendo
defini-las explicitamente via ``monkeypatch``.
"""

import pytest

_CHAVES_DE_PROVIDER = (
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "NVIDIA_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "COHERE_API_KEY",
    "OPENAI_API_KEY",
    "AI_CIENTISTA_API_KEY",
)


@pytest.fixture(autouse=True)
def _sem_chaves_de_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for chave in _CHAVES_DE_PROVIDER:
        monkeypatch.delenv(chave, raising=False)

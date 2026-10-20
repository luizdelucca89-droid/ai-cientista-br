"""Agente de auto-debug: repara e reexecuta o código do experimento.

O pipeline científico não pode morrer porque o código gerado para o experimento
tem um bug (erro de sintaxe, shape mismatch no scikit-learn, coluna inexistente).
Esta camada:

1. executa o código do experimento em um sandbox restrito;
2. ao capturar uma exceção, envia o **traceback + o código** ao LLM e pede apenas
   a correção mínima;
3. reexecuta o código corrigido, até ``max_retries`` vezes;
4. se ainda falhar, devolve um registro honesto de falha
   (``status_experimento = "falha_apos_debug"``) para que o ciclo continue e o
   artigo reporte a dificuldade metodológica — o que é cientificamente válido.

Nenhum dado sintético é criado aqui: o código só manipula as observações reais
fornecidas no ``namespace``. O sandbox bloqueia ``open``/``exec``/``eval`` e
qualquer acesso ao sistema de arquivos.
"""

from __future__ import annotations

import ast
import traceback
from dataclasses import dataclass, field
from typing import Any

from core.utils import setup_logger, truncate

logger = setup_logger(__name__)

# Módulos que o código do experimento pode importar. Fora desta lista, o import
# é bloqueado: impede ``os``/``subprocess``/``socket`` de escaparem do sandbox.
MODULOS_PERMITIDOS: frozenset[str] = frozenset(
    {
        "math",
        "statistics",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "skimage",
    }
)

# Nomes embutidos que permitem I/O ou execução dinâmica. ``__import__`` fica de
# fora para que o único caminho de import seja o hook ``__builtins__`` abaixo.
_BUILTINS_BLOQUEADOS = ("open", "eval", "exec", "compile", "input", "breakpoint")

PROMPT_CORRECAO = """O seguinte código Python de um experimento científico falhou.

OBJETIVO CIENTÍFICO (não altere o método, corrija só o necessário para rodar):
{objetivo}

CÓDIGO QUE FALHOU:
```python
{codigo}
```

ERRO (traceback):
```
{traceback}
```

CORRIJA apenas o necessário para que o código rode e cumpra o objetivo.
Regras:
- Devolva APENAS o código Python corrigido, sem texto explicativo e sem cercas
  de código markdown.
- Não invente dados: use as variáveis já disponíveis (``X``, ``y``, ``meta``).
- Não use ``open``, ``exec``, ``eval`` nem leia arquivos.
- Preserve a lógica experimental; corrija o erro."""


class ErroDeAutoDebug(RuntimeError):
    """Falha de execução que o auto-debug não conseguiu resolver."""


@dataclass
class ResultadoExecucao:
    """Resultado de uma execução do código do experimento."""

    sucesso: bool
    codigo: str
    namespace: dict[str, Any] = field(default_factory=dict)
    erro: str = ""
    traceback: str = ""
    tentativas: int = 1
    correcoes: list[str] = field(default_factory=list)
    status_experimento: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sucesso": self.sucesso,
            "status_experimento": self.status_experimento,
            "tentativas": self.tentativas,
            "erro_final": self.erro or None,
            "correcoes_aplicadas": self.correcoes,
        }


def _import_seguro(nome: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """Substitui ``__import__`` restringindo os módulos permitidos."""
    raiz = (nome or "").split(".")[0]
    if raiz not in MODULOS_PERMITIDOS:
        raise ImportError(
            f"Import bloqueado pelo sandbox de auto-debug: {nome!r}. "
            f"Permitidos: {', '.join(sorted(MODULOS_PERMITIDOS))}"
        )
    import builtins

    return builtins.__import__(nome, globals_, locals_, fromlist, level)


def sandbox_namespace(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Monta um namespace restrito para execução de código não confiável.

    Mantém a matemática útil (``abs``, ``min``, ``sum``...) e remove tudo que
    toca o sistema operacional. ``__import__`` é trocado por ``_import_seguro``.
    """
    import builtins

    permitidos = {
        nome: getattr(builtins, nome)
        for nome in dir(builtins)
        if not nome.startswith("__") and nome not in _BUILTINS_BLOQUEADOS
    }
    permitidos["__import__"] = _import_seguro
    namespace: dict[str, Any] = {"__builtins__": permitidos}
    if base:
        namespace.update(base)
    return namespace


def _limpar_codigo(texto: str) -> str:
    """Remove cercas markdown e prefixos que o LLM costuma devolver."""
    limpo = texto.strip()
    if limpo.startswith("```"):
        linhas = limpo.splitlines()
        linhas = linhas[1:]
        if linhas and linhas[-1].strip().startswith("```"):
            linhas = linhas[:-1]
        limpo = "\n".join(linhas)
    return limpo.strip()


def executar_com_auto_debug(
    codigo: str,
    *,
    llm: Any = None,
    objetivo: str = "",
    base: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> ResultadoExecucao:
    """Executa ``codigo`` e, a cada falha, pede a correção mínima ao LLM.

    ``base`` fornece as variáveis do experimento (``X``, ``y``, ``meta``,
    ``resultado``). O namespace é recriado a cada tentativa para que uma execução
    parcial não contamine a seguinte.

    Sem ``llm``, faz uma única tentativa e devolve a falha — nunca fabrica
    sucesso.
    """
    if not codigo or not codigo.strip():
        return ResultadoExecucao(
            sucesso=False,
            codigo=codigo,
            erro="Código do experimento vazio.",
            status_experimento="falha_apos_debug",
        )

    atual = codigo
    ultimo_erro = ""
    ultimo_tb = ""
    correcoes: list[str] = []
    tentativas = max_retries if llm is not None else 1

    for tentativa in range(1, tentativas + 1):
        namespace = sandbox_namespace(base)
        try:
            ast.parse(atual)  # falha cedo em erro de sintaxe, com linha exata
            exec(compile(atual, "<experimento>", "exec"), namespace)  # noqa: S102 - sandbox restrito
            return ResultadoExecucao(
                sucesso=True,
                codigo=atual,
                namespace=namespace,
                tentativas=tentativa,
                correcoes=correcoes,
                status_experimento="ok",
            )
        except Exception as exc:  # noqa: BLE001 - qualquer falha é candidata a correção
            ultimo_erro = f"{type(exc).__name__}: {exc}"
            ultimo_tb = traceback.format_exc()
            logger.warning("Auto-debug: tentativa %s/%s falhou: %s", tentativa, tentativas, ultimo_erro)

            if llm is None or tentativa == tentativas:
                break

            prompt = PROMPT_CORRECAO.format(
                objetivo=objetivo or "executar o experimento fornecido",
                codigo=atual,
                traceback=truncate(ultimo_tb, 3000),
            )
            try:
                resposta = llm.complete(prompt, task="auto_debug")
                corrigido = _limpar_codigo(getattr(resposta, "text", "") or "")
            except Exception as exc_llm:  # noqa: BLE001
                logger.warning("Auto-debug: LLM indisponível (%s)", exc_llm)
                break
            if not corrigido:
                logger.warning("Auto-debug: LLM devolveu correção vazia.")
                break
            atual = corrigido
            correcoes.append(truncate(corrigido, 300))

    logger.error("Auto-debug esgotado após %s tentativa(s): %s", tentativas, ultimo_erro)
    return ResultadoExecucao(
        sucesso=False,
        codigo=atual,
        erro=ultimo_erro,
        traceback=ultimo_tb,
        tentativas=tentativas,
        correcoes=correcoes,
        status_experimento="falha_apos_debug",
    )


__all__ = [
    "ErroDeAutoDebug",
    "MODULOS_PERMITIDOS",
    "PROMPT_CORRECAO",
    "ResultadoExecucao",
    "executar_com_auto_debug",
    "sandbox_namespace",
]

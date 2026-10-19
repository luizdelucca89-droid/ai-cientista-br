"""Conferência determinística de fidelidade numérica do manuscrito.

O validador por LLM lê o texto e opina, mas não compara número a número. Na
prática ele deixou passar um erro factual: o manuscrito afirmou que as semanas
epidemiológicas iam de ``200201`` a ``202351`` quando o dado real era ``202201``.
Um dígito trocado, e a afirmação sobre a base observada fica falsa.

Este módulo faz a conferência que não depende de julgamento: extrai os números do
manuscrito e verifica se cada um deles existe, a menos de arredondamento, no
conjunto de números produzidos pelo experimento e pela coleta de dados. O que não
tem respaldo é reportado como número não sustentado.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Números com 3+ dígitos ou com parte decimal são os que carregam informação
# (contagens, métricas, semanas). Inteiros pequenos soltos são quase sempre
# numeração de seção, item de lista ou índice de referência — não vale acusá-los.
_PADRAO_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")

# Faixa que concentra numeração estrutural do texto (seções, listas, tabelas).
_LIMITE_ESTRUTURAL = 20

# Numeração de seção/subseção no início de uma linha de título ("### 3.1. Foo").
_PADRAO_TITULO_NUMERADO = re.compile(r"^(#{1,6}\s*)\d+(?:[.,]\d+)*[.)]?\s+", re.MULTILINE)

# Referências cruzadas no corpo ("Seção 3.1", "Tabela 2.4", "Figura 1").
_PADRAO_REFERENCIA_CRUZADA = re.compile(
    r"(?:se[çc][ãa]o|secoes|tabela|figura|equa[çc][ãa]o|quadro|item|passo|ap[êe]ndice)\s+\d+(?:[.,]\d+)*",
    re.IGNORECASE,
)

# Versões de software ("Python 3.13.15"), que não são medições.
_PADRAO_VERSAO = re.compile(r"[A-Za-z]+\s+\d+(?:\.\d+){1,3}")


def _remover_estrutura(manuscrito: str) -> str:
    """Remove numeração que não é afirmação sobre dados.

    Títulos numerados pelo próprio LLM ("### 3.1. Amostragem"), referências
    cruzadas ("Seção 3.1") e versões de software ("Python 3.13.15") geram tokens
    que parecem números de resultado, mas são estrutura do texto. Sem podá-los, o
    conferidor acusa numeração de seção como se fosse dado sem lastro.
    """
    texto = _PADRAO_TITULO_NUMERADO.sub(r"\1", manuscrito)
    texto = _PADRAO_REFERENCIA_CRUZADA.sub("", texto)
    texto = _PADRAO_VERSAO.sub("", texto)
    return texto


def _para_float(token: str) -> float | None:
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def collect_numbers(origem: Any) -> set[float]:
    """Percorre uma estrutura (dict/list/escalar) e reúne todos os números.

    Percorre as chaves e os valores: nomes de variáveis e rótulos como
    ``acuracia_ic95_inf`` não contêm dígitos relevantes, mas chaves de séries e
    códigos numéricos contêm, e também precisam servir de lastro.
    """
    numeros: set[float] = set()

    def visitar(no: Any) -> None:
        if isinstance(no, bool):
            return
        if isinstance(no, (int, float)):
            numeros.add(float(no))
        elif isinstance(no, str):
            for token in _PADRAO_NUMERO.findall(no):
                valor = _para_float(token)
                if valor is not None:
                    numeros.add(valor)
        elif isinstance(no, dict):
            for chave, valor in no.items():
                visitar(chave)
                visitar(valor)
        elif isinstance(no, (list, tuple, set)):
            for item in no:
                visitar(item)

    visitar(origem)
    return numeros


def _sustentado(valor: float, fontes: Iterable[float], tolerancia: float = 1e-6) -> bool:
    """Diz se ``valor`` encontra lastro num dos números das fontes.

    Aceita igualdade exata, arredondamento (o texto pode exibir 0,8989 para
    0,89891) e truncamento na casa exibida. Sem isso, todo número arredondado
    seria acusado.
    """
    for fonte in fontes:
        if abs(valor - fonte) <= tolerancia:
            return True
        if round(fonte, 4) == round(valor, 4):
            return True
        if round(fonte, 3) == round(valor, 3):
            return True
        if round(fonte, 2) == round(valor, 2):
            return True
        if int(fonte) == int(valor) and abs(valor - int(valor)) < tolerancia:
            return True
    return False


def find_unsupported_numbers(manuscrito: str, fontes: Iterable[float]) -> list[float]:
    """Números do manuscrito sem lastro nos dados; ordenados e sem repetição.

    A comparação é feita na casa exibida: ``200201`` só é aceito se existir nas
    fontes, então um dígito trocado é acusado.
    """
    disponiveis = list(fontes)
    suspeitos: set[float] = set()
    for token in _PADRAO_NUMERO.findall(_remover_estrutura(manuscrito)):
        valor = _para_float(token)
        if valor is None:
            continue
        tem_decimal = "." in token or "," in token
        if not tem_decimal and valor < _LIMITE_ESTRUTURAL:
            continue  # numeração de seção, item de lista, índice de referência
        if not _sustentado(valor, disponiveis):
            suspeitos.add(valor)
    return sorted(suspeitos)

"""Normalização de títulos de seção entre o texto do LLM e o formatador."""

from __future__ import annotations

import re
import unicodedata


def _chave(texto: str) -> str:
    """Reduz um título a uma forma comparável.

    Ignora marcadores (``##``, ``**``, ``\\section{``), numeração ("1.", "2)")
    e acentuação/pontuação, de modo que "1. INTRODUÇÃO", "**RESUMO**" e
    "Introdução" coincidam.
    """
    texto = texto.strip().lower()
    texto = re.sub(r"^(#{1,6}\s*|\\[a-z]*section\*?\{)", "", texto)
    texto = re.sub(r"^[*_]{1,3}", "", texto)
    texto = re.sub(r"[*_]{1,3}$", "", texto)
    texto = texto.rstrip("}").strip()
    texto = re.sub(r"^[0-9]+[.)]?\s*", "", texto)
    return re.sub(r"[^a-zà-ú0-9]", "", texto)


def _tem_marcador(linha: str) -> bool:
    """Indica se a linha é um título, e não uma frase que começa com o tema.

    "Introdução" (sem marcador) pode ser o começo de uma frase; "## Introdução"
    ou "**Introdução**" é claramente um título.
    """
    return bool(re.match(r"^(#{1,6}\s|\*{1,3}\S|\\[a-z]*section\*?\{)", linha))


def strip_duplicate_heading(content: str, heading: str) -> str:
    """Remove o título que o LLM já escreveu no início do corpo da seção.

    O redator costuma devolver "## 1. INTRODUÇÃO\\n\\ntexto...", e o formatador
    adiciona o seu próprio título. Sem remover o do LLM, o paper sai com
    "## 1 INTRODUÇÃO" seguido de "## 1. INTRODUÇÃO" — o defeito aparecia em
    todas as seções de todos os ciclos.

    Duas salvaguardas evitam comer texto do corpo:

    - a linha só é removida se tiver marcador de título (``#``, ``**`` ou
      ``\\section{``), o que distingue um título de uma frase que por acaso
      começa com a mesma palavra ("Introdução é o tema...");
    - a comparação aceita prefixo, porque o LLM abrevia ("3. RESULTADOS" para
      a seção "3 RESULTADOS E DISCUSSÃO").
    """
    if not content:
        return content
    linhas = content.lstrip("\n").split("\n")
    if not linhas:
        return content
    primeira = linhas[0].strip()
    if not _tem_marcador(primeira):
        return content
    alvo, esperado = _chave(primeira), _chave(heading)
    if alvo and (alvo == esperado or esperado.startswith(alvo) or alvo.startswith(esperado)):
        return "\n".join(linhas[1:]).lstrip("\n")
    return content


def _sem_acento(texto: str) -> str:
    """Remove diacríticos para comparar títulos sem depender da grafia."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )


def _e_declaracao_ia(linha: str) -> bool:
    """Reconhece os vários títulos que o LLM usa para a declaração de IA.

    Aparecem "Declaração de Uso de IA", "Declaração sobre o Uso de Inteligência
    Artificial" e "Declaração de Uso de Inteligência Artificial" — todos são a
    mesma declaração e todos devem ser podados do corpo das seções.
    """
    chave = _sem_acento(_chave(linha))
    if not chave.startswith("declaracao"):
        return False
    return "usodeia" in chave or "usodeinteligencia" in chave


def renumber_subsections(content: str, section_number: int) -> str:
    """Ajusta o prefixo numérico das subseções ao número da seção.

    O redator numera as subseções por conta própria e erra: escreve "### 3.1"
    dentro da seção 2 (METODOLOGIA) e "### 4.1" dentro da seção 3 (RESULTADOS).
    Aqui o prefixo é reescrito para "2.1", "3.1" etc., preservando o restante do
    título. Subseções sem número são deixadas como estão.
    """
    if not content or section_number < 1:
        return content
    linhas = content.split("\n")
    sub = 0
    for i, linha in enumerate(linhas):
        # Casa "### 3.1. Título", "### 3.1 Título" e "### 3. Título"; o número
        # de subseção pode ter um ou dois níveis.
        m = re.match(r"^(#{3,6}\s*)[0-9]+(?:\.[0-9]+)*[.)]?\s+(.+)$", linha.strip())
        if m:
            sub += 1
            linhas[i] = f"{m.group(1)}{section_number}.{sub}. {m.group(2)}"
    return "\n".join(linhas)


def strip_ai_declaration_blocks(content: str) -> str:
    """Remove blocos de "Declaração de Uso de IA" escritos dentro da seção.

    O redator encerra cada seção com a sua própria declaração de uso de IA, e o
    formatador emite uma declaração canônica ao final do documento. Sem podar as
    internas, o paper repetia o mesmo parágrafo quatro vezes, o que passa
    impressão de descuido e infla o texto.

    Só remove o bloco quando ele é o fecho da seção (do título até o fim), para
    não apagar conteúdo que venha depois.
    """
    if not content:
        return content
    linhas = content.split("\n")
    corte: int | None = None
    for i, linha in enumerate(linhas):
        if not _tem_marcador(linha.strip()):
            continue
        if _e_declaracao_ia(linha):
            corte = i
            break
    if corte is None:
        return content
    return "\n".join(linhas[:corte]).rstrip().rstrip("-").rstrip()

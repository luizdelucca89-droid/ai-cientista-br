"""Benchmark de modelos LLM gratuitos contra as tarefas reais do projeto.

Testa cada modelo com o formato que o pipeline de fato exige — uma resposta
JSON em português com o desenho experimental — porque é aí que os provedores
falham na prática ("esgotou as tentativas de JSON", timeouts do redator).

Mede por modelo: disponibilidade, latência, JSON válido e presença de
português. Não inventa dado: apenas chama a API e registra o que voltou.

Uso::

    export OPENROUTER_API_KEY=...   # opcional
    export MISTRAL_API_KEY=...      # opcional
    export GEMINI_API_KEY=...       # opcional
    python benchmark_models.py [--provider openrouter|mistral|gemini|all]

Grava ``outputs/benchmark_modelos.json`` e imprime uma tabela.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.utils import ensure_dir, setup_logger  # noqa: E402

logger = setup_logger("benchmark_modelos")

ARQUIVO_SAIDA = "outputs/benchmark_modelos.json"

# Prompt espelha a tarefa `experiment_plan` do experimentalista: exige JSON
# estrito com campos que o runner consome (modelos, defasagem, métricas).
PROMPT = (
    "Você é um experimentalista. Para o tópico 'Predição de surtos de dengue no "
    "Brasil com dados climáticos', proponha um desenho experimental.\n"
    "Responda APENAS com JSON válido, sem texto fora do JSON, no formato:\n"
    '{"modelos": ["..."], "defasagem_semanas": 0, "metricas": ["..."], '
    '"justificativa": "uma frase em português"}'
)

# Modelos gratuitos do OpenRouter que são de chat/texto. Excluídos os de
# embedding, guardrail, áudio e imagem, que não completam a tarefa.
MODELOS_OPENROUTER = [
    "cohere/north-mini-code:free",
    "dots-studio/dots-3-note-preview:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-2.6b:free",
    "minimax/minimax-m3.1-flash:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "thinkingmachines/inkling-small:free",
    "thinkingmachines/inkling:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
]

# Modelos Mistral com completion_chat. Excluídos OCR, TTS, transcribe e embed.
MODELOS_MISTRAL = [
    "mistral-medium-latest",
    "mistral-small-latest",
    "mistral-medium-3-5",
    "magistral-medium-latest",
    "magistral-small-latest",
    "ministral-14b-latest",
    "ministral-8b-latest",
    "codestral-latest",
    "labs-leanstral-1-5",
    "voxtral-small-latest",
]

MODELOS_GEMINI = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Modelos NVIDIA NIM (catálogo integrate.api.nvidia.com), OpenAI-compatible.
# Foco nos de texto/raciocínio; excluídos visão, áudio, safety e embeddings.
MODELOS_NVIDIA = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3",
    "moonshotai/kimi-k2.6",
    "mistralai/mistral-large",
    "openai/gpt-oss-20b",
    "google/gemma-4-31b-it",
    "meta/muse-glimmer-30b",
]

_ENDPOINTS = {
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "nvidia": ("NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1"),
}


def _tem_portugues(texto: str) -> bool:
    """Heurística simples: marcadores comuns de PT-BR no texto."""
    marcadores = ("ão", "ção", "ê", "ã", "õ", "português", "justificativa", "semanas")
    return any(m in texto.lower() for m in marcadores)


def _extrair_json(texto: str) -> dict | None:
    """Tenta decodificar JSON puro ou o primeiro bloco {...} da resposta."""
    texto = texto.strip()
    for candidato in (texto, _bloco_mais_externo(texto)):
        if not candidato:
            continue
        try:
            valor = json.loads(candidato)
            return valor if isinstance(valor, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _bloco_mais_externo(texto: str) -> str | None:
    """Recorta do primeiro ``{`` ao último ``}`` (respostas com cercas markdown)."""
    ini, fim = texto.find("{"), texto.rfind("}")
    return texto[ini : fim + 1] if ini != -1 and fim > ini else None


def _chamar(base_url: str, api_key: str, modelo: str, timeout: float = 120.0, tentativas: int = 3) -> dict:
    """Uma chamada de chat completion; devolve métricas sem levantar exceção.

    ``max_tokens`` alto porque modelos de raciocínio gastam o orçamento em
    ``reasoning`` antes de emitir o JSON: com 512 tokens o conteúdo vinha vazio e
    o modelo era reprovado por truncamento, não por incapacidade. O 429 é
    retentado com espera crescente, porque os tiers gratuitos limitam por minuto.
    """
    payload = {
        "model": modelo,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ai-cientista-br-benchmark/1.0",
        },
        method="POST",
    )
    inicio = time.perf_counter()
    for tentativa in range(tentativas):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                corpo = json.loads(resp.read().decode("utf-8"))
                status = resp.status
            break
        except urllib.error.HTTPError as exc:
            detalhe = exc.read().decode("utf-8", errors="replace")[:200]
            # 429 (rate limit) e 5xx são transitórios; 4xx definitivos não.
            if exc.code in (429, 500, 502, 503) and tentativa < tentativas - 1:
                time.sleep(5 * (tentativa + 1))
                continue
            return {
                "modelo": modelo, "ok": False, "status": exc.code,
                "erro": f"HTTP {exc.code}: {detalhe}", "latencia_s": round(time.perf_counter() - inicio, 2),
            }
        except Exception as exc:  # noqa: BLE001 - o benchmark continua os demais
            if tentativa < tentativas - 1:
                time.sleep(5 * (tentativa + 1))
                continue
            return {
                "modelo": modelo, "ok": False, "status": None,
                "erro": f"{type(exc).__name__}: {exc}", "latencia_s": round(time.perf_counter() - inicio, 2),
            }

    latencia = round(time.perf_counter() - inicio, 2)
    escolha = (corpo.get("choices") or [{}])[0]
    mensagem = escolha.get("message") or {}
    texto = (mensagem.get("content")) or ""
    # Alguns modelos devolvem só o rastro de raciocínio; sem conteúdo não há JSON.
    uso = corpo.get("usage") or {}
    obj = _extrair_json(texto)
    return {
        "modelo": modelo,
        "ok": True,
        "status": status,
        "latencia_s": latencia,
        "json_valido": obj is not None,
        "portugues": _tem_portugues(texto),
        "tokens_saida": uso.get("completion_tokens"),
        "trecho": texto.strip()[:160],
    }


def _testar_provider(nome: str, modelos: list[str], pausa: float = 1.5) -> list[dict]:
    env, base_url = _ENDPOINTS[nome]
    chave = os.environ.get(env)
    if not chave:
        logger.warning("Provider %s sem %s no ambiente; pulando", nome, env)
        return []
    logger.info("Testando %s (%s modelos)", nome, len(modelos))
    resultados: list[dict] = []
    for modelo in modelos:
        r = _chamar(base_url, chave, modelo)
        r["provider"] = nome
        resultados.append(r)
        marca = "OK " if r.get("ok") and r.get("json_valido") else "FALHA"
        print(
            f"  [{marca}] {nome}/{modelo} "
            f"({r.get('latencia_s')}s, json={r.get('json_valido')}, pt={r.get('portugues')})"
            + (f" -> {r.get('erro')}" if not r.get("ok") else ""),
            flush=True,
        )
        time.sleep(pausa)
    return resultados


def _tabela(resultados: list[dict]) -> str:
    linhas = [
        "=" * 96,
        "BENCHMARK DE MODELOS LLM — TAREFA: desenho experimental em JSON (PT-BR)",
        "=" * 96,
        f"{'PROVIDER':<12} {'MODELO':<48} {'OK':<3} {'JSON':<5} {'PT':<3} {'LAT(s)':<7} {'TOK':<5}",
        "-" * 96,
    ]
    for r in resultados:
        linhas.append(
            f"{r.get('provider',''):<12} {r['modelo']:<48} "
            f"{('sim' if r.get('ok') else 'nao'):<3} "
            f"{('sim' if r.get('json_valido') else 'nao'):<5} "
            f"{('sim' if r.get('portugues') else 'nao'):<3} "
            f"{str(r.get('latencia_s','-')):<7} {str(r.get('tokens_saida','-')):<5}"
        )
    linhas.append("-" * 96)
    aprovados = [r for r in resultados if r.get("ok") and r.get("json_valido")]
    linhas.append(f"APROVADOS (JSON válido): {len(aprovados)}/{len(resultados)}")
    if aprovados:
        linhas.append("Melhores por latência:")
        for r in sorted(aprovados, key=lambda x: x.get("latencia_s", 9e9))[:5]:
            linhas.append(f"  - {r['provider']}/{r['modelo']} ({r['latencia_s']}s)")
    linhas.append("=" * 96)
    return "\n".join(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark de modelos LLM gratuitos")
    ap.add_argument("--provider", default="all", choices=["openrouter", "mistral", "gemini", "nvidia", "all"])
    args = ap.parse_args()

    resultados: list[dict] = []
    if args.provider in ("openrouter", "all"):
        resultados += _testar_provider("openrouter", MODELOS_OPENROUTER)
    if args.provider in ("mistral", "all"):
        resultados += _testar_provider("mistral", MODELOS_MISTRAL)
    if args.provider in ("gemini", "all"):
        resultados += _testar_provider("gemini", MODELOS_GEMINI)
    if args.provider in ("nvidia", "all"):
        resultados += _testar_provider("nvidia", MODELOS_NVIDIA)

    relatorio = _tabela(resultados)
    print("\n" + relatorio)
    ensure_dir(Path(ARQUIVO_SAIDA).parent)
    Path(ARQUIVO_SAIDA).write_text(
        json.dumps({"resultados": resultados}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Resultados gravados em %s", ARQUIVO_SAIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

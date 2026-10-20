# Modelos LLM validados — referência permanente

**Última execução:** 2026-10-19 · **Harness:** `benchmark_models.py` · **Tarefa:** desenho experimental em JSON (PT-BR), a mesma que o experimentalista usa.

Este arquivo é o registro de quais modelos gratuitos **de fato funcionam** neste
projeto e para qual função cada um é melhor. Não precisa refazer os testes: só
reexecute o harness se uma chave expirar ou um modelo for descontinuado.

> ⚠️ Tiers gratuitos mudam sem aviso. Vários modelos que existiam em 2025 já
> retornam 404/410 (ver seção Descontinuados). Confira o `resultado` antes de
> confiar.

## Como reproduzir

```bash
export OPENROUTER_API_KEY=...   # https://openrouter.ai/keys
export MISTRAL_API_KEY=...      # https://console.mistral.ai
export NVIDIA_API_KEY=nvapi-... # https://build.nvidia.com
export GEMINI_API_KEY=...       # https://aistudio.google.com/apikey
python benchmark_models.py --provider all
```

Grava `outputs/benchmark_modelos.json` e imprime a tabela.

## Aprovados (JSON válido na tarefa real)

### 🧠 Raciocínio / planejamento científico — melhor escolha

| Modelo | Provider | Latência | Por que |
|---|---|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA NIM | 5,0 s | **Melhor custo-benefício.** 550B MoE (55B ativos), 1M ctx, forte em planejamento multi-etapa. Rápido e confiável. |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter | 6,5 s | Mesmo modelo via OpenRouter (limite 50 req/dia). |
| `gemini-3.5-flash` | Google | 8,1 s | Rápido e forte em ciência (GPQA líder na família). Bom para `gap`/`hypotheses`. |

### ✍️ Redação / texto longo em português — melhor escolha

| Modelo | Provider | Latência | Por que |
|---|---|---|---|
| `gemini-flash-latest` | Google | 73,2 s | Contexto longo, prosa PT-BR sólida. Lento — use só na redação, não em loop. |
| `nvidia/muse-glimmer-30b` | NVIDIA NIM | 52,9 s | Multimodal, boa escrita; lento. |
| `google/gemma-4-31b-it` | NVIDIA NIM | 20,8 s | 140+ idiomas, Apache 2.0, bom em documento/ABNT. |
| `mistral/ministral-14b-latest` | Mistral | 10,5 s | Equilíbrio qualidade/velocidade em PT. |

### 🧪 Código / experimento — melhor escolha

| Modelo | Provider | Latência | Por que |
|---|---|---|---|
| `mistral/codestral-latest` | Mistral | 1,1 s | **Mais rápido.** Especializado em código; ideal para `experiment_plan`/`auto_debug`. |
| `mistral/voxtral-small-latest` | Mistral | 1,1 s | Velocíssimo; também aceita áudio. Ótimo para tarefas curtas de volume. |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | NVIDIA NIM | 9,1 s | MoE 30B A3B, foco em tarefas agênticas especializadas. |
| `nvidia/nemotron-3.5-lightning:free` | OpenRouter | 14,0 s | Mesmo modelo, via OpenRouter. |
| `cohere/north-mini-code:free` | OpenRouter | 9,9 s | Agêntico de código, Apache 2.0, 256K ctx. |

### ⚡ Alto volume / classificação / score

| Modelo | Provider | Latência | Por que |
|---|---|---|---|
| `mistral/ministral-8b-latest` | Mistral | 4,8 s | Rápido e barato; bom para `score`/`compliance`. |
| `mistral/ministral-3b-latest` | Mistral | — | Ainda mais leve, para tarefas triviais de volume. |
| `nvidia/nemotron-3.5-super-vl:free` | OpenRouter | 7,8 s | O mais rápido do OpenRouter no teste; multimodal. |
| `minimax/minimax-m3.1-flash:free` | OpenRouter | 10,0 s | 1M ctx, rápido. |
| `poolside/laguna-xs-2.1:free` | OpenRouter | 10,3 s | Agêntico de código, 256K ctx. |

### 🔄 Validação cruzada (segunda opinião do revisor crítico)

Para o parecer adversário, chame o **mesmo prompt em providers diferentes** e
compare — é o que dá robustez real ao ensemble:

| Modelo | Provider | Latência |
|---|---|---|
| `xingchen-agi/xing4.0-29b-a4b:free` | OpenRouter | 11,4 s |
| `dots-studio/dots-3-note-preview:free` | OpenRouter | 14,9 s |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | OpenRouter | 11,7 s |
| `liquid/lfm-2.5-2.6b:free` | OpenRouter | 18,9 s |
| `z-ai/glm-5.3` | NVIDIA NIM | 31,5 s |
| `poolside/laguna-s-2.1:free` | OpenRouter | 26,5 s |
| `openai/gpt-oss-20b` | NVIDIA NIM | 14,1 s |
| `gemini-3.5-flash-lite` | Google | 15,6 s |

## Reprovações (e o motivo — nem sempre é o modelo)

| Modelo | Provider | Status | Motivo real |
|---|---|---|---|
| `mistral-medium-latest` | Mistral | 429 | Cota por minuto da conta = 0 (plano free). Não é falha do modelo. |
| `mistral-small-latest` | Mistral | 429 | Idem. |
| `gemini-3.1-pro-preview` | Google | 429 | Cota free esgotada para Pro. |
| `gemini-pro-latest` | Google | 429 | Idem. |
| `gemini-3.8-flash` | Google | 503 | Alta demanda momentânea. Retentar. |
| `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | 404 | Função não habilitada para esta conta. |
| `nvidia/nemotron-nano-3-30b-a3b` | NVIDIA | 404 | Idem. |
| `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA | 404 | Idem. |
| `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter | 404 | Provider upstream indisponível. |
| `moonshotai/kimi-k2.6` | NVIDIA | 404 | Não habilitado para esta conta. |
| `mistralai/mistral-large` | NVIDIA | 404 | Idem. |
| `google/gemma-4-26b-a4b-it:free` | OpenRouter | 429/erro | Provider upstream instável. |
| `google/gemma-4-31b-it:free` | OpenRouter | 429/erro | Idem. Use via NVIDIA NIM (funciona). |
| `thinkingmachines/inkling*:free` | OpenRouter | 403 | Só disponível em "agentic harnesses", não via API direta. |
| `deepseek-ai/deepseek-v4.1-flash` | NVIDIA | timeout | 375 s — inviável para pipeline. |

### Descontinuados (não usar — retornam 404/410)

- `gemini-2.5-flash`, `gemini-2.5-pro` → substituir por `gemini-3.5-flash`.
- `meta/llama-3.3-70b-instruct` → EOL em 2026-08-26.
- `gemini-2.0-flash` → migrar para `gemini-3.8-flash`.

## Recomendação de stack (roteada por tarefa)

A dica `task=` já existe nos agentes e agora é **consumida pelo `LLMClient`**
(ver `TaskRouter` em `llm_providers/__init__.py`). A rota de cada categoria vem
de `roteamento_por_tarefa` no `config.modelos.json`:

```
raciocinio  (gap, hypotheses, novelty, review)      -> nemotron-3-ultra (NVIDIA) / gemini-3.5-flash
redacao     (section, translate, rebuttal)          -> gemini-flash-latest / gemma-4-31b-it
codigo      (experiment_plan, analysis, auto_debug) -> codestral-latest (1,1 s) / north-mini-code
alto_volume (score, compliance, voting)             -> ministral-8b-latest / nemotron-3.5-super-vl
multimodal  (satellite_features, chart_reading)     -> nemotron-3.5-super-vl (NVIDIA/OpenRouter)
validacao   (revisor critico, 2a opiniao)           -> xing4.0 / glm-5.3 / gpt-oss-20b
fallback global                                     -> gemini-flash-latest -> mock
```

## Notas de operação

- **`max_tokens` importa.** Modelos de raciocínio gastam o orçamento em
  `reasoning` antes de emitir o JSON. Com 512 tokens o conteúdo vinha vazio e o
  modelo era reprovado por truncamento, não por incapacidade. Use ≥2048 no
  benchmark e ≥8192 no redator.
- **OpenRouter free:** 50 requisições/dia (1.000/dia após US$10 de crédito).
  Confira com `GET /api/v1/key`.
- **Privacidade:** o free tier do Gemini usa os prompts para melhorar produtos
  do Google. Para dados de pesquisa sensíveis, prefira Mistral/NVIDIA.
- **NVIDIA NIM:** algumas funções exigem habilitação na conta — daí os 404.

_Registro gerado por agente de IA (OpenHands) a partir de execução real do
`benchmark_models.py`._

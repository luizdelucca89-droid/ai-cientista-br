# AI Cientista Brasileiro

Sistema multi-agente de pesquisa científica autônoma em português do Brasil.
Coleta evidências em fontes abertas reais, formula hipóteses, **executa
experimentos de verdade**, analisa os resultados com estatística e escreve um
artigo formatado em ABNT.

O sistema funciona **sem custo**: o provider padrão é offline e determinístico,
e os provedores pagos são opcionais.

---

## O que é real (e o que não é)

Honestidade é parte do projeto. Esta tabela diz exatamente o que o software faz.

| Componente | Situação |
|---|---|
| Coleta de evidências | **Real.** Crossref, SciELO/ArticleMeta, OpenAlex, Banco Central (SGS) e IBGE são consultados ao vivo. |
| Execução de experimentos | **Real.** Treina regressão logística, floresta aleatória e gradient boosting; calcula acurácia, F1, AUC, intervalos de confiança bootstrap e testes t pareado/Wilcoxon. |
| Métricas no artigo | **Reais.** Os números vêm da execução, nunca de texto gerado. |
| Revisão por pares | **Real.** Painel de 5 revisores independentes com meta-revisor e voto majoritário ordinal. |
| Formatação ABNT | **Real.** Gera Markdown, PDF (ABNT) e LaTeX (`.tex` + `paper_latex.pdf` compilado com `pdflatex`). |
| Base experimental | **Real quando existe fonte pública.** Para temas com dados abertos brasileiros, o sistema baixa observações reais (ex.: InfoDengue/Fiocruz para dengue) e usa corte temporal com linha de base de persistência. Sem fonte disponível, cai numa base sintética reprodutível (seed fixa), e os metadados marcam `tipo: sintetica`. |
| Fidelidade numérica | **Real.** Antes de fechar o ciclo, o sistema confere cada número do manuscrito contra os dados produzidos e reporta os que não têm lastro. |
| Texto do artigo | Gerado pelo LLM escolhido. O provider `mock` produz texto estrutural, não prosa científica. |
| Lattes | **Sem API pública.** O sistema declara a limitação e devolve um link de consulta manual; não simula dados. |
| dados.gov.br | Exige token de API; por isso o contexto de dados abertos usa IBGE/SIDRA, que é público. |

Sempre divulgue o uso de IA e revise o resultado antes de publicar.

---

## Comparação com o AI Scientist (Sakana AI)

O projeto foi inspirado no [AI Scientist](https://github.com/SakanaAI/AI-Scientist)
e no [AI Scientist v3](https://github.com/SakanaAI/AI-Scientist-v3). O foco aqui é
diferente: **custo zero, português do Brasil, ABNT e fontes de dados nacionais**.
As técnicas abaixo foram portadas (ou reimplementadas) a partir dos originais.

| Técnica | Origem | Situação aqui |
|---|---|---|
| Contrato JSON com retentativa de correção | v3 `aireview.py` | `core/agents/base.py` — repete uma vez com instrução corretiva antes de aceitar texto solto. |
| Painel de N revisores + meta-revisão | v1 `perform_review.py`, v3 | `core/review/panel.py` — 5 revisores em paralelo, meta-revisor, voto majoritário ordinal. |
| Exemplos few-shot para calibrar o revisor | v1 `fewshot_examples/` | `core/prompts.py` — parecer e hipótese modelo em português. |
| Busca bibliográfica no OpenAlex | v1 (`engine="openalex"`) | `integrations/brazilian_apis.py` — enriquece referências com contagem de citações. |
| Contabilidade de tokens/custo | v3 `cost_tracking.py` | `LLMClient.stats()` — tokens estimados, cache hits, latência por provider. |
| Diagnóstico do ambiente (`doctor`) | v3 `agsci doctor` | `python main.py --doctor` — Python, dependências, saída, providers e fontes. |
| Execução paralela de vários tópicos | v1 `--parallel` | `python main.py --topics-file ... --parallel N` — um processo por tópico. |
| Artigo em LaTeX | v1 `--writeup latex` | `core/paper/latex.py` — gera `paper.tex` e compila `paper_latex.pdf` com `pdflatex`. |

O que **não** foi portado, por depender de infraestrutura ou pressupostos que não
se aplicam aqui: execução em contêiner Docker com agentes de código (v3),
barramento de GPUs, presets de tarefas com GPU e integração com CLIs comerciais
(`claude`, `codex`, `cursor`).

---

## Instalação

```bash
git clone <url-do-repositorio>
cd ai-cientista-brasileiro

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Para gerar PDF via LaTeX (opcional, mas recomendado):

```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-lang-portuguese
```

Sem o `pdflatex`, o ciclo continua normalmente e grava apenas `paper.tex`; o
`paper.pdf` (ABNT) é sempre gerado via `fpdf2`.

## Uso

```bash
# Diagnóstico do ambiente (não executa o pipeline)
python main.py --doctor

# Lista tópicos de exemplo
python main.py --examples

# Ciclo completo (provider offline por padrão)
python main.py --topic "Predição de surtos de dengue usando dados climáticos no Brasil"

# Apenas coletar evidências, sem LLM
python main.py --topic "dengue e clima" --collect-only

# Vários tópicos em paralelo (um processo por tópico)
printf '%s\n' "dengue e clima" "irrigação no Cerrado" > topicos.txt
python main.py --topics-file topicos.txt --parallel 2

# Estatísticas da memória persistente
python main.py --memory-stats

# Repetir um tópico já estudado
python main.py --topic "..." --force
```

Exemplos executáveis:

```bash
python examples/quick_start.py      # dengue
python examples/agronegocio.py      # irrigação no Cerrado
python examples/educacao.py         # gamificação
```

Uso como biblioteca:

```python
from core.config import Config
from core.orchestrator import Orchestrator

orc = Orchestrator(config=Config.load())
orc.initialize_agents()
try:
    results = orc.run_research_cycle("Predição de surtos de dengue no Brasil", force=True)
finally:
    orc.close()

print(results["status"])                                  # concluido
print(results["phases"]["experimento"]["resultado"]["best_model"])
print(results["phases"]["artefatos"]["final_paper"])      # caminho do PDF
```

---

## Providers de LLM

Configure em `config.json` (ou por variáveis de ambiente).

| Provider | Requer | Observação |
|---|---|---|
| `mock` (padrão) | nada | Offline, determinístico, custo zero. Texto estrutural. |
| `openai_compatible` | `OPENAI_API_KEY` | OpenAI, Groq, Together, Ollama, vLLM, LM Studio... |
| `selenium` | sessão autenticada | Automação de interfaces web. Desativado por padrão; respeite os ToS. |

```bash
# Exemplo com Groq (tier gratuito)
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_API_KEY="sua-chave"
python main.py --topic "..." --provider openai_compatible --model openai/gpt-oss-120b
```

Variáveis de ambiente aceitas: `AI_CIENTISTA_PROVIDER`, `AI_CIENTISTA_MODEL`,
`OPENAI_BASE_URL`, `AI_CIENTISTA_OUTPUT_DIR`, `AI_CIENTISTA_SEED`.

O cliente aplica cache em SQLite e uma cadeia de fallback: se o provider
principal falhar, o ciclo continua com o próximo da lista. O resumo final mostra
quantas chamadas foram para cada provider e avisa quando parte delas caiu no
`mock`, para que o texto gerado não seja atribuído ao modelo errado por engano.

### Limites de uso

Tiers gratuitos impõem teto de tokens por minuto (TPM) **e por dia** (TPD). A
Groq, por exemplo, permite 8.000 TPM e 200.000 TPD no `on_demand`: um ciclo
completo consome ~60 mil tokens, então três execuções já esgotam o dia.

O cliente trata isso em camadas:

1. Respeita o tempo sugerido pela API em HTTP 429 (`try again in 27.66s` e
   também `try again in 26m16s`).
2. Se a espera pedida passa de 30s, **desiste do provider e passa ao próximo**
   da cadeia, em vez de insistir num provedor que só volta horas depois.
3. Só então recorre ao `mock`, e o resumo final avisa quantas chamadas caíram
   nele.

### Empilhando tiers gratuitos

Um único tier não sustenta um ciclo inteiro. Configure vários provedores em
`llm.providers` — cada um com a sua variável de chave, de modo que uma não
sobrescreva a outra — e o cliente usa a cadeia como failover. Veja
`config.free-stack.json`:

```bash
export GEMINI_API_KEY="..."      # 1.500 req/dia, 250 mil TPM
export GROQ_API_KEY="..."        # 8 mil TPM, 200 mil TPD
export MISTRAL_API_KEY="..."     # ~1 req/s, ~1 bi tokens/mês
export OPENROUTER_API_KEY="..."  # 50 req/dia (1.000 após US$ 10)

python main.py --config config.free-stack.json --topic "..." --force
```

Um `--config` parcial é combinado com o `config.json` (merge profundo), então
você não precisa repetir as integrações nem os caminhos.

Os exemplos em `examples/` carregam `config.json`. Para rodá-los com o stack
gratuito, aponte `AI_CIENTISTA_CONFIG` para o outro arquivo:

```bash
AI_CIENTISTA_CONFIG=config.free-stack.json python examples/quick_start.py
```

### Quais tiers realmente funcionam

Limites e condições mudam rápido; confira antes de confiar.

| Provedor | Cartão? | Limite gratuito | Observação |
|---|---|---|---|
| **Google AI Studio** | não | ~1.500 req/dia, 250 mil TPM | Melhor volume diário. Use `gemini-3.5-flash-lite`; `gemini-2.5-flash-lite` já foi retirado. Modelos `pro` saíram do gratuito em abr/2026. |
| **Groq** | não | 8 mil TPM, 200 mil TPD | Rápido, mas estoura o dia em ~3 ciclos. |
| **Mistral** | não | ~1 req/s, ~1 bi tokens/mês | Não publica os números; veja o Admin Console. Uso não comercial. |
| **OpenRouter** | não | 50 req/dia (`:free`) | 1.000 req/dia após US$ 10 uma vez. |
| **GitHub Models** | — | **aposentado** | Encerrado em 30/jul/2026. |
| **Cerebras** | **sim** | US$ 5, expira em 30 dias | O tier permanente acabou em set/2026. |

Recomendação prática: empilhe **Gemini → Groq → Mistral → OpenRouter → mock**.
Com dois ou três desses ativos, o ciclo raramente chega ao `mock`.

**Sobre privacidade:** nos tiers gratuitos, Google e Mistral podem usar seus
prompts para treinar. Para pesquisa com dados sensíveis, use um modelo local
(Ollama, vLLM) apontando `base_url` para `http://localhost:11434/v1`.

Para validar o pipeline sem gastar cota, use `--provider mock`.

---

## Arquitetura

```
core/
  config.py                 Configuração + overrides de ambiente
  memory.py                 Memória persistente (SQLite)
  utils.py                  Logging, JSON, slug, extração de JSON
  prompts.py                Exemplos few-shot (revisão e hipóteses)
  doctor.py                 Diagnóstico do ambiente (--doctor)
  batch.py                  Execução paralela de vários tópicos
  orchestrator/             Coordenação do ciclo em 8 fases
  agents/                   planejador, experimentalista, revisor,
                            redator, validador, integrador de dados
  review/                   Painel de N revisores + meta-revisor
                            fidelidade numérica (numeros.py)
  experiments/              Datasets reais (InfoDengue) + execução e estatística
  paper/                    Formatação ABNT + PDF + LaTeX
integrations/               Crossref, SciELO, OpenAlex, Banco Central,
                            IBGE, InfoDengue, Lattes
llm_providers/              mock, openai_compatible, selenium (bypass/)
examples/                   Três ciclos completos executáveis
tests/                      Suíte pytest (offline + marcados como network)
```

Pipeline (`Orchestrator.run_research_cycle`):

1. Coleta de evidências abertas (Crossref + SciELO + OpenAlex + BCB + IBGE).
2. Identificação de lacuna.
3. Geração de hipóteses testáveis (com exemplos few-shot).
4. Validação de novidade das hipóteses.
5. Planejamento experimental.
6. Execução real do experimento + análise estatística.
7. Redação iterativa com parecer crítico.
8. Validação de qualidade, painel de revisão, formatação ABNT/LaTeX e
   gravação de artefatos.

Cada ciclo grava, em `outputs/ciclo_<id>_<slug>/`: `paper.md`, `paper.pdf`
(ABNT), `paper.tex`, `paper_latex.pdf`, `experimento_H1.json` e
`ciclo_completo.json`.

---

## Testes

```bash
pytest -q -m "not network"   # rápido, offline, determinístico
pytest -q                    # inclui os testes que acessam APIs públicas
```

Os testes exercitam código real: SQLite em disco, treino de modelos,
estatística, geração de PDF, LaTeX, painel de revisão, execução em lote e um
servidor HTTP local que valida o cliente compatível com OpenAI.

---

## Uso responsável

- Divulgue sempre que o texto foi gerado com auxílio de IA.
- Revise e valide todos os resultados antes de publicar.
- Respeite os termos de serviço de qualquer serviço externo utilizado.
- Não use para plágio ou fraude acadêmica.

## Licença

MIT. Veja [LICENSE](LICENSE).

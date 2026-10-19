# AGENTS.md

Contexto para agentes que trabalharem neste repositório.

## O que é

Sistema multi-agente de pesquisa científica autônoma em português do Brasil.
Coleta evidências em APIs abertas reais, gera hipóteses, executa experimentos de
verdade com estatística e escreve um artigo em ABNT (Markdown, PDF e LaTeX).

## Comandos essenciais

```bash
python main.py --doctor                    # diagnóstico do ambiente, sem executar o pipeline
python main.py --topic "..."               # ciclo completo (provider mock por padrão)
python main.py --topic "..." --collect-only
python main.py --topics-file arquivo.txt --parallel 3
python main.py --memory-stats
pytest -q -m "not network"                 # rápido, offline, determinístico
pytest -q                                  # inclui testes que consultam APIs públicas
```

## Convenções

- Todo o código, comentários, docstrings e saídas ficam em **português do Brasil**.
- O provider padrão é `mock`: offline, determinístico, custo zero. Nunca assuma
  acesso a um LLM real nos testes.
- Testes que dependem de internet levam o marcador `@pytest.mark.network`.
- Números no artigo devem vir sempre da execução real de código, nunca de texto
  gerado pelo LLM. Essa separação é um princípio do projeto, não um detalhe.
- Não invente dados de APIs que não existem. Quando uma fonte não tem API pública
  (ex.: Lattes), o sistema declara a limitação em vez de simular.

## Arquitetura

- `core/orchestrator/` — coordena o ciclo em 8 fases.
- `core/agents/` — planejador, experimentalista, revisor, redator, validador,
  integrador de dados. `core/agents/base.py` tem o contrato JSON com retentativa.
- `core/review/panel.py` — painel de N revisores + meta-revisor + voto majoritário.
- `core/experiments/` — execução real (sklearn/scipy) e estatística.
- `core/paper/` — ABNT (`abnt.py`) e LaTeX (`latex.py`).
- `integrations/brazilian_apis.py` — Crossref, SciELO, OpenAlex, BCB, IBGE, Lattes.
- `llm_providers/` — mock, openai_compatible, selenium. O cache SQLite é
  compartilhado entre threads do painel de revisão; precisa continuar
  thread-safe (`check_same_thread=False` + lock).

## Técnicas portadas do AI Scientist (Sakana AI)

Ver a seção "Comparação com o AI Scientist" no README. Ao mexer em JSON de
agentes, painel de revisão, OpenAlex, tokens, doctor, lote ou LaTeX, mantenha o
comportamento documentado lá.

## Armadilhas conhecidas

- **Ambiente efêmero**: `/home/openhands/.local` (onde o `pip` instala por padrão)
  é apagado quando o runtime fica ocioso, e o TeX Live instalado via `apt` também
  se perde. `/workspace/project` é persistente. Por isso, use sempre um venv
  dentro do workspace: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
  Sem isso, o ciclo falha com `ModuleNotFoundError: No module named 'numpy'`.
- Depois de um reset, `apt` fica sem índices: rode `sudo apt-get update` antes de
  instalar o TeX Live, senão o `apt-get install` falha com "Unable to locate package".
- `pdflatex` pode não estar instalado; o código deve degradar para gerar apenas
  `paper.tex` sem falhar. Os dois testes de compilação usam `pytest.skip` nesse caso.
- O token de sistema `GITHUB_TOKEN` é somente leitura em alguns repositórios; para
  push/PR use o token do usuário (`GH_TOKEN`).
- `outputs/`, `*.sqlite3` e `*.pdf` estão no `.gitignore` — artefatos de execução
  não vão para o controle de versão.
- `pypdf` é apenas ferramenta de verificação, não é dependência do projeto.

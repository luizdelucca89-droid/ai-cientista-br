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
streamlit run app.py                       # interface gráfica (abas Nova pesquisa / Histórico / Config)
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
- **Zero dados sintéticos.** Nenhuma matriz de atributos é gerada ou simulada.
  `core/experiments/datasets.py` só devolve observações públicas reais, com
  procedência em `meta` (`tipo: dados_reais`, `origem`, `fonte_url`, `coletado_em`).
  Sem fonte para o tópico, `ExperimentRunner.run` levanta `RealDataUnavailableError`.
- Não invente dados de APIs que não existem. Quando uma fonte não tem API pública
  (ex.: Lattes) ou exige credencial (ex.: Portal da Transparência), o sistema
  declara a limitação em vez de simular.

## Dados reais (fontes verificadas)

- **InfoDengue** `https://info.dengue.mat.br/api/alertcity` — dengue (surto na semana seguinte).
- **INPE/PRODES** `https://terrabrasilis.dpi.inpe.br/geoserver/ows` (WFS) — desmatamento por UF.
  Use `/geoserver/...`; `api/geoserver` responde 404.
- **IBGE/SIDRA** `https://apisidra.ibge.gov.br/values` — t.1612 (PAM, agricultura), t.9544 (ODS, educação).
  Valores ausentes vêm como `..`, `...` ou `-`: filtre antes de converter.
- **BDTD** `https://bdtd.ibict.br/vufind/api/v1/search` (VuFind) — teses/dissertações.
- **Dataverse FGV** `https://dataverse.fgv.br/api/v1/search` — metadados (Dataverse USP não resolve DNS).
- **DATASUS** FTP `ftp.datasus.gov.br` `/dissemin/publicos/...` — microdados `.dbc`
  (`pyreaddbc`+`dbfread`). O TabNet é CGI e devolve HTML, não dados brutos.
- **Kaggle** `/api/v1/datasets/list` — só metadados; download exige credencial.

### Cache e replay offline

`data/dataset_cache/*.json` guarda as respostas cruas das APIs (observações reais, não regeneradas).
`ExperimentRunner(offline=True, cache_dir=...)` reproduz o experimento sem rede — é como os testes
offline exercitam o caminho real. `_load_records` procura primeiro no `cache_dir` da execução e
depois em `data/dataset_cache`. Ao trocar as observações, regrave o cache (rodada online).

Roteamento de tópico → dataset: dengue → InfoDengue; agricultura (irrigação/soja/agro) → SIDRA PAM;
desmatamento (Amazônia/cerrado/bioma) → PRODES; educação (escola/ensino/aprendizado) → SIDRA ODS.
A ordem importa: agricultura vem antes de desmatamento porque "irrigação no Cerrado" casa com ambos.

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

## Escopo dos dados: experimento real quando há fonte pública

As integrações (Crossref, SciELO, OpenAlex, BCB, IBGE, InfoDengue) alimentam as
fases de **evidências** e de **experimento**. `core/experiments/runner.py`
consulta `core/experiments/datasets.py::build_real_dataset(topic)` antes de
treinar. Se houver fonte pública para o tema, o experimento roda sobre
**observações reais**: dengue usa o InfoDengue (Fiocruz/FGV), com casos e clima
por semana epidemiológica; desmatamento usa o PRODES (INPE); agricultura e
educação usam o SIDRA (IBGE). O corte é sempre **temporal** (treino no passado,
teste no futuro) e as linhas de base são de **persistência** (séries) ou de
**tendência** (painéis anuais).

Sem fonte pública para o tema, o runner levanta `RealDataUnavailableError`. Não
existe gerador sintético: `generate_synthetic_dataset` foi removido. Nenhum
resultado é produzido a partir de dados fabricados.

Depois de validar, o orquestrador roda uma conferência determinística de
**fidelidade numérica** (`core/review/numeros.py`): cada número do manuscrito é
comparado com os dados produzidos, e os que não têm lastro são reportados. Isso
pegou um erro real — um `200201` onde o dado dizia `202201`.

## Interface gráfica e publicação

- `app.py` — UI Streamlit (abas Nova pesquisa / Histórico / Configurações).
  Dispara o mesmo `Orchestrator`; não contorna a regra de zero dados sintéticos.
  Rodar com `streamlit run app.py`. Testada no CI apenas com `py_compile`
  (a UI precisa de servidor), por isso a checagem é de sintaxe.
- `integrations/publishing.py` — `generate_submission_package(cycle_dir)` monta
  o ZIP de submissão (paper.pdf, paper.tex, bibliografia.bib,
  codigo_experimento.py, metadata_dados.json, INSTRUCOES_SUBMISSAO.md).
  Levanta `IntegrityError` se `dataset.tipo != "dados_reais"` ou se houver termo
  sintético nos metadados do **bloco `experimento`** (a prosa das seções não é
  varrida: um artigo pode citar "dados sintéticos" ao discutir a literatura).
  A comparação normaliza acentos (NFKD), então `sintéticos` casa com `sintetic`.



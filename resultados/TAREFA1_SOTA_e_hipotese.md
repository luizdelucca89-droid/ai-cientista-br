# TAREFA 1 — Mineração de SOTA e Hipótese Formal

**Tema:** Predição de surtos de dengue no Brasil usando aprendizado de máquina e dados multimodais
**Data da coleta:** 2026-10-19
**Fontes consultadas:** Crossref (API REST), OpenAlex (API REST, abstracts via índice invertido), SciELO (ArticleMeta)
**Artefatos brutos:** `sota_mineracao_bruta.json`, `sota_openalex_abstracts.json`, `sota_scielo.json`, `referencias_verificadas.json`

Todas as referências deste documento foram verificadas uma a uma contra a API do
Crossref (`/workspaces/{DOI}`). Nenhum DOI foi inferido, completado ou escrito de
memória: os 30 DOIs listados retornaram HTTP 200 e metadados reais.

---

## 1. Estado da arte (SOTA) — métricas extraídas de abstracts reais

### 1.1 Predição de surtos de dengue no Brasil

| Referência | Dados | Métrica reportada (texto original) |
|---|---|---|
| Souza et al. (2022), *Expert Systems with Applications* — `10.1016/j.eswa.2021.116324` | clima, grandes centros urbanos BR | "improved earlier prediction accuracy scores **from 0.72 to 0.80**" |
| Sebastianelli et al. (2024), *Scientific Reports* — `10.1038/s41598-024-52796-9` | ensemble ML reprodutível | (sem métrica no abstract) |
| Francisco et al. (2024), *PLoS NTD* — `10.1371/journal.pntd.0012599` | dados zero-inflados | "improves accuracy of dengue …" |
| Martinez & Silva (2011), *Rev. Soc. Bras. Med. Trop.* — `10.1590/s0037-86822011000400007` | SARIMA, Campinas | modelo estatístico clássico |

**SOTA de referência adotado:** Souza et al. (2022) reporta **acurácia 0,80**
(contra 0,72 do baseline anterior) usando **apenas dados climáticos**, em grandes
centros urbanos brasileiros. É a referência brasileira mais diretamente comparável.

### 1.2 Dengue + aprendizado de máquina (contexto internacional)

| Referência | Métrica reportada (texto original) |
|---|---|
| Salim et al. (2021), *Scientific Reports* — `10.1038/s41598-020-79193-2` | "Accuracy = 70%, Sensitivity = 14%, Specificity = 95%, Precision = 56%" |
| Yavari Nejad & Varathan (2021), *BMC Med. Inform. Decis. Mak.* — `10.1186/s12911-021-01493-y` | "increased accuracy to **92.35%**" (Bayes Network) |
| Iqbal et al. (2019), *Informatica* — `10.31449/inf.v43i3.1548` | "accuracy of **92%** with sensitivity and specificity of 90 and 94%" |
| Thirugnanam et al. (2023), *Rev. d'Intelligence Artificielle* — `10.18280/ria.370521` | "F1 Score of **98**" |
| Kerdprasop et al. (2020), *Adv. Sci. Technol. Eng. Syst.* — `10.25046/aj050440` | comparação CI vs. aprendizado estatístico |

### 1.3 Dengue + sensoriamento remoto (o nicho da nossa hipótese)

| Referência | Relevância |
|---|---|
| Singh et al. (2026), *Science of Remote Sensing* — `10.1016/j.srs.2026.100490` | predição de risco por ecorregião usando **sensoriamento remoto + ML** (Peru) |
| *(sem autor no Crossref)* (2018), *IJMLC* — `10.18178/ijmlc.2018.8.4.718` | **dados de sensoriamento remoto** para estimar surto via rede bayesiana |
| Brum-Bastos et al. (2020), *Ecological Informatics* — `10.1016/j.ecoinf.2020.101149` | fusão de múltiplas fontes de **imagens de satélite ópticas** |
| Li et al. (2022), *Remote Sensing* — `10.3390/rs14195052` | big geospatial data para dengue urbano |
| Fuller et al. (2010), *Int. J. Remote Sensing* — `10.1080/01431160902865756` | habitats larvais de *Aedes aegypti* em ambiente urbano |

**Observação crítica (lacuna identificada):** a literatura de dengue + satélite é
dominada por estudos fora do Brasil (Peru, Malásia, Taiwan). Não foi localizado, nas
três bases consultadas, um estudo brasileiro que compare **explicitamente** o ganho
marginal de features de satélite contra um baseline tabular de InfoDengue com
**ablação** e **validação estritamente temporal**. É exatamente essa lacuna que a
hipótese abaixo endereça.

---

## 2. Hipótese estatística formal

Seja $\Delta F1$ a diferença de F1-Score entre o modelo proposto (B) e o baseline (A),
medida sob validação estritamente temporal.

**H0 (nula):** A adição de features de satélite **não** melhora significativamente a
predição.
$$H_0: \Delta F1 \leq 0{,}02 \quad \text{ou} \quad p \geq 0{,}05$$

**H1 (alternativa):** A adição de features de satélite **melhora** significativamente a
predição.
$$H_1: \Delta F1 > 0{,}02 \quad \text{e} \quad p < 0{,}05$$

**Testes planejados (pré-registrados):**
- **B vs A** — efeito conjunto de (algoritmo + satélite)
- **B vs C** — efeito **isolado** do satélite (mesmo algoritmo nos dois lados)
- Wilcoxon pareado (não-paramétrico, pareado por dobra temporal)
- IC 95% por bootstrap com 1000 iterações
- Cohen's $d$ para tamanho de efeito

**Limiar de significância prática:** $\Delta F1 = 0{,}02$. Justificativa: ganhos abaixo
disso não alteram a decisão operacional de um sistema de alerta em nível municipal.

---

## 3. Regra de ouro (pré-registrada)

> Se os dados reais refutarem H1, o resultado nulo será reportado honestamente.
> Resultados negativos rigorosos são contribuições científicas válidas.

Consequências operacionais, definidas **antes** de ver os resultados:

1. Nenhum ajuste de hiperparâmetro será feito para "melhorar a história".
2. A semente é fixa (`seed=42`) e o desenho é temporal.
3. Se o satélite não ajudar, o paper reportará isso como achado principal.
4. Se o satélite **atrapalhar**, isso também será reportado.
5. A análise de sensibilidade (removendo atributos de nível de alerta) será reportada
   mesmo que contrarie a narrativa favorável.

---

## 4. Referências verificadas

30 DOIs verificados contra o Crossref. Lista completa em
`resultados/referencias_verificadas.json` (campos: `doi`, `titulo`, `primeiro_autor`,
`n_autores`, `ano`, `periodico`, `editora`, `tipo`, `verificado_crossref`).

Mínimo exigido pela TAREFA 3: 15 referências. Entregue: **30**.

# RELATÓRIO ELITE — AI Cientista BR

**Tema:** NDVI de satélite na predição de surtos de dengue no Brasil.
**Data:** 2026-10-20 23:55 UTC
**Veredito:** **APROVADO**

## Checklist rigoroso

- [x] Portão de fidelidade numérica: 0 números sem lastro? — **Sim** (0 números sem lastro)
- [x] Validação estritamente temporal (sem data leakage)? — **Sim** (temporal (TimeSeriesSplit expansivo + holdout cronológico))
- [x] Citações cruzadas com metadados reais (sem DOI alucinado)? — **Sim** (16 citações, 0 DOIs não verificados)
- [x] Estudo de ablação e interpretação SHAP incluídos? — **Sim** (ablação=True, SHAP=True)
- [x] Pacote ZIP contém os 5 artefatos? — **Sim** (paper (tex+pdf)=ok, codigo_experimento.py=ok, requirements_frozen.txt=ok, metadata_dados.json=ok, DECLARACAO_IA_E_REPRODUTIBILIDADE.md=ok)
- [x] Seção de Broader Impact e viés de dados incluída? — **Sim** (seção presente=True)

## Resultado científico (honesto)

| Cenário | F1 | AUC | IC 95% F1 |
|---|---|---|---|
| A (baseline, LogReg tabular) | 0.9434 | 0.9930 | [0.8992; 0.9762] |
| B (proposta, GB + NDVI) | 0.9615 | 0.9956 | [0.9254; 0.9877] |
| C (ablação, GB tabular) | 0.9554 | 0.9962 | [0.9172; 0.9873] |

- **ΔF1 (B vs A):** +0.0177 (IC 95% [-0.0267; +0.0674])
- **ΔF1 satélite isolado (B vs C):** +0.0049
- **p-valor (Wilcoxon, B vs A):** 0.0000
- **Limiar pré-registrado:** ΔF1 > 0.02 e p < 0.05
- **Decisão:** não rejeitar H0 (resultado nulo)

**Leitura honesta:** o NDVI não melhorou a predição de forma confiável. O ganho
aparente de B sobre A (+0.0177) vem da troca de algoritmo
(regressão logística → Gradient Boosting), não do satélite — quando se isola o
satélite (B vs C) o efeito é +0.0049. Na análise de
sensibilidade, sem os atributos de nível de alerta, o satélite passa a
**atrapalhar** (ΔF1 = -0.0483).

## Nota estatística reportada no paper
O IC 95% do ΔF1 inclui zero, então a direção do efeito não está firmemente estabelecida, apesar do p-valor de Wilcoxon. Wilcoxon testa a mediana das diferenças pareadas (sensível ao sinal consistente); o IC bootstrap testa a magnitude média. Com efeito pequeno e variância alta os dois divergem, e o resultado prático é de melhora não confiável.

## Evidências
- Portão de fidelidade: `resultados/resultado_consolidado.json` (fontes) vs `paper/paper_sbc.tex`.
- Roteador MoE (log real): `resultados/log_roteador_moe.txt`.
- Referências verificadas no Crossref: `resultados/referencias_crossref.json`.
- Pacote reprodutível: `pacote_open_science.zip`.

## Roteador MoE (verificado por log)
- `task=experiment_plan`, `task=auto_debug` → **mistral/codestral-latest** (código).
- `task=section` → **gemini-flash-latest** (redação).
- `task=voting` → **mistral/ministral-8b-latest** (alto volume).
- `task=gap`, `task=review` → **nvidia/nemotron-3-ultra-550b-a55b** (raciocínio).
- `task=satellite_features` → **nvidia/nemotron-3.5-super-vl:free** (multimodal).

## Reprodutibilidade
```
python experimentos/dengue_ndvi.py          # dados reais, ~6 min
python experimentos/gerar_paper.py --compilar
python experimentos/empacotar.py
python experimentos/relatorio_elite.py
```
Custo: R$ 0,00. Seed: 42. Zero dados sintéticos.

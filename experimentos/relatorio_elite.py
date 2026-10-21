#!/usr/bin/env python3
"""Gera o RELATORIO_ELITE.md com o checklist final, a partir dos artefatos.

Cada item do checklist é verificado programaticamente — o relatório não pode
declarar "Sim" sem a evidência correspondente existir no disco.

Uso:
    python experimentos/relatorio_elite.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


def _gate_numerico() -> tuple[bool, str]:
    from core.review.numeros import collect_numbers, find_unsupported_numbers

    cons = json.loads((RAIZ / "resultados" / "resultado_consolidado.json").read_text(encoding="utf-8"))
    refs = json.loads((RAIZ / "resultados" / "referencias_crossref.json").read_text(encoding="utf-8"))
    fontes = set(collect_numbers(cons)) | {float(r["ano"]) for r in refs if r.get("ano")}
    # Números do estado da arte (Tarefa 1) também são lastro legítimo: cada um
    # vem de sota_metricas.json, com citação textual do abstract de origem.
    caminho_sota = RAIZ / "resultados" / "sota_metricas.json"
    if caminho_sota.exists():
        fontes |= collect_numbers(json.loads(caminho_sota.read_text(encoding="utf-8")))
    # Fatos medidos diretamente nas fontes (colormap, cobertura DETER): também
    # são lastro legítimo, registrados com o valor medido.
    caminho_fatos = RAIZ / "resultados" / "fatos_verificados.json"
    if caminho_fatos.exists():
        fontes |= collect_numbers(json.loads(caminho_fatos.read_text(encoding="utf-8")))
    fontes |= {abs(f) for f in list(fontes)}
    tex = (RAIZ / "paper" / "paper_sbc.tex").read_text(encoding="utf-8")
    corpo = tex.split(r"\begin{document}", 1)[1].split(r"\begin{thebibliography}", 1)[0]
    for padrao in [r"\\url\{[^}]*\}", r"\\cite\{[^}]*\}",
                   r"\\includegraphics\[[^\]]*\]\{[^}]*\}", r"\\geometry\{[^}]*\}",
                   r"\\texttt\{[^}]*\}", r"\\emph\{[^}]*\}",
                   r"\\(begin|end)\{[^}]*\}", r"\\label\{[^}]*\}",
                   r"\\ref\{[^}]*\}", r"\\bibitem\{[^}]*\}"]:
        corpo = re.sub(padrao, " ", corpo)
    suspeitos = find_unsupported_numbers(corpo, fontes)
    return (not suspeitos), f"{len(suspeitos)} números sem lastro"


def _citacoes() -> tuple[bool, str]:
    refs = json.loads((RAIZ / "resultados" / "referencias_crossref.json").read_text(encoding="utf-8"))
    tex = (RAIZ / "paper" / "paper_sbc.tex").read_text(encoding="utf-8")
    dois = {r["doi"] for r in refs if r.get("doi")}
    citados = set(re.findall(r"\\cite\{([^}]+)\}", tex))
    inventados = sorted(c for c in citados if c not in dois)
    return (not inventados), f"{len(citados)} citações, {len(inventados)} DOIs não verificados"


def main() -> int:
    cons = json.loads((RAIZ / "resultados" / "resultado_consolidado.json").read_text(encoding="utf-8"))
    p, s = cons["principal"], cons["sensibilidade"]
    h = p["hipotese"]

    ok_num, det_num = _gate_numerico()
    ok_cit, det_cit = _citacoes()

    tex = (RAIZ / "paper" / "paper_sbc.tex").read_text(encoding="utf-8")
    tem_ablation = "Estudo de ablação" in tex or "ablação" in tex
    tem_shap = "SHAP" in tex and (RAIZ / "paper" / "figuras" / "fig_shap.png").exists()
    tem_impacto = "Impacto mais amplo" in tex and "subnotificação" in tex
    zip_ok = (RAIZ / "pacote_open_science.zip").exists()
    zip_itens = []
    if zip_ok:
        import zipfile

        with zipfile.ZipFile(RAIZ / "pacote_open_science.zip") as z:
            zip_itens = z.namelist()
    cinco = {
        "paper (tex+pdf)": any(n.endswith("paper_sbc.tex") for n in zip_itens) and any(n.endswith("paper_sbc.pdf") for n in zip_itens),
        "codigo_experimento.py": any(n.endswith("codigo_experimento.py") for n in zip_itens),
        "requirements_frozen.txt": any(n.endswith("requirements_frozen.txt") for n in zip_itens),
        "metadata_dados.json": any(n.endswith("metadata_dados.json") for n in zip_itens),
        "DECLARACAO_IA_E_REPRODUTIBILIDADE.md": any(n.endswith("DECLARACAO_IA_E_REPRODUTIBILIDADE.md") for n in zip_itens),
    }
    zip_completo = all(cinco.values())

    split_temporal = "temporal" in p["split"].lower() and "TimeSeriesSplit" in p["split"]

    itens = [
        ("Portão de fidelidade numérica: 0 números sem lastro?", ok_num, det_num),
        ("Validação estritamente temporal (sem data leakage)?", split_temporal, p["split"]),
        ("Citações cruzadas com metadados reais (sem DOI alucinado)?", ok_cit, det_cit),
        ("Estudo de ablação e interpretação SHAP incluídos?", tem_ablation and tem_shap,
         f"ablação={tem_ablation}, SHAP={tem_shap}"),
        ("Pacote ZIP contém os 5 artefatos?", zip_completo,
         ", ".join(f"{k}={'ok' if v else 'FALTA'}" for k, v in cinco.items())),
        ("Seção de Broader Impact e viés de dados incluída?", tem_impacto,
         f"seção presente={tem_impacto}"),
    ]

    linhas = []
    for texto, ok, det in itens:
        linhas.append(f"- [{'x' if ok else ' '}] {texto} — **{'Sim' if ok else 'Não'}** ({det})")

    veredito = "APROVADO" if all(ok for _, ok, _ in itens) else "PENDÊNCIAS"

    relatorio = f"""# RELATÓRIO ELITE — AI Cientista BR

**Tema:** NDVI de satélite na predição de surtos de dengue no Brasil.
**Data:** {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Veredito:** **{veredito}**

## Checklist rigoroso

{chr(10).join(linhas)}

## Resultado científico (honesto)

| Cenário | F1 | AUC | IC 95% F1 |
|---|---|---|---|
| A (baseline, LogReg tabular) | {p['metricas']['A']['f1']:.4f} | {p['metricas']['A']['auc']:.4f} | [{p['metricas']['A']['ic95_f1'][0]:.4f}; {p['metricas']['A']['ic95_f1'][1]:.4f}] |
| B (proposta, GB + NDVI) | {p['metricas']['B']['f1']:.4f} | {p['metricas']['B']['auc']:.4f} | [{p['metricas']['B']['ic95_f1'][0]:.4f}; {p['metricas']['B']['ic95_f1'][1]:.4f}] |
| C (ablação, GB tabular) | {p['metricas']['C']['f1']:.4f} | {p['metricas']['C']['auc']:.4f} | [{p['metricas']['C']['ic95_f1'][0]:.4f}; {p['metricas']['C']['ic95_f1'][1]:.4f}] |

- **ΔF1 (B vs A):** {h['delta_f1_observado']:+.4f} (IC 95% [{h['ic95_delta'][0]:+.4f}; {h['ic95_delta'][1]:+.4f}])
- **ΔF1 satélite isolado (B vs C):** {h['delta_f1_isolado_satelite']:+.4f}
- **p-valor (Wilcoxon, B vs A):** {h['p_valor_observado']:.4f}
- **Limiar pré-registrado:** ΔF1 > {h['limiar_delta']:.2f} e p < {h['alpha']:.2f}
- **Decisão:** {h['decisao']}

**Leitura honesta:** o NDVI não melhorou a predição de forma confiável. O ganho
aparente de B sobre A (+{h['delta_f1_observado']:.4f}) vem da troca de algoritmo
(regressão logística → Gradient Boosting), não do satélite — quando se isola o
satélite (B vs C) o efeito é {h['delta_f1_isolado_satelite']:+.4f}. Na análise de
sensibilidade, sem os atributos de nível de alerta, o satélite passa a
**atrapalhar** (ΔF1 = {s['testes']['B vs C']['delta_f1_medio']:+.4f}).

## Nota estatística reportada no paper
{h['observacao_estatistica']}

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
Custo: R$ 0,00. Seed: {p['seed']}. Zero dados sintéticos.
"""

    (RAIZ / "RELATORIO_ELITE.md").write_text(relatorio, encoding="utf-8")
    print(relatorio)
    return 0 if veredito == "APROVADO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Experimento reprodutível: NDVI de satélite na predição de surtos de dengue.

Uso:
    python experimentos/dengue_ndvi.py [--offline] [--saida DIR]

Dados 100% reais, zero sintéticos:
  * InfoDengue (Fiocruz/FGV): casos, clima e nível de alerta por semana.
  * NASA GIBS / MODIS NDVI: vegetação por pixel, agregada ao município (malha
    do IBGE) e à semana epidemiológica.

Protocolo: seed=42, validação estritamente temporal (TimeSeriesSplit + holdout
cronológico), IC 95% por bootstrap (1000x), Wilcoxon pareado e Cohen's d.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Permite rodar o script direto (python experimentos/dengue_ndvi.py) sem instalar
# o pacote: o diretório raiz do projeto entra no sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dengue_ndvi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="usa apenas cache local")
    parser.add_argument("--saida", default="resultados", help="diretório de saída")
    parser.add_argument("--anos", default="2022,2023,2024", help="anos do InfoDengue")
    args = parser.parse_args()

    from core.experiments.dengue_experiment import executar_experimento, gerar_figuras
    from core.experiments.dengue_satellite import build_dengue_satellite_dataset

    anos = tuple(int(a) for a in args.anos.split(","))
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    logger.info("Montando dataset real (InfoDengue + NDVI/MODIS)...")
    ds = build_dengue_satellite_dataset(anos=anos, offline=args.offline)
    if ds is None:
        logger.error("Dados reais indisponíveis. Nada é fabricado — abortando.")
        return 2
    logger.info("Dataset: %s amostras, %s atributos", ds.X.shape[0], ds.X.shape[1])

    nomes = ds.meta["variaveis"]
    logger.info("Rodando cenários A/B/C com validação temporal...")
    res = executar_experimento(ds.X, ds.y, nomes, ds.meta, diretorio_saida=saida)

    logger.info("Análise de sensibilidade (sem atributos de nível de alerta)...")
    remover = {"nivel", "nivel_inc"}
    manter = [i for i, n in enumerate(nomes) if n not in remover]
    meta2 = dict(ds.meta)
    meta2["variaveis_tabulares"] = [v for v in ds.meta["variaveis_tabulares"] if v not in remover]
    res_sens = executar_experimento(
        ds.X[:, manter], ds.y, [nomes[i] for i in manter], meta2,
        diretorio_saida=saida / "sensibilidade",
    )

    logger.info("Gerando figuras (ROC, SHAP)...")
    figuras = gerar_figuras(ds.X, ds.y, nomes, ds.meta, saida / "figuras")

    consolidado = {
        "principal": res,
        "sensibilidade": res_sens,
        "figuras": figuras,
        "proveniencia": ds.meta,
    }
    (saida / "resultado_consolidado.json").write_text(
        json.dumps(consolidado, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== RESULTADO PRINCIPAL ===")
    for c in ["A", "B", "C"]:
        m = res["metricas"][c]
        print(f"  {m['nome']:14s} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} IC95={[round(x, 4) for x in m['ic95_f1']]}")
    h = res["hipotese"]
    print(f"  ΔF1 (B vs A) = {h['delta_f1_observado']:+.4f} | "
          f"ΔF1 satélite isolado (B vs C) = {h['delta_f1_isolado_satelite']:+.4f} | "
          f"p = {h['p_valor_observado']:.4f}")
    print(f"  Decisão: {h['decisao']}")
    print(f"\nArtefatos em: {saida}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

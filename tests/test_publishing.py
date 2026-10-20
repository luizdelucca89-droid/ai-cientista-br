"""Testes do pipeline de submissão (``integrations.publishing``).

Cobrem a montagem do ZIP, a lista de arquivos obrigatórios e a regra de
integridade que rejeita pacotes cujos metadados não atestem dados 100% reais.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from integrations.publishing import IntegrityError, generate_submission_package

ARQUIVOS_OBRIGATORIOS = {
    "paper.pdf",
    "paper.tex",
    "bibliografia.bib",
    "codigo_experimento.py",
    "metadata_dados.json",
    "INSTRUCOES_SUBMISSAO.md",
}


def _ciclo_falso(tmp_path: Path, *, tipo: str = "dados_reais", extra: dict | None = None) -> Path:
    """Cria um diretório de ciclo mínimo com os artefatos exigidos."""
    ciclo_dir = tmp_path / "ciclo_1_dengue"
    ciclo_dir.mkdir(parents=True)
    (ciclo_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    (ciclo_dir / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")

    payload = {
        "cycle_id": 1,
        "evidencias": {"artigos": [{"titulo": "Estudo X", "autores": ["Silva, A."], "ano": 2023, "doi": "10.1/x"}]},
        "experimento": {
            "desenho": {"amostra": "observações públicas reais"},
            "resultado": {
                "topic": "dengue",
                "seed": 42,
                "best_model": "floresta_aleatoria",
                "metrics": {"floresta_aleatoria": {"f1_score": 0.89}},
                "dataset": {
                    "tipo": tipo,
                    "origem": "InfoDengue (Fiocruz/FGV)",
                    "fonte_url": "https://info.dengue.mat.br",
                    "coletado_em": "2026-01-01T00:00:00Z",
                    "n_amostras": 817,
                    "n_variaveis": 8,
                    "split": "temporal",
                },
            },
        },
    }
    if extra:
        payload["experimento"]["desenho"].update(extra)
    (ciclo_dir / "ciclo_completo.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return ciclo_dir


def test_generate_submission_package_cria_zip(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path)
    pacote = generate_submission_package(ciclo_dir)

    destino = Path(pacote.path)
    assert destino.exists()
    assert destino.suffix == ".zip"
    assert zipfile.is_zipfile(destino)
    assert pacote.tamanho_bytes > 0
    assert pacote.cycle_id == "1"


def test_zip_contem_arquivos_esperados(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path)
    pacote = generate_submission_package(ciclo_dir)

    with zipfile.ZipFile(pacote.path) as zf:
        nomes = set(zf.namelist())

    assert ARQUIVOS_OBRIGATORIOS.issubset(nomes)


def test_instrucoes_mencionam_sbc_e_scielo(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path)
    pacote = generate_submission_package(ciclo_dir)

    with zipfile.ZipFile(pacote.path) as zf:
        texto = zf.read("INSTRUCOES_SUBMISSAO.md").decode("utf-8")

    assert "SBC" in texto
    assert "SciELO" in texto
    assert "arXiv" in texto


def test_metadata_dados_traz_procedencia(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path)
    pacote = generate_submission_package(ciclo_dir)

    with zipfile.ZipFile(pacote.path) as zf:
        meta = json.loads(zf.read("metadata_dados.json"))

    assert meta["tipo"] == "dados_reais"
    assert meta["fonte_url"] == "https://info.dengue.mat.br"
    assert meta["n_amostras"] == 817


def test_codigo_reproducao_e_python_valido(tmp_path: Path) -> None:
    import ast

    ciclo_dir = _ciclo_falso(tmp_path)
    pacote = generate_submission_package(ciclo_dir)

    with zipfile.ZipFile(pacote.path) as zf:
        codigo = zf.read("codigo_experimento.py").decode("utf-8")

    ast.parse(codigo)
    assert "ExperimentRunner" in codigo
    assert "dengue" in codigo


def test_rejeita_dados_nao_reais(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path, tipo="dados_sinteticos")

    with pytest.raises(IntegrityError, match="100% reais"):
        generate_submission_package(ciclo_dir)


def test_rejeita_metadados_com_termo_sintetico(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path, extra={"nota": "gerado a partir de dados sintéticos"})

    with pytest.raises(IntegrityError, match="100% reais"):
        generate_submission_package(ciclo_dir)


def test_aceita_metodologia_com_simulacao(tmp_path: Path) -> None:
    """Simulação é método legítimo, não dado fabricado.

    O desenho pode descrever "ataques DoS simulados" ou "rede IoT simulada" com
    dados 100% reais; o termo genérico "simulad" rejeitava esses ciclos.
    """
    ciclo_dir = _ciclo_falso(
        tmp_path,
        extra={"desenho": "Experimento com injeção de falhas em rede IoT simulada."},
    )
    pacote = generate_submission_package(ciclo_dir)
    assert Path(pacote.path).exists()


def test_rejeita_sem_paper_pdf(tmp_path: Path) -> None:
    ciclo_dir = _ciclo_falso(tmp_path)
    (ciclo_dir / "paper.pdf").unlink()

    with pytest.raises(IntegrityError, match="obrigatórios"):
        generate_submission_package(ciclo_dir)

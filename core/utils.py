"""Utilidades compartilhadas: logging, serialização e manipulação de texto."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def utcnow_iso() -> str:
    """Retorna o instante atual em UTC no formato ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Garante que o diretório exista e devolve o Path correspondente."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike[str], data: Any) -> Path:
    """Grava um objeto JSON legível em disco."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Lê um arquivo JSON, devolvendo ``default`` quando ausente ou inválido."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def slugify(text: str, max_length: int = 60) -> str:
    """Converte texto livre em um slug seguro para nomes de arquivo."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return (slug[:max_length] or "sem-titulo").strip("-")


def sha256_text(text: str) -> str:
    """Hash estável usado para chaves de cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate(text: str, limit: int = 400, suffix: str = "...") -> str:
    """Encurta um texto preservando o começo."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def extract_json_block(text: str) -> Any:
    """Extrai o primeiro objeto/array JSON de um texto gerado por LLM.

    Aceita blocos cercados por ```json ... ``` e devolve ``None`` quando não
    encontra JSON válido.
    """
    if not text:
        return None
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        start = min(
            (i for i in (candidate.find("{"), candidate.find("[")) if i != -1),
            default=-1,
        )
        if start == -1:
            continue
        for end in range(len(candidate), start, -1):
            chunk = candidate[start:end]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    return None


def setup_logger(name: str = "ai_cientista", level: str | None = None,
                 logfile: str | os.PathLike[str] | None = None) -> logging.Logger:
    """Configura um logger com saída em console e, opcionalmente, em arquivo."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, (level or os.getenv("AI_CIENTISTA_LOG", "INFO")).upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if logfile:
        path = Path(logfile)
        ensure_dir(path.parent)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in logger.handlers):
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
            logger.addHandler(handler)
    return logger

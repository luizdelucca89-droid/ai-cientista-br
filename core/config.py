"""Carregamento e validação da configuração do AI Cientista Brasileiro.

A configuração é lida de ``config.json`` (na raiz do projeto) e pode ser
sobrescrita por variáveis de ambiente, o que permite trocar de provider de LLM
sem editar arquivos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.utils import ROOT, read_json

DEFAULT_CONFIG_PATH = ROOT / "config.json"

ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "AI_CIENTISTA_PROVIDER": ("llm", "provider"),
    "AI_CIENTISTA_MODEL": ("llm", "model"),
    "OPENAI_BASE_URL": ("llm", "base_url"),
    "AI_CIENTISTA_TIMEOUT": ("llm", "timeout"),
    "AI_CIENTISTA_OUTPUT_DIR": ("paths", "output_dir"),
    "AI_CIENTISTA_SEED": ("experiments", "seed"),
}


@dataclass
class Config:
    """Configuração efetiva do sistema, já com overrides aplicados."""

    data: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        data = read_json(cfg_path, default=None)
        if data is None:
            data = read_json(DEFAULT_CONFIG_PATH, default={}) or {}
            cfg_path = DEFAULT_CONFIG_PATH if cfg_path.exists() else None
        config = cls(data=dict(data), source=cfg_path)
        config._apply_env_overrides()
        return config

    def _apply_env_overrides(self) -> None:
        for env_key, path in ENV_OVERRIDES.items():
            value = os.getenv(env_key)
            if value in (None, ""):
                continue
            node = self.data
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value

    # --------------------------------------------------------------- getters
    def get(self, *path: str, default: Any = None) -> Any:
        """Busca um valor aninhado usando uma sequência de chaves."""
        node: Any = self.data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def llm(self) -> dict[str, Any]:
        return dict(self.data.get("llm", {}))

    @property
    def output_dir(self) -> Path:
        raw = self.get("paths", "output_dir", default="outputs")
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path

    @property
    def memory_path(self) -> Path:
        raw = self.get("paths", "memory_db", default="outputs/memory.sqlite3")
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path

    @property
    def seed(self) -> int:
        return int(self.get("experiments", "seed", default=42))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.data)

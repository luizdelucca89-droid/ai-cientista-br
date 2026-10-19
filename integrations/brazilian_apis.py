"""Integrações com fontes de dados abertas e APIs brasileiras.

Fontes implementadas (todas verificadas contra os serviços reais):

==============================  ==============================================
Fonte                           Uso no pipeline
==============================  ==============================================
Crossref (``api.crossref.org``) Metadados de artigos com DOI; o modo
                                ``prefix:10.1590`` isola produção SciELO.
SciELO / ArticleMeta            Produção científica brasileira em acesso aberto.
Banco Central (SGS)             Séries temporais econômicas brasileiras reais.
IBGE (localidades, SIDRA)       Unidades da federação, municípios e indicadores.
Lattes (CNPq)                   Sem API pública — o cliente documenta a
                                limitação e devolve um link de consulta manual.
==============================  ==============================================

Todas as chamadas usam rate limiting, timeout e cache em memória, e degradam
com elegância (retornam ``[]``) quando um serviço está indisponível.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import requests

from core.utils import setup_logger, truncate, utcnow_iso

logger = setup_logger(__name__)

USER_AGENT = "AI-Cientista-Brasileiro/1.0 (+mailto:ai.cientista@example.org)"
SCIELO_DOI_PREFIX = "10.1590"


@dataclass
class RateLimiter:
    """Intervalo mínimo entre chamadas ao mesmo host."""

    min_interval: float = 1.0
    _last: float = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.time()


@dataclass
class BaseAPI:
    """Base para clientes HTTP com rate limit, timeout e cache."""

    base_url: str
    timeout: int = 15
    rate_limit_seconds: float = 1.0
    retries: int = 2
    session: requests.Session = field(default_factory=requests.Session)
    limiter: RateLimiter | None = None
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)
    name: str = "base"

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(self.rate_limit_seconds)
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def _fetch(self, url: str, params: dict[str, Any] | None = None) -> requests.Response | None:
        cache_key = url + "?" + urllib.parse.urlencode(sorted((params or {}).items()))
        if cache_key in self._cache:
            return self._cache[cache_key]
        assert self.limiter is not None
        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        if last_error is not None or response is None:
            logger.warning("%s indisponível após %s tentativas: %s", self.name, self.retries + 1, last_error)
            return None
        if response.status_code != 200:
            logger.warning("%s retornou HTTP %s", self.name, response.status_code)
            return None
        self._cache[cache_key] = response
        return response

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self._fetch(url, params)
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            logger.warning("%s devolveu conteúdo não-JSON", self.name)
            return None


def _clean_markup(raw: str | None, limit: int = 600) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return truncate(text, limit, suffix="")


# --------------------------------------------------------------------------- #
# Crossref
# --------------------------------------------------------------------------- #
@dataclass
class CrossrefAPI(BaseAPI):
    """Metadados de artigos indexados com DOI."""

    base_url: str = "https://api.crossref.org/works"
    mailto: str = "ai.cientista@example.org"
    name: str = "crossref"

    def search(self, query: str, rows: int = 8, prefix: str | None = None) -> list[dict[str, Any]]:
        """Busca artigos; ``prefix`` restringe a um registrante (ex.: 10.1590 = SciELO)."""
        params: dict[str, Any] = {
            "query": query,
            "rows": rows,
            "select": "DOI,title,author,issued,container-title,abstract,URL,type,publisher",
            "mailto": self.mailto,
        }
        if prefix:
            params["filter"] = f"prefix:{prefix}"
        data = self.get_json(self.base_url, params)
        if not data:
            return []
        items = data.get("message", {}).get("items", [])
        return [self._normalize(item) for item in items]

    def search_scielo(self, query: str, rows: int = 8) -> list[dict[str, Any]]:
        """Artigos de periódicos SciELO (prefixo DOI 10.1590)."""
        return self.search(query, rows=rows, prefix=SCIELO_DOI_PREFIX)

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any]:
        authors = []
        for author in item.get("author", []) or []:
            name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
            if name:
                authors.append(name)
        issued = (item.get("issued", {}) or {}).get("date-parts", [[None]])[0]
        year = issued[0] if issued else None
        doi = item.get("DOI")
        return {
            "fonte": "crossref",
            "doi": doi,
            "titulo": (item.get("title") or ["(sem título)"])[0],
            "autores": authors,
            "ano": year,
            "periodico": (item.get("container-title") or [None])[0],
            "tipo": item.get("type"),
            "editora": item.get("publisher"),
            "url": item.get("URL") or (f"https://doi.org/{doi}" if doi else None),
            "resumo": _clean_markup(item.get("abstract")),
            "acesso": "aberto",
        }


# --------------------------------------------------------------------------- #
# OpenAlex
# --------------------------------------------------------------------------- #
@dataclass
class OpenAlexAPI(BaseAPI):
    """OpenAlex: catálogo aberto com contagem de citações.

    Usado pelo AI Scientist v1 como motor alternativo de busca bibliográfica
    (``engine="openalex"``). Aqui serve para *enriquecer* as referências com
    ``citationCount`` e status de acesso aberto, o que ajuda o revisor a
    distinguir trabalhos de impacto de citações incidentais.

    É gratuito e não exige chave; informar ``mailto`` apenas dá acesso ao
    "polite pool", com limites de taxa mais generosos.
    """

    base_url: str = "https://api.openalex.org/works"
    mailto: str = "ai.cientista@example.org"
    name: str = "openalex"

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        data = self.get_json(
            self.base_url,
            {
                "search": query,
                "per-page": max(1, min(limit, 25)),
                "mailto": self.mailto,
                "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count,open_access,type",
            },
        )
        if not data:
            return []
        return [self._normalize(item) for item in data.get("results", [])]

    def citation_count(self, doi: str) -> int | None:
        """Devolve o número de citações de um DOI, ou ``None`` se desconhecido."""
        doi = (doi or "").strip()
        if not doi:
            return None
        clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        data = self.get_json(f"{self.base_url}/doi:{clean}", {"mailto": self.mailto, "select": "cited_by_count"})
        if not data:
            return None
        value = data.get("cited_by_count")
        return int(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any]:
        source = ((item.get("primary_location") or {}).get("source") or {})
        authors = [
            (a.get("author") or {}).get("display_name")
            for a in (item.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ]
        doi = item.get("doi") or ""
        return {
            "fonte": "openalex",
            "doi": doi.replace("https://doi.org/", "") if doi else None,
            "titulo": item.get("title") or "(sem título)",
            "autores": authors,
            "ano": item.get("publication_year"),
            "periodico": source.get("display_name"),
            "tipo": item.get("type"),
            "citacoes": item.get("cited_by_count"),
            "acesso": (item.get("open_access") or {}).get("oa_status") or "desconhecido",
            "url": doi or item.get("id"),
            "resumo": None,
        }


# --------------------------------------------------------------------------- #
# SciELO
# --------------------------------------------------------------------------- #
@dataclass
class SciELOAPI(BaseAPI):
    """Produção científica brasileira (SciELO).

    Estratégia 1: Crossref restrito ao prefixo DOI da SciELO (rápido e estável).
    Estratégia 2: ArticleMeta (``articlemeta.scielo.org``) — metadados completos
    da coleção Brasil, filtrados por termos no título/resumo.
    """

    base_url: str = "https://articlemeta.scielo.org/api/v1"
    crossref: CrossrefAPI | None = None
    name: str = "scielo"

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        results = self.search_via_crossref(query, limit)
        if len(results) >= limit:
            return results
        extra = self.search_via_articlemeta(query, limit - len(results))
        seen = {(a.get("doi") or a.get("titulo") or "").lower() for a in results}
        for article in extra:
            key = (article.get("doi") or article.get("titulo") or "").lower()
            if key and key not in seen:
                seen.add(key)
                results.append(article)
        return results[:limit]

    def search_via_crossref(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        api = self.crossref or CrossrefAPI(rate_limit_seconds=self.rate_limit_seconds, timeout=self.timeout)
        articles = api.search_scielo(query, rows=limit)
        for article in articles:
            article["fonte"] = "scielo/crossref"
            article["colecao"] = "SciELO"
        return articles

    def search_via_articlemeta(self, query: str, limit: int = 4, window: int = 12) -> list[dict[str, Any]]:
        """Varre uma janela recente da coleção Brasil filtrando por termos."""
        terms = [t.lower() for t in re.findall(r"\w{5,}", query)][:4]
        if not terms or limit <= 0:
            return []
        identifiers = self.get_json(
            f"{self.base_url}/article/identifiers/",
            {"collection": "scl", "limit": window, "from": "2024-01-01"},
        )
        if not identifiers:
            return []
        found: list[dict[str, Any]] = []
        for item in identifiers.get("objects", []):
            code = item.get("code")
            if not code:
                continue
            detail = self.get_json(f"{self.base_url}/article/", {"code": code, "collection": "scl"})
            article = self._normalize_article(detail, item)
            if not article:
                continue
            haystack = f"{article['titulo']} {article['resumo']}".lower()
            if any(term in haystack for term in terms):
                found.append(article)
            if len(found) >= limit:
                break
        return found

    @staticmethod
    def _normalize_article(detail: dict[str, Any] | None, identifier: dict[str, Any]) -> dict[str, Any] | None:
        if not detail or "article" not in detail:
            return None
        art = detail["article"]
        titles = art.get("v12") or art.get("v14") or []
        title = next((t.get("_") for t in titles if t.get("_")), None)
        if not title:
            return None
        authors = []
        for author in art.get("v10", []) or []:
            name = " ".join(p for p in [author.get("n"), author.get("s")] if p)
            if name:
                authors.append(name)
        abstract = next((a.get("_") for a in (art.get("v83") or []) if a.get("_")), "")
        year = None
        for key in ("v65", "v223"):
            if art.get(key):
                year = art[key][0].get("_")
                break
        journal = next((j.get("_") for j in (art.get("v30") or []) if j.get("_")), None)
        doi = identifier.get("doi")
        return {
            "fonte": "scielo/articlemeta",
            "colecao": "SciELO Brasil",
            "doi": doi,
            "titulo": title,
            "autores": authors,
            "ano": year,
            "periodico": journal,
            "url": f"https://doi.org/{doi}" if doi else f"https://articlemeta.scielo.org/?code={identifier.get('code')}",
            "resumo": _clean_markup(abstract),
            "acesso": "aberto",
        }


# --------------------------------------------------------------------------- #
# Banco Central do Brasil (SGS)
# --------------------------------------------------------------------------- #
@dataclass
class BancoCentralAPI(BaseAPI):
    """Séries temporais do Sistema Gerenciador de Séries Temporais (SGS)."""

    base_url: str = "https://api.bcb.gov.br/dados/serie"
    name: str = "bcb"

    SERIES = {
        433: "IPCA (variação mensal %)",
        1: "Taxa de câmbio USD/BRL (compra)",
        21084: "Taxa de desocupação (%)",
        4390: "Crédito total ao setor privado (% do PIB)",
        4389: "Taxa de juros - Selic acumulada no mês (%)",
    }

    def ultimos(self, codigo: int, n: int = 6) -> list[dict[str, Any]]:
        data = self.get_json(f"{self.base_url}/bcdata.sgs.{codigo}/dados/ultimos/{n}", {"formato": "json"})
        if not data:
            return []
        return [
            {
                "serie": codigo,
                "descricao": self.SERIES.get(codigo, f"série SGS {codigo}"),
                "data": row.get("data"),
                "valor": row.get("valor"),
            }
            for row in data
        ]

    def painel_brasil(self, series: tuple[int, ...] = (433, 1, 21084), n: int = 3) -> list[dict[str, Any]]:
        """Últimas observações de séries macroeconômicas brasileiras."""
        panel: list[dict[str, Any]] = []
        for codigo in series:
            panel.extend(self.ultimos(codigo, n))
        return panel


# --------------------------------------------------------------------------- #
# IBGE
# --------------------------------------------------------------------------- #
@dataclass
class IBGEAPI(BaseAPI):
    """Localidades, municípios e indicadores agregados do IBGE."""

    base_url: str = "https://servicodados.ibge.gov.br/api"
    name: str = "ibge"

    def estados(self) -> list[dict[str, Any]]:
        data = self.get_json(f"{self.base_url}/v1/localidades/estados") or []
        return [{"id": e.get("id"), "sigla": e.get("sigla"), "nome": e.get("nome")} for e in data]

    def municipios(self, uf_id: int, limite: int = 5) -> list[dict[str, Any]]:
        data = self.get_json(f"{self.base_url}/v1/localidades/estados/{uf_id}/municipios") or []
        return [{"id": m.get("id"), "nome": m.get("nome")} for m in data[:limite]]

    def populacao_estimada(self, ano: int = 2021) -> list[dict[str, Any]]:
        """População residente estimada por UF (SIDRA tabela 6579, variável 9324)."""
        data = self.get_json(f"https://apisidra.ibge.gov.br/values/t/6579/n3/all/v/9324/p/{ano}")
        if not isinstance(data, list) or len(data) < 2:
            return []
        results = []
        for row in data[1:]:  # a primeira linha é o cabeçalho
            results.append(
                {
                    "uf": row.get("D1N"),
                    "uf_codigo": row.get("D1C"),
                    "populacao": row.get("V"),
                    "ano": row.get("D3N"),
                    "fonte": "IBGE/SIDRA 6579",
                }
            )
        return results

    def contexto_brasil(self, limite_ufs: int = 5) -> dict[str, Any]:
        """Contexto territorial real usado como base regional dos estudos."""
        todos = self.estados()
        estados = todos[:limite_ufs]
        populacao = {p["uf"]: p["populacao"] for p in self.populacao_estimada()}
        for estado in estados:
            estado["populacao_estimada"] = populacao.get(estado["nome"])
        return {"unidades_federacao": estados, "total_ufs_disponiveis": len(todos)}


# --------------------------------------------------------------------------- #
# Lattes (sem API pública)
# --------------------------------------------------------------------------- #
@dataclass
class LattesAPI(BaseAPI):
    """Currículos Lattes (CNPq).

    A Plataforma Lattes **não** oferece API pública de consulta programática: o
    acesso em massa exige o serviço oficial *Lattes Extrator*, restrito a
    instituições. Este cliente não simula dados — devolve explicitamente a
    limitação e um link de consulta manual.
    """

    base_url: str = "https://buscatextual.cnpq.br/buscatextual/busca.do"
    name: str = "lattes"

    def search(self, query: str, limit: int = 1) -> list[dict[str, Any]]:
        url = f"{self.base_url}?metodo=forwardToSearch&textoBusca={urllib.parse.quote(query)}"
        logger.info("Lattes não possui API pública; consulta manual: %s", url)
        return [
            {
                "fonte": "lattes",
                "disponivel_via_api": False,
                "motivo": "A Plataforma Lattes não expõe API pública de consulta programática (requer Lattes Extrator).",
                "consulta_manual": url,
                "consulta_origem": query,
            }
        ]


# --------------------------------------------------------------------------- #
# Agregador
# --------------------------------------------------------------------------- #
@dataclass
class BrazilianDataAggregator:
    """Agrega Crossref, SciELO, Banco Central, IBGE e Lattes em um só contexto."""

    crossref: CrossrefAPI = field(default_factory=CrossrefAPI)
    scielo: SciELOAPI = field(default_factory=SciELOAPI)
    openalex: OpenAlexAPI = field(default_factory=OpenAlexAPI)
    bcb: BancoCentralAPI = field(default_factory=BancoCentralAPI)
    ibge: IBGEAPI = field(default_factory=IBGEAPI)
    lattes: LattesAPI = field(default_factory=LattesAPI)
    max_sources: int = 12
    offline: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BrazilianDataAggregator":
        integ = config.get("integrations", {})
        agents = config.get("agents", {})
        cr_cfg = integ.get("crossref", {})
        sc_cfg = integ.get("scielo", {})
        ibge_cfg = integ.get("ibge", {})
        bcb_cfg = integ.get("bcb", {})
        crossref = CrossrefAPI(
            base_url=cr_cfg.get("base_url", "https://api.crossref.org/works"),
            timeout=cr_cfg.get("timeout", 15),
            rate_limit_seconds=cr_cfg.get("rate_limit_seconds", 1.0),
            mailto=cr_cfg.get("mailto", "ai.cientista@example.org"),
        )
        return cls(
            crossref=crossref,
            scielo=SciELOAPI(
                base_url=sc_cfg.get("base_url", "https://articlemeta.scielo.org/api/v1"),
                timeout=sc_cfg.get("timeout", 15),
                rate_limit_seconds=sc_cfg.get("rate_limit_seconds", 1.0),
                crossref=crossref,
            ),
            bcb=BancoCentralAPI(
                base_url=bcb_cfg.get("base_url", "https://api.bcb.gov.br/dados/serie"),
                timeout=bcb_cfg.get("timeout", 15),
                rate_limit_seconds=bcb_cfg.get("rate_limit_seconds", 1.0),
            ),
            ibge=IBGEAPI(
                base_url=ibge_cfg.get("base_url", "https://servicodados.ibge.gov.br/api"),
                timeout=ibge_cfg.get("timeout", 15),
                rate_limit_seconds=ibge_cfg.get("rate_limit_seconds", 0.5),
            ),
            max_sources=int(agents.get("data_integrator", {}).get("max_sources", 12)),
            offline=bool(integ.get("offline", False)),
        )

    # ------------------------------------------------------------------ gather
    def gather(self, topic: str, limit_per_source: int = 5) -> dict[str, Any]:
        """Coleta e deduplica evidências reais para o tópico informado."""
        started = time.time()
        if self.offline:
            return {
                "topico": topic,
                "coletado_em": utcnow_iso(),
                "duracao_s": 0.0,
                "artigos": [],
                "series_brasil": [],
                "contexto_brasil": {},
                "lattes": [],
                "total_fontes": 0,
                "modo": "offline",
                "avisos": ["Coleta desabilitada (integrations.offline = true)."],
            }

        crossref = self.crossref.search(topic, rows=limit_per_source)
        scielo = self.scielo.search(topic, limit=limit_per_source)
        openalex = self.openalex.search(topic, limit=limit_per_source)
        series = self.bcb.painel_brasil(n=2)
        contexto = self.ibge.contexto_brasil(limite_ufs=5)
        lattes = self.lattes.search(topic, limit=1)

        articles = self._dedupe(scielo + crossref + openalex)[: self.max_sources]
        return {
            "topico": topic,
            "coletado_em": utcnow_iso(),
            "duracao_s": round(time.time() - started, 2),
            "artigos": articles,
            "artigos_crossref": len(crossref),
            "artigos_scielo": len(scielo),
            "artigos_openalex": len(openalex),
            "series_brasil": series,
            "contexto_brasil": contexto,
            "lattes": lattes,
            "total_fontes": len(articles) + len(series) + len(contexto.get("unidades_federacao", [])),
            "modo": "online",
        }

    @staticmethod
    def _dedupe(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicatas por DOI (ou título) preservando o registro mais rico.

        O OpenAlex traz contagem de citações, mas não resumo; Crossref traz
        resumo, mas não citações. Ao mesclar por DOI, ficamos com os dois.
        """
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for article in articles:
            key = (article.get("doi") or "").lower().strip()
            if not key:
                key = (article.get("titulo") or "").lower().strip()[:80]
            if not key:
                continue
            if key not in merged:
                merged[key] = dict(article)
                order.append(key)
                continue
            current = merged[key]
            for field, value in article.items():
                if value in (None, "", [], {}):
                    continue
                if current.get(field) in (None, "", [], {}):
                    current[field] = value
            if article.get("citacoes") is not None:
                current["citacoes"] = article["citacoes"]
            fontes = {current.get("fonte"), article.get("fonte")} - {None}
            if len(fontes) > 1:
                current["fonte"] = "+".join(sorted(fontes))
        return [merged[key] for key in order]

    # --------------------------------------------------------------- context
    def reference_context(self, gathered: dict[str, Any], max_refs: int = 12) -> str:
        """Formata as evidências coletadas como contexto textual para os agentes."""
        lines = ["EVIDÊNCIAS COLETADAS (fontes abertas verificáveis):"]
        artigos = gathered.get("artigos", [])
        if not artigos:
            lines.append("- Nenhum artigo recuperado nas fontes consultadas.")
        for i, art in enumerate(artigos[:max_refs], start=1):
            autores = ", ".join(art.get("autores", [])[:3]) or "Autoria não informada"
            citacoes = art.get("citacoes")
            cit_str = f", citações={citacoes}" if citacoes is not None else ""
            lines.append(
                f"[{i}] {art.get('titulo')} — {autores} ({art.get('ano') or 's/d'}), "
                f"{art.get('periodico') or 'periódico não informado'} "
                f"[fonte={art.get('fonte')}, DOI={art.get('doi') or 'n/a'}, "
                f"acesso={art.get('acesso') or 'n/a'}{cit_str}]"
            )
        series = gathered.get("series_brasil", [])
        if series:
            lines.append("\nINDICADORES BRASILEIROS (Banco Central/SGS):")
            for item in series[:6]:
                lines.append(f"- {item.get('descricao')} em {item.get('data')}: {item.get('valor')}")
        ufs = (gathered.get("contexto_brasil") or {}).get("unidades_federacao", [])
        if ufs:
            lines.append("\nCONTEXTO TERRITORIAL (IBGE):")
            for uf in ufs:
                pop = uf.get("populacao_estimada") or "n/d"
                lines.append(f"- {uf.get('nome')} ({uf.get('sigla')}): população estimada {pop}")
        return "\n".join(lines)

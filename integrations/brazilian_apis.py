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


def _para_float(valor: Any) -> float | None:
    """Converte um valor da API para ``float``; devolve ``None`` se não numérico.

    As APIs brasileiras alternam entre número JSON e string numérica, e usam
    ``None``/``""`` para semanas sem medição. Sem normalizar, a montagem do
    dataset real quebra em ``float(None)``.
    """
    if valor is None:
        return None
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    try:
        return float(str(valor).strip())
    except (TypeError, ValueError):
        return None


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
# InfoDengue (Fiocruz/FGV) — vigilância de arboviroses com clima acoplado
# --------------------------------------------------------------------------- #
@dataclass
class InfoDengueAPI(BaseAPI):
    """Semanas epidemiológicas do InfoDengue, com clima real acoplado.

    O InfoDengue (Fiocruz + FGV) publica, por município e semana epidemiológica,
    os casos notificados/estimados de dengue e as variáveis climáticas que
    entraram no modelo de nowcasting (temperatura e umidade médias). Isso permite
    um experimento **com dados observados**, sem gerador sintético: as features
    são medições reais e o alvo é o nível de alerta da semana seguinte.
    """

    base_url: str = "https://info.dengue.mat.br/api"
    name: str = "infodengue"

    CIDADES = {
        3304557: "Rio de Janeiro",
        3550308: "São Paulo",
        3106200: "Belo Horizonte",
        4106902: "Curitiba",
        5208707: "Goiânia",
        2927408: "Salvador",
        1302603: "Manaus",
        2611606: "Recife",
    }

    # Campos numéricos que o endpoint devolve e que usamos como features.
    CAMPOS_NUMERICOS = (
        "tempmed",
        "umidmed",
        "casos",
        "casos_est",
        "p_inc100k",
        "Rt",
        "nivel",
        "nivel_inc",
    )

    def semanas(self, geocode: int, ano: int, ew_inicio: int = 1, ew_fim: int = 52) -> list[dict[str, Any]]:
        """Semanas epidemiológicas de um município em um ano, ordenadas por SE."""
        data = self.get_json(
            f"{self.base_url}/alertcity",
            {
                "geocode": geocode,
                "disease": "dengue",
                "format": "json",
                "ew_start": ew_inicio,
                "ew_end": ew_fim,
                "ey_start": ano,
                "ey_end": ano,
            },
        )
        if not data:
            return []
        return sorted(data, key=lambda x: x.get("SE", 0))

    def coletar(
        self,
        municipios: dict[int, str] | None = None,
        anos: tuple[int, ...] = (2022, 2023),
    ) -> list[dict[str, Any]]:
        """Séries semanais de vários municípios e anos, em ordem cronológica.

        Cada item traz ``municipio``/``geocode``/``SE`` e os campos numéricos
        convertidos para ``float`` (``None`` quando a fonte não publica o valor).
        """
        alvos = municipios or self.CIDADES
        registros: list[dict[str, Any]] = []
        for geocode, nome in alvos.items():
            for ano in anos:
                for semana in self.semanas(geocode, ano):
                    item: dict[str, Any] = {
                        "geocode": geocode,
                        "municipio": nome,
                        "SE": semana.get("SE"),
                        "ano": ano,
                    }
                    for campo in self.CAMPOS_NUMERICOS:
                        item[campo] = _para_float(semana.get(campo))
                    registros.append(item)
        return registros


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
# DATASUS (microdados SINAN/SIM via FTP)
# --------------------------------------------------------------------------- #
@dataclass
class DATASUSAPI(BaseAPI):
    """Microdados do DATASUS via FTP oficial (``ftp.datasus.gov.br``).

    O TabNet (``tabnet.datasus.gov.br``) é apenas uma interface de tabulação: o
    ``POST`` exige a sessão do CGI e devolve HTML, não dados brutos. A rota
    programática confiável é o FTP público ``/dissemin/publicos``, que serve os
    arquivos ``.dbc`` (DBF comprimido) de origem do SINAN, SIM e SINASC.

    Este cliente baixa, converte ``.dbc`` → ``.dbf`` e lê os registros, com
    cache em disco para não repetir transferências pesadas. A conversão usa
    ``pyreaddbc``/``dbfread`` (dependências de ``pysus``); sem elas, o cliente
    sinaliza a indisponibilidade em vez de fabricar dados.
    """

    base_url: str = "ftp.datasus.gov.br"
    name: str = "datasus"
    cache_dir: Any = None
    # Subdiretórios reais dos conjuntos usados pelo pipeline.
    DIRETORIOS = {
        "sinan_dengue": "/dissemin/publicos/SINAN/DADOS/FINAIS",
        "sim_dores": "/dissemin/publicos/SIM/CID10/DORES",
        "sinasc": "/dissemin/publicos/SINASC/NOV/DNRES",
    }

    def _cache_path(self) -> Any:
        from pathlib import Path

        base = Path(self.cache_dir) if self.cache_dir else Path("outputs/datasus_cache")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def arquivos(self, conjunto: str = "sinan_dengue", padrao: str = "") -> list[str]:
        """Lista os arquivos de um conjunto no FTP (ex.: ``DENGBR*``)."""
        import ftplib

        diretorio = self.DIRETORIOS.get(conjunto)
        if diretorio is None:
            raise ValueError(f"conjunto desconhecido: {conjunto}")
        ftp = ftplib.FTP(self.base_url, timeout=self.timeout)
        try:
            ftp.login()
            ftp.cwd(diretorio)
            nomes = ftp.nlst()
        finally:
            try:
                ftp.quit()
            except Exception:  # noqa: BLE001
                ftp.close()
        if padrao:
            alvo = padrao.upper()
            nomes = [n for n in nomes if alvo in n.upper()]
        return sorted(nomes)

    def baixar(self, conjunto: str, arquivo: str) -> Any:
        """Baixa um ``.dbc`` para o cache local e devolve o caminho."""
        import ftplib
        from pathlib import Path

        destino = self._cache_path() / f"{conjunto}_{arquivo}"
        if destino.exists() and destino.stat().st_size > 0:
            logger.info("DATASUS: usando cache %s", destino)
            return destino
        diretorio = self.DIRETORIOS.get(conjunto)
        if diretorio is None:
            raise ValueError(f"conjunto desconhecido: {conjunto}")
        ftp = ftplib.FTP(self.base_url, timeout=max(self.timeout, 60))
        try:
            ftp.login()
            ftp.cwd(diretorio)
            with open(destino, "wb") as handle:
                ftp.retrbinary(f"RETR {arquivo}", handle.write)
        finally:
            try:
                ftp.quit()
            except Exception:  # noqa: BLE001
                ftp.close()
        logger.info("DATASUS: baixado %s (%.1f MB)", arquivo, destino.stat().st_size / 1e6)
        return destino

    def ler_dbc(self, caminho: Any) -> list[dict[str, Any]]:
        """Converte ``.dbc`` → ``.dbf`` e lê os registros como dicionários.

        Levanta ``RuntimeError`` quando as dependências de conversão faltam ou o
        arquivo está corrompido — nunca devolve dados inventados.
        """
        from pathlib import Path

        origem = Path(caminho)
        try:
            import pyreaddbc
            from dbfread import DBF
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise RuntimeError(
                "Leitura de .dbc requer pyreaddbc e dbfread (instale pysus)."
            ) from exc

        dbf_path = origem.with_suffix(".dbf")
        if not dbf_path.exists():
            pyreaddbc.dbc2dbf(str(origem), str(dbf_path))
        return [dict(registro) for registro in DBF(str(dbf_path), encoding="latin-1")]

    def coletar_dengue(self, anos: tuple[int, ...] = (2019, 2020, 2021)) -> list[dict[str, Any]]:
        """Notificações semanais de dengue por UF, a partir dos microdados SINAN.

        Agrega os casos notificados por UF e semana epidemiológica de
        notificação, que é a granularidade usada pela vigilância. Cada item traz
        ``uf``, ``ano``, ``semana`` e ``casos`` (contagem real de registros).
        """
        from collections import Counter

        contagem: Counter[tuple[str, int, int]] = Counter()
        for ano in anos:
            arquivo = f"DENGBR{str(ano)[-2:]}.dbc"
            try:
                caminho = self.baixar("sinan_dengue", arquivo)
                registros = self.ler_dbc(caminho)
            except (RuntimeError, OSError) as exc:
                logger.warning("DATASUS: %s indisponível (%s)", arquivo, exc)
                continue
            for registro in registros:
                uf = str(registro.get("SG_UF_NOT") or registro.get("SG_UF") or "").strip()
                semana = _para_float(registro.get("SEM_NOT"))
                if not uf or semana is None:
                    continue
                contagem[(uf, ano, int(semana))] += 1
        registros_finais = [
            {"uf": uf, "ano": ano, "semana": semana, "casos": casos, "fonte": "DATASUS/SINAN"}
            for (uf, ano, semana), casos in sorted(contagem.items())
        ]
        logger.info("DATASUS: %s semanas UF×SE agregadas de %s anos", len(registros_finais), len(anos))
        return registros_finais


# --------------------------------------------------------------------------- #
# INPE / TerraBrasilis (PRODES — desmatamento)
# --------------------------------------------------------------------------- #
@dataclass
class INPEAPI(BaseAPI):
    """Desmatamento oficial do INPE (PRODES) via WFS do TerraBrasilis.

    O TerraBrasilis publica o PRODES como GeoServer WFS. Consultamos a camada
    ``yearly_deforestation_biome`` (polígonos anuais por bioma) com filtro CQL
    por ano e pedimos a saída em CSV — dados oficiais de área desmatada
    (``area_km``) por estado e ano, sem intermediação de terceiros.
    """

    base_url: str = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"
    name: str = "inpe"
    layer: str = "prodes-amazon-nb:yearly_deforestation_biome"

    def desmatamento_por_estado(self, ano: int, layer: str | None = None) -> list[dict[str, Any]]:
        """Soma a área desmatada (km²) por UF em um ano do PRODES."""
        camada = layer or self.layer
        response = self._fetch(
            self.base_url,
            {
                "service": "WFS",
                "version": "1.0.0",
                "request": "GetFeature",
                "typeName": camada,
                "outputFormat": "csv",
                "propertyName": "state,year,area_km",
                "CQL_FILTER": f"year={int(ano)}",
            },
        )
        if response is None:
            return []
        import csv as _csv
        import io

        leitor = _csv.DictReader(io.StringIO(response.text))
        agregado: dict[str, float] = {}
        for linha in leitor:
            uf = (linha.get("state") or "").strip()
            area = _para_float(linha.get("area_km"))
            if not uf or area is None:
                continue
            agregado[uf] = agregado.get(uf, 0.0) + area
        return [
            {"uf": uf, "ano": int(ano), "area_km2": round(area, 4), "fonte": "INPE/PRODES"}
            for uf, area in sorted(agregado.items())
        ]

    def serie_desmatamento(self, anos: tuple[int, ...], layer: str | None = None) -> list[dict[str, Any]]:
        """Série de desmatamento por UF e ano (concatenando anos do PRODES)."""
        registros: list[dict[str, Any]] = []
        for ano in anos:
            registros.extend(self.desmatamento_por_estado(ano, layer=layer))
        return registros


# --------------------------------------------------------------------------- #
# IBGE SIDRA (indicadores agregados: agricultura, educação)
# --------------------------------------------------------------------------- #
@dataclass
class SIDRAAPI(BaseAPI):
    """Indicadores agregados do IBGE via API SIDRA (``apisidra.ibge.gov.br``).

    A SIDRA expõe as tabelas do IBGE em JSON. Cada resposta traz um cabeçalho
    seguido das linhas; ``V`` é o valor e os campos ``D*N`` identificam as
    dimensões. Usamos para produção agrícola municipal (PAM) e alfabetização.
    """

    base_url: str = "https://apisidra.ibge.gov.br/values"
    name: str = "sidra"

    def valores(self, tabela: int, variavel: int | str, periodo: str = "last%201", n3: str = "all") -> list[dict[str, Any]]:
        """Consulta uma tabela SIDRA e devolve as linhas (sem o cabeçalho)."""
        url = f"{self.base_url}/t/{tabela}/n3/{n3}/v/{variavel}/p/{periodo}"
        data = self.get_json(url)
        if not isinstance(data, list) or len(data) < 2:
            return []
        return data[1:]

    def producao_agricola(self, produto: int = 2713, anos: str = "last%202") -> list[dict[str, Any]]:
        """Produção (toneladas) de um produto da lavoura temporária por UF.

        Tabela 1612 (PAM), variável 214 (quantidade produzida), classificação
        ``c81`` (produto). Padrão: 2713 = soja.
        """
        url = f"{self.base_url}/t/1612/n3/all/v/214/p/{anos}/c81/{produto}"
        data = self.get_json(url)
        if not isinstance(data, list) or len(data) < 2:
            return []
        registros: list[dict[str, Any]] = []
        for linha in data[1:]:
            valor = _para_float(linha.get("V"))
            ano = _para_float(linha.get("D3N"))
            if valor is None or ano is None:
                continue
            registros.append(
                {
                    "uf": linha.get("D1N"),
                    "ano": int(ano),
                    "producao_t": valor,
                    "produto": linha.get("D4N") or str(produto),
                    "fonte": "IBGE/SIDRA 1612",
                }
            )
        return registros

    def alfabetizacao(self, periodo: str = "last%201") -> list[dict[str, Any]]:
        """Taxa de alfabetização (%) das pessoas de 15+ anos por UF (PNAD/SIDRA)."""
        linhas = self.valores(9543, "allxp", periodo=periodo)
        registros: list[dict[str, Any]] = []
        for linha in linhas:
            valor = _para_float(linha.get("V"))
            if valor is None or not linha.get("D1N"):
                continue
            registros.append(
                {
                    "uf": linha.get("D1N"),
                    "ano": _para_float(linha.get("D5N")),
                    "alfabetizacao_pct": valor,
                    "fonte": "IBGE/SIDRA 9543",
                }
            )
        return registros


# --------------------------------------------------------------------------- #
# BDTD (teses e dissertações brasileiras)
# --------------------------------------------------------------------------- #
@dataclass
class BDTDAPI(BaseAPI):
    """Biblioteca Digital de Teses e Dissertações (IBICT) via API VuFind.

    A BDTD usa o VuFind, cuja API REST (``/vufind/api/v1/search``) devolve JSON
    com autores, títulos, resumos e instituições — produção acadêmica brasileira
    que não aparece no Crossref. É usada para ampliar o estado da arte com
    trabalhos nacionais.
    """

    base_url: str = "https://bdtd.ibict.br/vufind/api/v1/search"
    name: str = "bdtd"

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        data = self.get_json(
            self.base_url,
            {
                "lookfor": query,
                "type": "AllFields",
                "limit": max(1, min(limit, 20)),
                "field[]": ["title", "authors", "year", "summary", "urls", "formats"],
            },
        )
        if not data:
            return []
        registros: list[dict[str, Any]] = []
        for item in data.get("records", []):
            autores = list((item.get("authors") or {}).get("primary", {}).keys())
            urls = [u.get("url") for u in (item.get("urls") or []) if u.get("url")]
            registros.append(
                {
                    "fonte": "bdtd",
                    "titulo": item.get("title") or "(sem título)",
                    "autores": autores,
                    "ano": (item.get("year") or [None])[0] if isinstance(item.get("year"), list) else item.get("year"),
                    "tipo": (item.get("formats") or [None])[0],
                    "resumo": _clean_markup((item.get("summary") or [""])[0] if item.get("summary") else ""),
                    "url": urls[0] if urls else None,
                    "acesso": "aberto",
                }
            )
        return registros


# --------------------------------------------------------------------------- #
# Dataverse (repositórios de dados brasileiros)
# --------------------------------------------------------------------------- #
@dataclass
class DataverseAPI(BaseAPI):
    """Dataverses brasileiros (ex.: FGV) via API de busca pública.

    Devolve metadados de conjuntos de dados depositados, com identificador
    persistente (``hdl``) e citação sugerida — a procedência exigida para citar
    dados de repositório.
    """

    base_url: str = "https://dataverse.fgv.br/api/v1/search"
    name: str = "dataverse"

    def search(self, query: str, limit: int = 5, tipo: str = "dataset") -> list[dict[str, Any]]:
        data = self.get_json(
            self.base_url,
            {"q": query, "type": tipo, "per_page": max(1, min(limit, 20))},
        )
        if not data:
            return []
        itens = (data.get("data") or {}).get("items", [])
        return [
            {
                "fonte": "dataverse",
                "titulo": item.get("name"),
                "autores": item.get("authors") or [],
                "ano": (item.get("published_at") or "")[:4] or None,
                "identificador": item.get("global_id"),
                "url": item.get("url"),
                "citacao": item.get("citation"),
                "publicador": item.get("publisher"),
                "acesso": "aberto",
            }
            for item in itens
        ]


# --------------------------------------------------------------------------- #
# Portal da Transparência (requer chave) e Kaggle (busca pública)
# --------------------------------------------------------------------------- #
@dataclass
class PortalTransparenciaAPI(BaseAPI):
    """Portal da Transparência (CGU).

    A API oficial exige chave de acesso gratuita (``api.portaldatransparencia.gov.br``);
    sem ``token`` o serviço responde HTTP 401. Este cliente não contorna a
    autenticação: devolve explicitamente a exigência de chave para que o
    pipeline registre a fonte como indisponível, em vez de inventar números.
    """

    base_url: str = "https://api.portaldatransparencia.gov.br/api-de-dados"
    name: str = "transparencia"
    token: str | None = None

    def search(self, endpoint: str = "orgaos-siafi", params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.token:
            logger.info(
                "Portal da Transparência exige chave de API gratuita; fonte não consultada "
                "(cadastre em portaldatransparencia.gov.br/api-de-dados)."
            )
            return []
        self.session.headers.update({"chave-api-dados": self.token})
        data = self.get_json(f"{self.base_url}/{endpoint}", params or {})
        return data if isinstance(data, list) else []


@dataclass
class KaggleBRAPI(BaseAPI):
    """Busca pública de datasets do Kaggle (metadados, sem autenticação).

    O catálogo do Kaggle é consultável sem chave (``/api/v1/datasets/list``),
    mas o *download* dos arquivos exige credenciais. Usamos a busca para citar
    conjuntos públicos com licença e procedência; não baixamos arquivos sem
    autenticação.
    """

    base_url: str = "https://www.kaggle.com/api/v1/datasets/list"
    name: str = "kaggle"

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        data = self.get_json(self.base_url, {"search": query, "page": 1})
        if not isinstance(data, list):
            return []
        registros: list[dict[str, Any]] = []
        for item in data[:limit]:
            registros.append(
                {
                    "fonte": "kaggle",
                    "titulo": item.get("titleNullable") or item.get("title"),
                    "autores": [item.get("creatorNameNullable")] if item.get("creatorNameNullable") else [],
                    "ano": None,
                    "licenca": item.get("licenseNameNullable"),
                    "url": item.get("urlNullable"),
                    "download_requer_autenticacao": True,
                    "acesso": "aberto",
                }
            )
        return registros


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
    bdtd: BDTDAPI = field(default_factory=BDTDAPI)
    datasus: DATASUSAPI = field(default_factory=DATASUSAPI)
    inpe: INPEAPI = field(default_factory=INPEAPI)
    sidra: SIDRAAPI = field(default_factory=SIDRAAPI)
    dataverse: DataverseAPI = field(default_factory=DataverseAPI)
    kaggle: KaggleBRAPI = field(default_factory=KaggleBRAPI)
    transparencia: PortalTransparenciaAPI = field(default_factory=PortalTransparenciaAPI)
    max_sources: int = 12
    offline: bool = False
    # Ano de referência do PRODES consultado (o bioma é atualizado anualmente).
    prodes_ano: int = 2022
    # Fontes de microdados são pesadas; desabilitadas por padrão no agregador.
    incluir_microdados: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BrazilianDataAggregator":
        integ = config.get("integrations", {})
        agents = config.get("agents", {})
        cr_cfg = integ.get("crossref", {})
        sc_cfg = integ.get("scielo", {})
        ibge_cfg = integ.get("ibge", {})
        bcb_cfg = integ.get("bcb", {})
        bdtd_cfg = integ.get("bdtd", {})
        inpe_cfg = integ.get("inpe", {})
        sidra_cfg = integ.get("sidra", {})
        dv_cfg = integ.get("dataverse", {})
        ds_cfg = integ.get("datasus", {})
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
            bdtd=BDTDAPI(
                base_url=bdtd_cfg.get("base_url", "https://bdtd.ibict.br/vufind/api/v1/search"),
                timeout=bdtd_cfg.get("timeout", 20),
                rate_limit_seconds=bdtd_cfg.get("rate_limit_seconds", 1.0),
            ),
            datasus=DATASUSAPI(
                base_url=ds_cfg.get("base_url", "ftp.datasus.gov.br"),
                timeout=ds_cfg.get("timeout", 60),
                rate_limit_seconds=ds_cfg.get("rate_limit_seconds", 0.0),
                cache_dir=ds_cfg.get("cache_dir"),
            ),
            inpe=INPEAPI(
                base_url=inpe_cfg.get("base_url", "https://terrabrasilis.dpi.inpe.br/geoserver/ows"),
                timeout=inpe_cfg.get("timeout", 60),
                rate_limit_seconds=inpe_cfg.get("rate_limit_seconds", 1.0),
                layer=inpe_cfg.get("layer", "prodes-amazon-nb:yearly_deforestation_biome"),
            ),
            sidra=SIDRAAPI(
                base_url=sidra_cfg.get("base_url", "https://apisidra.ibge.gov.br/values"),
                timeout=sidra_cfg.get("timeout", 30),
                rate_limit_seconds=sidra_cfg.get("rate_limit_seconds", 0.5),
            ),
            dataverse=DataverseAPI(
                base_url=dv_cfg.get("base_url", "https://dataverse.fgv.br/api/v1/search"),
                timeout=dv_cfg.get("timeout", 30),
                rate_limit_seconds=dv_cfg.get("rate_limit_seconds", 1.0),
            ),
            kaggle=KaggleBRAPI(
                timeout=integ.get("kaggle", {}).get("timeout", 20),
                rate_limit_seconds=integ.get("kaggle", {}).get("rate_limit_seconds", 1.0),
            ),
            transparencia=PortalTransparenciaAPI(
                base_url=integ.get("transparencia", {}).get(
                    "base_url", "https://api.portaldatransparencia.gov.br/api-de-dados"
                ),
                timeout=integ.get("transparencia", {}).get("timeout", 20),
                token=integ.get("transparencia", {}).get("token"),
            ),
            max_sources=int(agents.get("data_integrator", {}).get("max_sources", 12)),
            offline=bool(integ.get("offline", False)),
            prodes_ano=int(inpe_cfg.get("ano", 2022)),
            incluir_microdados=bool(integ.get("incluir_microdados", False)),
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
                "datasets": [],
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
        bdtd = self.bdtd.search(topic, limit=limit_per_source)
        series = self.bcb.painel_brasil(n=2)
        contexto = self.ibge.contexto_brasil(limite_ufs=5)
        lattes = self.lattes.search(topic, limit=1)
        datasets = self._gather_datasets(topic, limit_per_source)

        articles = self._dedupe(scielo + crossref + openalex + bdtd)[: self.max_sources]
        indicadores = self._gather_indicadores()
        return {
            "topico": topic,
            "coletado_em": utcnow_iso(),
            "duracao_s": round(time.time() - started, 2),
            "artigos": articles,
            "artigos_crossref": len(crossref),
            "artigos_scielo": len(scielo),
            "artigos_openalex": len(openalex),
            "artigos_bdtd": len(bdtd),
            "datasets": datasets,
            "series_brasil": series,
            "indicadores_brasil": indicadores,
            "contexto_brasil": contexto,
            "lattes": lattes,
            "total_fontes": len(articles)
            + len(series)
            + len(datasets)
            + len(indicadores)
            + len(contexto.get("unidades_federacao", [])),
            "modo": "online",
        }

    def _gather_datasets(self, topic: str, limit: int) -> list[dict[str, Any]]:
        """Catálogos de dados (Dataverse, Kaggle) e microdados (DATASUS, INPE).

        Só as fontes leves rodam por padrão; os microdados do DATASUS e o
        PRODES/INPE só são consultados quando ``incluir_microdados`` está ligado,
        porque baixam centenas de MB.
        """
        datasets = self.dataverse.search(topic, limit=limit)
        datasets += self.kaggle.search(topic, limit=limit)
        if self.incluir_microdados:
            try:
                prodes = self.inpe.desmatamento_por_estado(self.prodes_ano)
                if prodes:
                    datasets.append(
                        {
                            "fonte": "inpe/prodes",
                            "titulo": f"Desmatamento PRODES {self.prodes_ano} por UF (km²)",
                            "ano": self.prodes_ano,
                            "registros": prodes,
                            "url": "https://terrabrasilis.dpi.inpe.br/",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - fonte externa instável
                logger.warning("INPE/PRODES indisponível: %s", exc)
        return datasets

    def _gather_indicadores(self) -> list[dict[str, Any]]:
        """Indicadores agregados reais (SIDRA) usados como contexto do país."""
        indicadores: list[dict[str, Any]] = []
        try:
            indicadores += self.sidra.producao_agricola()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SIDRA (produção agrícola) indisponível: %s", exc)
        try:
            indicadores += self.sidra.alfabetizacao()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SIDRA (alfabetização) indisponível: %s", exc)
        return indicadores

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
        indicadores = gathered.get("indicadores_brasil", [])
        if indicadores:
            lines.append("\nINDICADORES AGREGADOS (IBGE/SIDRA):")
            for item in indicadores[:6]:
                if "producao_t" in item:
                    lines.append(
                        f"- Produção de {item.get('produto')} em {item.get('uf')} ({item.get('ano')}): "
                        f"{item.get('producao_t')} t [{item.get('fonte')}]"
                    )
                else:
                    lines.append(
                        f"- Alfabetização (15+) em {item.get('uf')} ({item.get('ano')}): "
                        f"{item.get('alfabetizacao_pct')}% [{item.get('fonte')}]"
                    )
        datasets = gathered.get("datasets", [])
        if datasets:
            lines.append("\nCONJUNTOS DE DADOS PÚBLICOS (repositórios):")
            for item in datasets[:6]:
                ident = item.get("identificador") or item.get("url") or "s/identificador"
                lines.append(
                    f"- {item.get('titulo')} [{item.get('fonte')}, {ident}]"
                )
        return "\n".join(lines)

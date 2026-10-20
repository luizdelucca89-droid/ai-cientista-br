"""Interface gráfica (Streamlit) do AI Cientista Brasileiro.

Rodar::

    streamlit run app.py

A interface expõe quatro abas: iniciar uma pesquisa, consultar o histórico de
ciclos gravados na memória SQLite, revisar/rebuttar um ciclo existente e ajustar
configurações locais (``.env``).

O experimento continua sendo executado pelo mesmo ``Orchestrator`` da CLI —
a UI apenas dispara o ciclo e apresenta os artefatos. Nenhum dado é gerado
ou simulado: se não houver fonte pública para o tema, o ciclo falha com
``RealDataUnavailableError`` e a UI mostra o motivo.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.config import Config
from core.memory import MemorySystem
from core.orchestrator import Orchestrator
from core.rebuttal_engine import RebuttalEngine
from integrations.publishing import IntegrityError, generate_submission_package

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

TEMAS_EXEMPLO = {
    "Dengue (InfoDengue / Fiocruz)": "Predição de surtos de dengue usando dados climáticos no Brasil",
    "Desmatamento (INPE / PRODES)": "Desmatamento na Amazônia Legal e sua relação com o uso do solo",
    "Agricultura (IBGE / SIDRA PAM)": "Machine Learning aplicado ao agronegócio brasileiro",
    "Educação (IBGE / SIDRA ODS)": "Impacto da infraestrutura escolar no desempenho educacional brasileiro",
}

# As 8 fases que o orquestrador percorre, para o painel de progresso.
FASES = [
    "Coleta de evidências (SciELO, Crossref, OpenAlex)",
    "Planejamento e hipóteses",
    "Dados reais e experimento",
    "Redação iterativa",
    "Revisão por painel",
    "Validação de qualidade",
    "Formatação ABNT (Markdown/PDF/LaTeX)",
    "Persistência na memória",
]


@st.cache_resource(show_spinner=False)
def _config() -> Config:
    return Config.load()


def _carregar_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    dados: dict[str, str] = {}
    for linha in ENV_PATH.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        dados[chave.strip()] = valor.strip()
    return dados


def _salvar_env(valores: dict[str, str]) -> None:
    existentes = _carregar_env()
    existentes.update({k: v for k, v in valores.items() if v})
    linhas = [f"{k}={v}" for k, v in sorted(existentes.items())]
    ENV_PATH.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    for k, v in existentes.items():
        os.environ[k] = v


def _aba_nova_pesquisa(config: Config) -> None:
    st.subheader("🔬 Nova pesquisa científica")

    presets = list(TEMAS_EXEMPLO.keys()) + ["Outro (digitar)…"]
    escolha = st.selectbox("Tema com fonte pública conhecida", presets)
    if escolha == "Outro (digitar)…":
        topico = st.text_input("Tópico livre", placeholder="Ex.: qualidade do ar em São Paulo")
    else:
        topico = st.text_input("Tópico", value=TEMAS_EXEMPLO[escolha])

    col1, col2, col3 = st.columns(3)
    with col1:
        provider = st.selectbox("Provider de LLM", ["mock", "openai_compatible", "selenium"])
    with col2:
        force = st.checkbox("Repetir mesmo se já estudado", value=True)
    with col3:
        offline = st.checkbox("Somente cache local (offline)", value=False)

    st.caption(
        "Sem fonte pública para o tópico, o experimento é recusado com "
        "`RealDataUnavailableError` — o sistema não fabrica dados."
    )

    iniciar = st.button("🚀 Iniciar pesquisa científica", type="primary", disabled=not topico)

    if not iniciar:
        return

    progresso = st.progress(0.0, text="Inicializando…")
    estado = st.status("Executando o pipeline…", expanded=True)
    caixa: dict[str, object] = {}

    def _executar() -> None:
        try:
            cfg_data = config.as_dict()
            cfg_data.setdefault("llm", {})["provider"] = provider
            cfg_data.setdefault("integrations", {})["offline"] = offline
            orquestrador = Orchestrator(config=Config(data=cfg_data), output_dir=config.output_dir)
            caixa["resultado"] = orquestrador.run_research_cycle(topic=topico, force=force)
            orquestrador.close()
        except Exception as exc:  # noqa: BLE001 - a UI precisa mostrar o erro
            caixa["erro"] = exc

    thread = threading.Thread(target=_executar, daemon=True)
    thread.start()
    for i, fase in enumerate(FASES):
        thread.join(timeout=0.05)
        progresso.progress((i + 1) / len(FASES), text=fase)
    thread.join()

    if "erro" in caixa:
        estado.update(label="Falha no ciclo", state="error")
        st.error(f"{type(caixa['erro']).__name__}: {caixa['erro']}")
        return

    estado.update(label="Ciclo concluído", state="complete")
    _render_resultado(caixa["resultado"])


def _render_resultado(resultado: dict) -> None:
    fases = resultado.get("fases", resultado)
    experimento = (fases.get("experimento") or {}).get("resultado") or {}
    dataset = experimento.get("dataset") or {}
    artefatos = fases.get("artefatos") or {}

    st.success(f"Melhor modelo: **{experimento.get('best_model')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Amostras reais", dataset.get("n_amostras") or "—")
    c2.metric("Fonte", (dataset.get("origem") or "—")[:28])
    c3.metric("Score de validação", fases.get("validacao", {}).get("score") or "—")

    if dataset.get("fonte_url"):
        st.caption(f"Procedência: [{dataset.get('origem')}]({dataset.get('fonte_url')})")

    metricas = experimento.get("metrics") or {}
    _grafico_metricas(metricas)

    for figura in artefatos.get("figuras") or []:
        if Path(figura).exists():
            st.image(figura, caption="Comparação de modelos (F1-score)")

    run_dir = artefatos.get("diretorio")
    if not run_dir:
        return

    st.markdown("### Downloads")
    d1, d2, d3, d4 = st.columns(4)
    _botao_download(d1, artefatos.get("paper_pdf"), "📄 Paper (PDF ABNT)", "paper.pdf")
    _botao_download(d2, artefatos.get("paper_tex"), "📐 LaTeX (.tex)", "paper.tex")
    _botao_download(d3, artefatos.get("paper_markdown"), "📝 Markdown (.md)", "paper.md")

    with d4:
        if st.button("📦 Gerar pacote de submissão"):
            try:
                pacote = generate_submission_package(run_dir)
                st.session_state["pacote"] = pacote.path
                st.success("Pacote gerado.")
            except IntegrityError as exc:
                st.error(str(exc))

    if st.session_state.get("pacote"):
        _botao_download(None, st.session_state["pacote"], "⬇️ Pacote de submissão (.zip)", "submissao.zip")


def _grafico_metricas(metricas: dict) -> None:
    """Gráfico interativo de F1/AUC por modelo, quando há métricas disponíveis."""
    modelos = [m for m, v in metricas.items() if v.get("f1_score") is not None]
    if not modelos:
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fig = go.Figure()
    fig.add_bar(name="F1-score", x=modelos, y=[metricas[m]["f1_score"] for m in modelos])
    if any(metricas[m].get("auc_roc") is not None for m in modelos):
        fig.add_bar(name="AUC-ROC", x=modelos, y=[metricas[m].get("auc_roc") for m in modelos])
    fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _botao_download(coluna, caminho: str | None, rotulo: str, nome: str) -> None:
    if not caminho or not Path(caminho).exists():
        return
    dados = Path(caminho).read_bytes()
    destino = coluna if coluna is not None else st
    destino.download_button(rotulo, dados, file_name=nome, use_container_width=True)


def _aba_historico(config: Config) -> None:
    st.subheader("📚 Histórico de pesquisas")
    memoria = MemorySystem(path=config.memory_path)
    ciclos = memoria.list_cycles(limit=50)
    stats = memoria.stats()
    memoria.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("Ciclos", stats["ciclos"])
    c2.metric("Concluídos", stats["concluidos"])
    c3.metric("Fatos na base", stats["conhecimento"])

    if not ciclos:
        st.info("Nenhum ciclo registrado ainda. Rode uma pesquisa na primeira aba.")
        return

    linhas = []
    for ciclo in ciclos:
        payload = ciclo.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        experimento = ((payload.get("experimento") or {}).get("resultado")) or {}
        dataset = experimento.get("dataset") or {}
        linhas.append(
            {
                "ID": ciclo.get("id"),
                "Data": (ciclo.get("created_at") or "")[:19],
                "Tópico": ciclo.get("topic"),
                "Status": ciclo.get("status"),
                "Score": ciclo.get("score"),
                "Fonte de dados": dataset.get("origem") or "—",
                "Amostras": dataset.get("n_amostras"),
            }
        )
    st.dataframe(linhas, use_container_width=True)

    ids = [c["id"] for c in ciclos]
    selecionado = st.selectbox("Baixar artefatos do ciclo", ids)
    ciclo_dir = _diretorio_do_ciclo(config, selecionado)
    if ciclo_dir and ciclo_dir.is_dir():
        for nome, rotulo in [
            ("paper.pdf", "📄 paper.pdf"),
            ("paper.tex", "📐 paper.tex"),
            ("paper.md", "📝 paper.md"),
            ("ciclo_completo.json", "🗂️ ciclo_completo.json"),
        ]:
            caminho = ciclo_dir / nome
            if caminho.exists():
                st.download_button(rotulo, caminho.read_bytes(), file_name=nome, use_container_width=True)
    else:
        st.warning("Diretório do ciclo não encontrado no disco.")


def _diretorio_do_ciclo(config: Config, cycle_id: int) -> Path | None:
    for pasta in config.output_dir.glob(f"ciclo_{cycle_id}_*"):
        if pasta.is_dir():
            return pasta
    return None


def _aba_rebuttal(config: Config) -> None:
    st.subheader("🔄 Revisão e Rebuttal")
    st.caption(
        "Cole os comentários de um revisor humano. O sistema converte os comentários em "
        "ações, reexecuta o experimento quando pedido e atualiza o manuscrito."
    )

    try:
        memory = MemorySystem(config.output_dir / "memory.sqlite3")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível abrir a memória: {exc}")
        return

    ciclos = memory.list_cycles(limit=50)
    memory.close()
    concluidos = [c for c in ciclos if c.get("status") == "concluido"]
    if not concluidos:
        st.info("Nenhum ciclo concluído no histórico. Rode uma pesquisa primeiro.")
        return

    rotulos = {f"#{c['id']} — {c['topic'][:70]}": c["id"] for c in concluidos}
    escolha = st.selectbox("Ciclo a revisar", list(rotulos))
    cycle_id = rotulos[escolha]

    comentarios = st.text_area(
        "Comentários do revisor",
        height=200,
        placeholder=(
            "Ex.: O manuscrito deve re-executar o experimento reportando AUC além de F1. "
            "Reescreva a seção de resultados com mais clareza metodológica."
        ),
    )

    if st.button("🔄 Gerar Resposta e Atualizar Paper", type="primary"):
        if not comentarios.strip():
            st.warning("Cole os comentários do revisor antes de gerar a resposta.")
            return
        with st.spinner("Processando rebuttal: parseando ações, reexecutando e reescrevendo..."):
            try:
                llm = _construir_llm(config)
                engine = RebuttalEngine(llm=llm, output_dir=config.output_dir)
                relatorio = engine.process_rebuttal(cycle_id, comentarios)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha no rebuttal: {exc}")
                return
        st.session_state["rebuttal"] = relatorio

    relatorio = st.session_state.get("rebuttal")
    if not relatorio:
        return

    st.success(f"Rebuttal do ciclo #{relatorio['cycle_id']} concluído.")
    acoes = relatorio.get("acoes", [])
    st.markdown(f"**Ações identificadas:** {len(acoes)}")
    for acao in acoes:
        st.markdown(f"- `{acao['tipo']}` — {acao['descricao']}")

    comparacao = relatorio.get("comparacao_resultados") or {}
    if comparacao.get("antes") or comparacao.get("depois"):
        st.markdown("**Comparação de resultados (antes/depois)**")
        st.json(comparacao)

    caminho = Path(relatorio["resposta_md"])
    if caminho.exists():
        st.markdown("**Resposta aos revisores**")
        st.markdown(caminho.read_text(encoding="utf-8"))
        st.download_button(
            "⬇️ Baixar resposta_revisores.md",
            caminho.read_bytes(),
            file_name="resposta_revisores.md",
            use_container_width=True,
        )

    st.markdown("**Manuscrito atualizado**")
    run_dir = Path(relatorio["run_dir"])
    for nome, rotulo in (("paper.md", "⬇️ paper.md"), ("paper.tex", "⬇️ paper.tex")):
        arquivo = run_dir / nome
        if arquivo.exists():
            st.download_button(rotulo, arquivo.read_bytes(), file_name=nome, use_container_width=True)


def _construir_llm(config: Config):
    """Constrói o cliente LLM a partir da configuração, como a CLI faz."""
    from llm_providers import build_llm_client

    return build_llm_client(config.as_dict())


def _aba_configuracoes() -> None:
    st.subheader("⚙️ Configurações")
    st.caption("As chaves são gravadas em `.env` (fora do controle de versão) e no ambiente da sessão.")

    atual = _carregar_env()
    with st.form("config_form"):
        openai_key = st.text_input("OPENAI_API_KEY", value=atual.get("OPENAI_API_KEY", ""), type="password")
        base_url = st.text_input("OPENAI_BASE_URL", value=atual.get("OPENAI_BASE_URL", ""))
        scielo_token = st.text_input("SCIELO_TOKEN", value=atual.get("SCIELO_TOKEN", ""), type="password")
        arxiv_user = st.text_input("ARXIV_USERNAME", value=atual.get("ARXIV_USERNAME", ""))
        salvar = st.form_submit_button("💾 Salvar", type="primary")

    if salvar:
        _salvar_env(
            {
                "OPENAI_API_KEY": openai_key,
                "OPENAI_BASE_URL": base_url,
                "SCIELO_TOKEN": scielo_token,
                "ARXIV_USERNAME": arxiv_user,
            }
        )
        st.success(f"Configurações salvas em {ENV_PATH.name}.")


def main() -> None:
    st.set_page_config(page_title="AI Cientista Brasileiro", page_icon="🇧🇷", layout="wide")
    st.title("🇧🇷 AI Cientista Brasileiro")
    st.caption("Pesquisa científica autônoma, em português, sobre dados públicos reais do Brasil.")

    config = _config()
    st.session_state.setdefault("pacote", None)

    aba1, aba2, aba3, aba4 = st.tabs(
        ["🔬 Nova pesquisa", "📚 Histórico", "🔄 Revisão e Rebuttal", "⚙️ Configurações"]
    )
    with aba1:
        _aba_nova_pesquisa(config)
    with aba2:
        _aba_historico(config)
    with aba3:
        _aba_rebuttal(config)
    with aba4:
        _aba_configuracoes()


if __name__ == "__main__":
    main()

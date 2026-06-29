"""
app.py — Interface Streamlit
Sistema de Classificação de Editais de Licitação — PNCP
Disciplina: Mineração de Textos
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# Configuração geral da página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor de Editais — PNCP",
    page_icon="📋",
    layout="wide",
)

# ──────────────────────────────────────────────
# Caminhos
# ──────────────────────────────────────────────
PROCESSED_PATH = Path("dados/processed")
RAW_PATH       = Path("dados/raw")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
@st.cache_data
def carregarDados() -> pd.DataFrame:
    """Carrega o DataFrame processado salvo no notebook 1."""
    try:
        df = pd.read_parquet(PROCESSED_PATH / "editais_processado.parquet")
        if "uf" not in df.columns:
            df["uf"] = df["unidadeOrgao"].apply(
                lambda x: x.get("ufSigla") if isinstance(x, dict) else None
            )
        if "orgao" not in df.columns:
            df["orgao"] = df["unidadeOrgao"].apply(
                lambda x: x.get("nomeUnidade") if isinstance(x, dict) else None
            )
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def exibirImagem(nome_arquivo: str, legenda: str = ""):
    """Exibe um gráfico salvo em dados/processed/."""
    caminho = PROCESSED_PATH / nome_arquivo
    if caminho.exists():
        img = Image.open(caminho)
        st.image(img, caption=legenda, use_container_width=True)
    else:
        st.warning(f"Gráfico não encontrado: {nome_arquivo}")


def badgeRisco(nivel: str) -> str:
    """Retorna emoji + texto formatado para o alerta de risco."""
    mapa = {
        "baixo": "🟢 Baixo",
        "médio": "🟡 Médio",
        "alto":  "🔴 Alto",
    }
    return mapa.get(nivel.lower(), f"❓ {nivel}")


# ──────────────────────────────────────────────
# Barra lateral — navegação
# ──────────────────────────────────────────────
st.sidebar.title("📋 Monitor de Editais")
st.sidebar.markdown("Sistema de Classificação — PNCP")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegação",
    ["📊 Dashboard de Mercado", "🔍 Classificador de Editais"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Dados: 01/06/2026 a 03/06/2026")
st.sidebar.caption("Modelo LLM: LLaMA 3.3 70B (Groq)")
st.sidebar.caption("Embeddings: MiniLM Multilingual")


# ══════════════════════════════════════════════
# PÁGINA 1 — DASHBOARD DE MERCADO
# ══════════════════════════════════════════════
if pagina == "📊 Dashboard de Mercado":

    st.title("📊 Dashboard de Mercado")
    st.markdown(
        "Visão estratégica do mercado de licitações públicas brasileiras. "
        "Baseado em **3.009 editais** coletados via API do PNCP entre 01/06/2026 e 03/06/2026."
    )
    st.divider()

    # ── Métricas rápidas ──
    df = carregarDados()

    if not df.empty:
        total_editais   = len(df)
        total_modal     = df["categoria"].nunique()
        uf_top          = df["uf"].value_counts().idxmax() if "uf" in df.columns else "SP"
        orgao_top       = (df["orgao"].value_counts().idxmax()
                           if "orgao" in df.columns else "—")
        # trunca nome longo do órgão para caber no card
        orgao_top_curto = orgao_top[:40] + "..." if len(orgao_top) > 40 else orgao_top
    else:
        total_editais, total_modal, uf_top, orgao_top_curto = 3009, 6, "SP", "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Editais monitorados", f"{total_editais:,}")
    c2.metric("📑 Modalidades",         total_modal)
    c3.metric("🗺️ Estado com mais editais", uf_top)
    c4.metric("🏛️ Órgão com mais editais", orgao_top_curto)

    st.divider()

    # ── Gráficos EDA ──
    st.subheader("📈 Análise Exploratória dos Dados")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Distribuição por Modalidade**")
        exibirImagem("plot_distribuicao_categorias.png")
    with col_b:
        st.markdown("**Valor Estimado por Modalidade**")
        exibirImagem("plot_valor_estimado.png")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Top 10 Estados com Mais Editais**")
        exibirImagem("plot_distribuicao_uf.png")
    with col_d:
        st.markdown("**Top 15 Órgãos Compradores**")
        exibirImagem("plot_top_orgaos.png")

    st.markdown("**Evolução Temporal por Modalidade**")
    exibirImagem("plot_evolucao_temporal.png")

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("**Nuvem de Palavras**")
        exibirImagem("wordcloud_editais.png")
    with col_f:
        st.markdown("**Distribuição do Tamanho dos Textos**")
        exibirImagem("plot_tamanho_textos.png")

    st.divider()

    # ── Insights gerenciais ──
    st.subheader("💡 Insights Gerenciais")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background-color:#1a3a5c;padding:20px;border-radius:10px;height:180px">
        <h4 style="color:#7eb8f7">📦 Volume ≠ Valor</h4>
        <p style="color:#ddeeff;font-size:14px">
        O Pregão é a modalidade mais frequente, mas concentra os menores
        contratos individuais. A Concorrência Eletrônica representa menos
        editais, porém reúne os maiores valores.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color:#1a3a5c;padding:20px;border-radius:10px;height:180px">
        <h4 style="color:#7eb8f7">🗺️ Concentração Geográfica</h4>
        <p style="color:#ddeeff;font-size:14px">
        SP, MG, PR e RS concentram mais de 50% dos editais publicados.
        Cobrir apenas 4 estados já alcança metade do mercado nacional
        de licitações públicas.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color:#1a3a5c;padding:20px;border-radius:10px;height:180px">
        <h4 style="color:#7eb8f7">🏥 Domínio da Saúde</h4>
        <p style="color:#ddeeff;font-size:14px">
        O setor de saúde domina entre os maiores compradores públicos.
        Hospitais, secretarias de saúde e unidades de atenção básica
        são os clientes mais ativos no período analisado.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Desempenho dos modelos ──
    st.subheader("🤖 Desempenho Comparativo dos Modelos")
    st.caption(
        "Avaliação sobre amostra estratificada de 30 editais (10 por classe principal). "
        "Baseline ML avaliado sobre conjunto de teste completo (30% dos dados)."
    )

    tabela_modelos = pd.DataFrame([
        {"Modelo": "Logistic Regression",  "F1 Macro": 0.6471, "Precision": "—",    "Recall": "—",    "Accuracy": 0.91},
        {"Modelo": "Linear SVM",           "F1 Macro": 0.6020, "Precision": "—",    "Recall": "—",    "Accuracy": "—"},
        {"Modelo": "LLM zero-shot (Groq)", "F1 Macro": "—",    "Precision": "—",    "Recall": "—",    "Accuracy": "—"},
        {"Modelo": "LLM few-shot (Groq)",  "F1 Macro": "—",    "Precision": "—",    "Recall": "—",    "Accuracy": "—"},
    ])

    # tenta carregar métricas reais dos resultados salvos
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        def _mapearModalidade(m):
            mapa = {
                "pregão": "Pregão - Eletrônico",
                "concorrência": "Concorrência - Eletrônica",
                "dispensa": "Dispensa de Licitação",
                "inexigibilidade": "Inexigibilidade",
            }
            return mapa.get(str(m).lower(), None) if m else None

        def _extrairPrevisoes(df_res):
            y_real, y_pred = [], []
            for _, r in df_res.iterrows():
                if pd.isna(r.get("modalidade")):
                    continue
                pred = _mapearModalidade(r["modalidade"])
                if pred is None:
                    continue
                y_real.append(r["categoria_real"])
                y_pred.append(pred)
            return y_real, y_pred

        df_zs = pd.read_parquet(PROCESSED_PATH / "resultados_zero_shot.parquet")
        df_fs = pd.read_parquet(PROCESSED_PATH / "resultados_few_shot.parquet")

        y_real_zs, y_pred_zs = _extrairPrevisoes(df_zs)
        y_real_fs, y_pred_fs = _extrairPrevisoes(df_fs)

        if y_pred_zs:
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM zero-shot (Groq)", "F1 Macro"]  = round(f1_score(y_real_zs, y_pred_zs, average="macro", zero_division=0), 4)
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM zero-shot (Groq)", "Precision"] = round(precision_score(y_real_zs, y_pred_zs, average="macro", zero_division=0), 4)
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM zero-shot (Groq)", "Recall"]    = round(recall_score(y_real_zs, y_pred_zs, average="macro", zero_division=0), 4)
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM zero-shot (Groq)", "Accuracy"]  = round(accuracy_score(y_real_zs, y_pred_zs), 4)

        if y_pred_fs:
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM few-shot (Groq)", "F1 Macro"]  = round(f1_score(y_real_fs, y_pred_fs, average="macro", zero_division=0), 4)
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM few-shot (Groq)", "Precision"] = round(precision_score(y_real_fs, y_pred_fs, average="macro", zero_division=0), 4)
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM few-shot (Groq)", "Recall"]    = round(recall_score(y_real_fs, y_pred_fs, average="macro", zero_division=0), 4)
            tabela_modelos.loc[tabela_modelos["Modelo"] == "LLM few-shot (Groq)", "Accuracy"]  = round(accuracy_score(y_real_fs, y_pred_fs), 4)

    except Exception:
        pass  # mantém os valores "—" se os arquivos não existirem

    st.dataframe(tabela_modelos, use_container_width=True, hide_index=True)

    # tabela de custo
    st.markdown("**Estimativa de Custo por Modelo — 1.000 editais**")
    caminho_custo = PROCESSED_PATH / "tabela_custo_modelos.png"
    if caminho_custo.exists():
        exibirImagem("tabela_custo_modelos.png")
    else:
        tabela_custo = pd.DataFrame([
            {"Modelo": "llama-3.3-70b-versatile (Groq)",         "Tokens entrada (média)": 369.5, "Tokens saída (média)": 68.5, "Custo total / 1.000 editais (USD)": "$0.27"},
            {"Modelo": "deepseek-r1-distill-llama-70b (Groq)",   "Tokens entrada (média)": 369.5, "Tokens saída (média)": 68.5, "Custo total / 1.000 editais (USD)": "$0.35"},
            {"Modelo": "gemini-2.5-flash (Google)",              "Tokens entrada (média)": 369.5, "Tokens saída (média)": 68.5, "Custo total / 1.000 editais (USD)": "$0.28"},
        ])
        st.dataframe(tabela_custo, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PÁGINA 2 — CLASSIFICADOR DE EDITAIS
# ══════════════════════════════════════════════
elif pagina == "🔍 Classificador de Editais":

    st.title("🔍 Classificador de Editais")
    st.markdown(
        "Ferramenta operacional para analisar editais específicos. "
        "Cole o texto do edital para classificar ou faça uma busca semântica na base."
    )
    st.divider()

    # ── Seção 1: Classificação ──
    st.subheader("🤖 Classificação Automática")
    st.markdown("Cole o objeto do edital abaixo para classificar com ML e extrair informações com o LLM.")

    texto_input = st.text_area(
        "Texto do edital (campo objetoCompra)",
        height=150,
        placeholder="Ex: Aquisição de medicamentos e insumos hospitalares para a Secretaria Municipal de Saúde...",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        btn_classificar = st.button("🔎 Classificar", type="primary", use_container_width=True)

    if btn_classificar:
        if not texto_input.strip():
            st.warning("Cole o texto do edital antes de classificar.")
        else:
            # ── Classificação ML ──
            with st.spinner("Classificando com Logistic Regression..."):
                try:
                    from src.modelos import classificarComML
                    resultado_ml = classificarComML(texto_input)

                    st.success("✅ Classificação ML concluída")
                    col_ml1, col_ml2 = st.columns(2)
                    col_ml1.metric("Categoria predita", resultado_ml["categoria"])
                    col_ml2.metric("Confiança do modelo", f"{resultado_ml['confianca']:.1%}")

                except Exception as e:
                    st.error(f"Erro na classificação ML: {e}")

            # ── Classificação LLM ──
            with st.spinner("Extraindo detalhes com LLaMA 70B (Groq)... pode levar alguns segundos"):
                try:
                    from src.llm import classificarEdital
                    resultado_llm = classificarEdital(texto_input)

                    if resultado_llm:
                        st.success("✅ Extração LLM concluída")
                        st.markdown("**Campos extraídos pelo LLM:**")

                        col_l1, col_l2, col_l3 = st.columns(3)
                        col_l1.metric("Categoria do objeto", resultado_llm.categoria_objeto)
                        col_l2.metric("Modalidade",          resultado_llm.modalidade.capitalize())
                        col_l3.metric(
                            "Valor estimado",
                            f"R$ {resultado_llm.valor_estimado:,.2f}" if resultado_llm.valor_estimado else "Não informado"
                        )

                        col_l4, col_l5 = st.columns(2)
                        col_l4.metric(
                            "Alerta de risco",
                            badgeRisco(resultado_llm.alerta_risco)
                        )
                        col_l5.metric(
                            "Oportunidade recomendada",
                            "✅ Sim" if resultado_llm.oportunidade_recomendada else "❌ Não"
                        )

                        if resultado_llm.requisitos_habilitacao:
                            st.markdown("**Requisitos de habilitação identificados:**")
                            for req in resultado_llm.requisitos_habilitacao:
                                st.markdown(f"- {req}")
                        else:
                            st.info("Nenhum requisito de habilitação identificado no texto.")

                    else:
                        st.warning(
                            "O LLM não conseguiu extrair os campos nesta tentativa. "
                            "Tente novamente ou verifique se o texto é suficientemente detalhado."
                        )

                except Exception as e:
                    st.error(f"Erro na classificação LLM: {e}")

    st.divider()

    # ── Seção 2: Busca Semântica RAG ──
    st.subheader("🔎 Busca Semântica (RAG)")
    st.markdown("Digite uma pergunta para buscar editais similares na base e receber uma resposta do LLM.")

    # exemplos sugeridos
    with st.expander("💡 Exemplos de perguntas"):
        st.markdown("""
        - Quais editais de saúde estão disponíveis?
        - Editais de TI em São Paulo
        - Contratos de limpeza e conservação
        - Quais pregões envolvem obras de construção civil?
        - Editais de compra de veículos ou equipamentos de transporte
        """)

    pergunta_input = st.text_input(
        "Sua pergunta",
        placeholder="Ex: Quais editais de saúde estão disponíveis?",
    )

    col_corpus, col_btn_rag = st.columns([3, 1])
    with col_corpus:
        corpus_escolhido = st.radio(
            "Buscar em:",
            ["📚 Corpus histórico (3.009 editais)", "🔴 Editais abertos agora (simulado)"],
            horizontal=True,
        )
    with col_btn_rag:
        btn_buscar = st.button("🔎 Buscar", type="primary", use_container_width=True)

    if btn_buscar:
        if not pergunta_input.strip():
            st.warning("Digite uma pergunta antes de buscar.")
        else:
            usar_abertos = "abertos" in corpus_escolhido

            with st.spinner("Buscando editais similares..."):
                try:
                    from src.rag import buscarEditaisSimilares, responderComRAG

                    # ── Editais encontrados ──
                    editais = buscarEditaisSimilares(
                        pergunta_input,
                        usar_abertos=usar_abertos,
                        n_resultados=5
                    )

                    if not editais:
                        st.warning(
                            "Nenhum edital encontrado. "
                            "Verifique se o ChromaDB foi indexado (rode o Notebook 3)."
                        )
                    else:
                        st.markdown(f"**{len(editais)} editais mais similares encontrados:**")

                        for i, edital in enumerate(editais, 1):
                            meta  = edital["metadados"]
                            valor = meta.get("valor_estimado", 0)
                            valor_fmt = f"R$ {valor:,.2f}" if valor else "Não informado"

                            with st.expander(f"Edital {i} — {meta.get('modalidade', '—')} | {meta.get('uf', '—')}"):
                                st.markdown(f"**Texto:** {edital['texto']}")
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Modalidade",      meta.get("modalidade", "—"))
                                c2.metric("Órgão",           (meta.get("orgao", "—") or "—")[:30])
                                c3.metric("UF",              meta.get("uf", "—"))
                                c4.metric("Valor estimado",  valor_fmt)
                                st.caption(f"Distância semântica: {edital['distancia']}")

                        # ── Resposta RAG ──
                        st.markdown("---")
                        st.markdown("**💬 Resposta gerada pelo LLM com base nos editais encontrados:**")

                        with st.spinner("Gerando resposta com LLaMA 70B..."):
                            resultado_rag = responderComRAG(pergunta_input, usar_abertos=usar_abertos)
                            st.info(resultado_rag["resposta"])

                except Exception as e:
                    st.error(f"Erro na busca semântica: {e}")
                    st.caption(
                        "Dica: verifique se o ChromaDB foi indexado rodando o Notebook 3 "
                        "e se a GROQ_API_KEY está no arquivo .env"
                    )
import os
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.embeddings import gerarEmbedding

load_dotenv()

PROCESSED_PATH  = Path("dados/processed")
CHROMA_PATH     = PROCESSED_PATH / "chroma_db"
MODELO_PRINCIPAL = "llama-3.3-70b-versatile"

# cache das coleções
_cliente_chroma     = None
_colecao_historico  = None
_colecao_abertos    = None


def _criarClienteGroq() -> OpenAI:
    chave = os.getenv("GROQ_API_KEY")
    if not chave:
        raise ValueError("GROQ_API_KEY não encontrada no .env")
    return OpenAI(api_key=chave, base_url="https://api.groq.com/openai/v1")


def _carregarColecoes():
    """Abre as coleções do ChromaDB. Usa cache para não reabrir a cada chamada."""
    global _cliente_chroma, _colecao_historico, _colecao_abertos

    if _colecao_historico is not None:
        return _colecao_historico, _colecao_abertos

    _cliente_chroma    = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _colecao_historico = _cliente_chroma.get_or_create_collection(
        name="editais_historico",
        metadata={"hnsw:space": "cosine"}
    )
    _colecao_abertos   = _cliente_chroma.get_or_create_collection(
        name="editais_abertos",
        metadata={"hnsw:space": "cosine"}
    )
    return _colecao_historico, _colecao_abertos


def buscarEditaisSimilares(pergunta: str,
                           usar_abertos: bool = False,
                           n_resultados: int = 5) -> list:
    """
    Gera embedding da pergunta e busca os editais mais similares no ChromaDB.

    Parâmetros:
        pergunta      — texto da pergunta do usuário
        usar_abertos  — True para buscar na coleção de editais abertos,
                        False para o corpus histórico
        n_resultados  — quantidade de resultados a retornar

    Retorna:
        Lista de dicts com texto, metadados e distância de cada resultado.
    """
    colecao_historico, colecao_abertos = _carregarColecoes()
    colecao = colecao_abertos if usar_abertos else colecao_historico

    if colecao.count() == 0:
        return []

    embedding_pergunta = gerarEmbedding(pergunta)

    resultados = colecao.query(
        query_embeddings=[embedding_pergunta],
        n_results=min(n_resultados, colecao.count()),
        include=["documents", "metadatas", "distances"]
    )

    editais_encontrados = []
    for i in range(len(resultados["documents"][0])):
        editais_encontrados.append({
            "texto":     resultados["documents"][0][i],
            "metadados": resultados["metadatas"][0][i],
            "distancia": round(resultados["distances"][0][i], 4),
        })

    return editais_encontrados


def responderComRAG(pergunta: str, usar_abertos: bool = False) -> dict:
    """
    Recupera editais relevantes e usa o Groq para gerar uma resposta
    fundamentada nesses editais.

    Parâmetros:
        pergunta     — pergunta do usuário
        usar_abertos — True para buscar na coleção de editais abertos

    Retorna:
        Dict com 'pergunta', 'resposta' e 'editais_fonte'.
    """
    editais = buscarEditaisSimilares(pergunta, usar_abertos=usar_abertos, n_resultados=5)

    if not editais:
        return {
            "pergunta":      pergunta,
            "resposta":      "Nenhum edital encontrado na base para responder a essa pergunta.",
            "editais_fonte": [],
        }

    # monta contexto com os editais recuperados
    contexto = ""
    for i, e in enumerate(editais, 1):
        meta = e["metadados"]
        valor = meta.get("valor_estimado", 0)
        valor_fmt = f"R$ {valor:,.2f}" if valor else "não informado"
        contexto += (
            f"\n--- Edital {i} ---\n"
            f"Texto: {e['texto']}\n"
            f"Modalidade: {meta.get('modalidade', 'não informado')}\n"
            f"UF: {meta.get('uf', 'não informado')}\n"
            f"Órgão: {meta.get('orgao', 'não informado')}\n"
            f"Valor estimado: {valor_fmt}\n"
        )

    prompt = (
        "Você é um assistente especializado em licitações públicas brasileiras.\n"
        "Com base SOMENTE nos editais abaixo, responda à pergunta do usuário.\n"
        "Se os editais não tiverem a informação, diga que não encontrou.\n\n"
        f"PERGUNTA: {pergunta}\n\n"
        f"EDITAIS RECUPERADOS:{contexto}\n\n"
        "RESPOSTA:"
    )

    try:
        cliente       = _criarClienteGroq()
        resposta_api  = cliente.chat.completions.create(
            model=MODELO_PRINCIPAL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        texto_resposta = resposta_api.choices[0].message.content.strip()

    except Exception as e:
        texto_resposta = f"Erro ao chamar o modelo: {e}"

    return {
        "pergunta":      pergunta,
        "resposta":      texto_resposta,
        "editais_fonte": editais,
    }
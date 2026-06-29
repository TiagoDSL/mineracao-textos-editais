from sentence_transformers import SentenceTransformer

# carrega uma vez e reutiliza (evita recarregar a cada chamada)
_modelo_embedding = None


def carregarModeloEmbedding() -> SentenceTransformer:
    """
    Carrega o modelo MiniLM multilingual.
    Usa cache para não recarregar a cada chamada.
    """
    global _modelo_embedding
    if _modelo_embedding is None:
        _modelo_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _modelo_embedding


def gerarEmbedding(texto: str) -> list:
    """
    Gera o embedding de um único texto.
    Retorna uma lista de floats.
    """
    modelo = carregarModeloEmbedding()
    return modelo.encode([texto])[0].tolist()
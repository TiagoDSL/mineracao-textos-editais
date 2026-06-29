import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.preprocessamento import limparTexto
from src.embeddings import carregarModeloEmbedding

# caminhos padrão dos artefatos salvos pelo notebook 1
PROCESSED_PATH = Path("dados/processed")

# cache do modelo em memória
_modelo_lr = None
_modelo_embedding = None


def carregarModeloLR() -> LogisticRegression:
    """
    Reconstrói a Logistic Regression treinando com os embeddings salvos.
    Usa cache para não retreinar a cada chamada.
    """
    global _modelo_lr

    if _modelo_lr is not None:
        return _modelo_lr

    X_treino = np.load(PROCESSED_PATH / "X_treino.npy")
    y_treino = np.load(PROCESSED_PATH / "y_treino.npy", allow_pickle=True)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_treino, y_treino)

    _modelo_lr = lr
    return _modelo_lr


def classificarComML(texto: str) -> dict:
    """
    Classifica um texto usando a Logistic Regression treinada.

    Retorna dict com:
        categoria   — categoria predita
        confianca   — probabilidade da classe predita (0 a 1)
    """
    modelo_lr       = carregarModeloLR()
    modelo_emb      = carregarModeloEmbedding()

    texto_limpo     = limparTexto(texto)
    embedding       = modelo_emb.encode([texto_limpo])

    categoria       = modelo_lr.predict(embedding)[0]
    probabilidades  = modelo_lr.predict_proba(embedding)[0]
    confianca       = float(max(probabilidades))

    return {
        "categoria": categoria,
        "confianca": round(confianca, 4),
    }
import re
import nltk
from nltk.corpus import stopwords

# garante que os stopwords estão disponíveis
try:
    stop_words = set(stopwords.words("portuguese"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("portuguese"))


def limparTexto(texto: str) -> str:
    """
    Limpa e normaliza o texto de um edital.
    Remove pontuação, números e stopwords. Mantém apenas palavras com mais de 2 letras.
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúâêîôûãõàèìòùç\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.schema import EditalClassificado

load_dotenv()

MODELO_PRINCIPAL = "llama-3.3-70b-versatile"


def _criarCliente() -> OpenAI:
    """Cria e retorna o cliente Groq."""
    chave = os.getenv("GROQ_API_KEY")
    if not chave:
        raise ValueError("GROQ_API_KEY não encontrada no .env")
    return OpenAI(api_key=chave, base_url="https://api.groq.com/openai/v1")


def _carregarPrompt(nome_arquivo: str) -> str:
    """Lê um arquivo de prompt da pasta /prompts."""
    caminho = Path("prompts") / nome_arquivo
    return caminho.read_text(encoding="utf-8")


def _montarPrompt(template: str, texto: str) -> str:
    """Substitui {texto} no template pelo conteúdo do edital."""
    return template.replace("{texto}", texto)


def classificarEdital(texto: str) -> Optional[EditalClassificado]:
    """
    Classifica um edital usando o LLM (Groq LLaMA 70B) e valida com Pydantic.

    Fluxo:
        Tentativa 1 — prompt few-shot normal
        Tentativa 2 — retry com instrução adicional se validação falhar
        Falha       — retorna None

    Parâmetros:
        texto — texto original do campo objetoCompra

    Retorna:
        EditalClassificado validado ou None em caso de falha
    """
    try:
        prompt_template = _carregarPrompt("prompt_few_shot.txt")
    except FileNotFoundError:
        try:
            prompt_template = _carregarPrompt("prompt_zero_shot.txt")
        except FileNotFoundError:
            # fallback inline se os arquivos de prompt não existirem
            prompt_template = (
                "Você é um especialista em licitações públicas brasileiras.\n"
                "Analise o texto do edital abaixo e responda SOMENTE com JSON puro.\n"
                "O JSON deve conter exatamente estes campos:\n"
                "- categoria_objeto (string)\n"
                "- modalidade (pregão|concorrência|dispensa|inexigibilidade)\n"
                "- valor_estimado (número ou null)\n"
                "- requisitos_habilitacao (lista de strings, vazia se não houver)\n"
                "- alerta_risco (baixo|médio|alto)\n"
                "- oportunidade_recomendada (true|false)\n\n"
                "TEXTO DO EDITAL:\n{texto}\n\n"
                "JSON:"
            )

    prompt_retry = (
        "<instrucao_adicional>\n"
        "Sua resposta anterior não estava no formato correto.\n"
        "Responda SOMENTE com JSON puro, sem texto antes ou depois, "
        "sem blocos de código, sem markdown, sem explicações.\n"
        "O JSON deve conter exatamente estes campos:\n"
        "categoria_objeto (string), "
        "modalidade (pregão|concorrência|dispensa|inexigibilidade), "
        "valor_estimado (número ou null), "
        "requisitos_habilitacao (lista de strings), "
        "alerta_risco (baixo|médio|alto), "
        "oportunidade_recomendada (true|false).\n"
        "</instrucao_adicional>\n\n"
    )

    try:
        cliente = _criarCliente()
    except ValueError as e:
        print(f"[LLM] Erro ao criar cliente: {e}")
        return None

    for tentativa in range(1, 3):
        try:
            if tentativa == 1:
                prompt_final = _montarPrompt(prompt_template, texto)
            else:
                prompt_final = prompt_retry + _montarPrompt(prompt_template, texto)

            resposta = cliente.chat.completions.create(
                model=MODELO_PRINCIPAL,
                messages=[{"role": "user", "content": prompt_final}],
                temperature=0,
            )

            texto_resposta = resposta.choices[0].message.content.strip()

            # remove blocos de código markdown se o modelo incluir
            if texto_resposta.startswith("```"):
                linhas = texto_resposta.splitlines()
                linhas = [l for l in linhas if not l.startswith("```")]
                texto_resposta = "\n".join(linhas).strip()

            dados = json.loads(texto_resposta)
            resultado = EditalClassificado(**dados)
            return resultado

        except ValidationError as e:
            print(f"[LLM tentativa {tentativa}] Erro de validação Pydantic: {e}")
            if tentativa == 2:
                return None

        except json.JSONDecodeError as e:
            print(f"[LLM tentativa {tentativa}] JSON inválido: {e}")
            if tentativa == 2:
                return None

        except Exception as e:
            print(f"[LLM tentativa {tentativa}] Erro inesperado: {e}")
            if tentativa == 2:
                return None

        time.sleep(1.5)

    return None
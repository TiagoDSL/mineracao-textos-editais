
# Schema Pydantic para classificação de editais de licitação 

# Cada edital processado pelo LLM retorna um objeto deste schema com os campos extraídos e 
# classificados automaticamente


from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EditalClassificado(BaseModel):
   
    # Representa as informações extraídas de um edital de licitação
    # Preenchido pelo LLM a partir do texto do edital
   

    categoria_objeto: str = Field(
        description=(
            "Categoria do objeto licitado, descrita em linguagem simples. "
            "Exemplos: 'serviços de limpeza', 'aquisição de equipamentos de TI', "
            "'obras de reforma'. Extrair do texto, sem inventar."
        )
    )

    modalidade: Literal["pregão", "concorrência", "dispensa", "inexigibilidade"] = Field(
        description=(
            "Modalidade do processo licitatório. "
            "Usar 'pregão' para Pregão Eletrônico ou Presencial. "
            "Usar 'concorrência' para Concorrência Eletrônica ou Presencial. "
            "Usar 'dispensa' para Dispensa de Licitação. "
            "Usar 'inexigibilidade' para Inexigibilidade de Licitação."
        )
    )

    valor_estimado: Optional[float] = Field(
        default=None,
        description=(
            "Valor estimado do contrato em reais (R$), como número decimal. "
            "Extrair do texto quando disponível. "
            "Retornar null se o valor não estiver informado no edital."
        )
    )

    requisitos_habilitacao: List[str] = Field(
        description=(
            "Lista de exigências para o fornecedor participar da licitação. "
            "Exemplos: 'certidão negativa de débitos', 'registro no CREA', "
            "'balanço patrimonial dos últimos 2 anos'. "
            "Listar apenas os requisitos explícitos no texto. "
            "Retornar lista vazia se não houver requisitos listados."
        )
    )

    alerta_risco: Literal["baixo", "médio", "alto"] = Field(
        description=(
            "Nível de risco do edital para um fornecedor participar. "
            "'baixo': objeto simples, valor pequeno, poucos requisitos. "
            "'médio': requisitos moderados ou valor significativo. "
            "'alto': muitos requisitos técnicos, valor alto, prazo curto, "
            "ou objeto complexo."
        )
    )

    oportunidade_recomendada: bool = Field(
        description=(
            "Indica se o edital é uma boa oportunidade para participar. "
            "True se o objeto é comum, os requisitos são acessíveis e o "
            "valor justifica a participação. "
            "False se os requisitos são muito restritivos, o valor é baixo "
            "ou o objeto é muito especializado."
        )
    )
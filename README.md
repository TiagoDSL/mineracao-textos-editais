# Classificação de Editais de Licitação

Sistema de classificação automática de editais de licitação utilizando NLP, Machine Learning, LLMs e RAG com saída estruturada via Pydantic.

Projeto desenvolvido para a disciplina de Mineração de Textos da especialização em Data Science e Inteligência Artificial.

---

# Objetivo

Empresas fornecedoras precisam monitorar diariamente milhares de oportunidades publicadas em portais de compras públicas.

O objetivo deste projeto é automatizar a análise inicial desses editais, classificando oportunidades e extraindo informações relevantes por meio de técnicas de Processamento de Linguagem Natural (NLP), Machine Learning e Large Language Models (LLMs).

---

# Problema de Negócio

Uma empresa fornecedora deseja identificar rapidamente oportunidades de participação em licitações públicas.

O processo manual de leitura e triagem de editais é lento, repetitivo e pouco escalável.

O sistema desenvolvido busca:

* Classificar automaticamente editais;
* Extrair informações relevantes;
* Estruturar os dados para análise;
* Permitir busca semântica de oportunidades semelhantes;
* Apoiar a tomada de decisão comercial.

---

# Fonte dos Dados

Os dados foram coletados da API pública do Portal Nacional de Contratações Públicas (PNCP).

Endpoint utilizado:

```text
/api/consulta/v1/contratacoes/publicacao
```

Período coletado:

```text
01/06/2026 a 03/06/2026
```

Total de registros:

```text
3.009 editais
```

---

# Tecnologias Utilizadas

## Linguagem

* Python 3.11

## NLP e Machine Learning

* Pandas
* NumPy
* NLTK
* Sentence Transformers
* Scikit-Learn

## LLM

* Groq
* Llama 3.3 70B
* Pydantic

## RAG

* ChromaDB

## Visualização

* Matplotlib
* Seaborn
* WordCloud

## Interface

* Streamlit

---

# Estrutura do Projeto

```text
projeto_nlp/
├── .gitignore
├── DECISOES.md
├── README.md
├── requirements.txt
├── dados/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 1_dados_eda_ml.ipynb
│   ├── 2_llm_pydantic.ipynb
│   └── 3_rag_erros.ipynb
├── prompts/
└── src/
    ├── __init__.py
    ├── app.py
    ├── embeddings.py
    ├── llm.py
    ├── modelos.py
    ├── preprocessamento.py
    ├── rag.py
    └── schema.py
```

---

# Metodologia

O projeto foi dividido em três etapas principais.

## 1. Coleta e Análise Exploratória

* Coleta automática via API do PNCP;
* Tratamento de rate limit;
* Análise exploratória dos dados;
* Visualizações estatísticas;
* Pré-processamento textual.

## 2. Machine Learning

Geração de embeddings utilizando:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Modelos avaliados:

* Logistic Regression
* Linear SVM

Métrica principal:

```text
F1 Macro
```

## 3. LLM + Pydantic

Classificação utilizando:

* Zero-Shot
* Few-Shot

Saída estruturada através de schema Pydantic contendo:

* Categoria do objeto;
* Modalidade;
* Valor estimado;
* Requisitos de habilitação;
* Alerta de risco;
* Recomendação de oportunidade.

## 4. RAG

Implementação de busca semântica utilizando:

* Embeddings MiniLM
* ChromaDB

Permite recuperar editais semelhantes a partir de perguntas em linguagem natural.

---

# Principais Resultados

## Distribuição das Categorias

* Pregão Eletrônico: 63%
* Concorrência Eletrônica: 35%
* Concorrência Presencial: 2%

## Desempenho dos Modelos

| Modelo              | F1 Macro |
| ------------------- | -------- |
| Logistic Regression | 0.6471   |
| Linear SVM          | 0.6020   |

A Logistic Regression apresentou o melhor desempenho e foi adotada como baseline para comparação com os modelos baseados em LLM.

---

# Insights Obtidos

### 1. Volume não é igual a valor

O Pregão Eletrônico concentra o maior número de oportunidades, porém os menores valores médios por contrato.

Já as Concorrências concentram contratos de maior valor.

### 2. Forte concentração geográfica

São Paulo, Minas Gerais, Paraná e Rio Grande do Sul representam mais da metade dos editais coletados.

### 3. Saúde pública domina a demanda

Entre os principais órgãos compradores predominam secretarias de saúde, hospitais e fundos municipais de saúde.

---

# Como Executar

## Clonar o projeto

```bash
git clone https://github.com/TiagoDSL/mineracao-textos-editais.git
```

## Criar ambiente virtual

```bash
python -m venv .venv
```

## Ativar ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Executar os notebooks

```text
notebooks/1_dados_eda_ml.ipynb
notebooks/2_llm_pydantic.ipynb
notebooks/3_rag_erros.ipynb
```

## Executar a interface

```bash
streamlit run src/app.py
```

---

# Autor

Tiago Leite

Projeto acadêmico desenvolvido para a disciplina de Mineração de Textos.



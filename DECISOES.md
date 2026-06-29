# Decisões

## Notebooks

Para evitar um único notebook muito grande e facilitar a organização do projeto, optei por dividir o trabalho em três etapas:

* **1_dados_eda_ml.ipynb:** coleta dos dados via API do PNCP, análise exploratória, pré-processamento textual, geração de embeddings, divisão dos dados em treino e teste, treinamento e avaliação dos modelos Logistic Regression e Linear SVM.
* **2_llm_pydantic.ipynb:** definição do schema Pydantic, versionamento dos prompts, tratamento de falhas de validação, classificação zero-shot e few-shot utilizando LLM e comparação com o baseline de Machine Learning.
* **3_rag_erros.ipynb:** indexação vetorial com ChromaDB, busca semântica via RAG, avaliação de relevância, análise estruturada de erros e estimativa de custos.

---

## Split dos Dados

Dividi o dataset em treino (70%) e teste (30%), utilizando o parâmetro `stratify` para preservar a proporção das classes em ambos os conjuntos. O `random_state=42` foi utilizado para garantir a reprodutibilidade dos resultados.

Não foi criado um conjunto de validação separado, pois a comparação dos modelos foi realizada diretamente sobre o conjunto de teste.

---

## Escolha dos Modelos de Machine Learning

Foram avaliados dois modelos sobre os embeddings gerados pelo MiniLM:

* Logistic Regression
* Linear SVM

A escolha foi motivada pelo fato de ambos serem modelos lineares eficientes para dados representados por embeddings de alta dimensão.

No caso do Linear SVM, foi utilizado o `CalibratedClassifierCV` para permitir a obtenção de probabilidades por classe, recurso útil para futuras integrações na interface Streamlit.

---

## Resultados da Modelagem

A Logistic Regression apresentou **F1 Macro de 0,6471**, enquanto o Linear SVM obteve **F1 Macro de 0,6020**.

Os dois modelos apresentaram bom desempenho nas classes majoritárias, mas tiveram dificuldade na classe minoritária **Concorrência Presencial**, que representa apenas 2% do corpus.

Esse comportamento era esperado devido ao desbalanceamento natural dos dados, que foi preservado propositalmente para representar o cenário real do mercado de licitações.

---

## Modelo Selecionado

A Logistic Regression foi definida como modelo principal do projeto por apresentar o melhor F1 Macro entre os modelos avaliados.

O valor obtido foi adotado como baseline oficial para comparação com as abordagens baseadas em LLM desenvolvidas no Notebook 2.

---

## Avaliação com LLM

Para a avaliação dos prompts zero-shot e few-shot foi utilizada uma amostra balanceada das três categorias presentes no conjunto de teste.

A quantidade de exemplos foi limitada pela disponibilidade da classe minoritária **Concorrência Presencial** e pelas restrições de uso da API gratuita utilizada durante os experimentos.

A estratégia permitiu realizar uma comparação consistente entre abordagens supervisionadas e modelos de linguagem, mantendo representatividade das classes avaliadas e garantindo viabilidade operacional do experimento.

---

## RAG

A camada de RAG foi construída utilizando os embeddings já gerados anteriormente e armazenados em uma base vetorial ChromaDB.

O objetivo foi simular um cenário real de busca de oportunidades em licitações públicas, permitindo recuperar editais semanticamente semelhantes a partir de consultas em linguagem natural.

A avaliação foi realizada por meio de perguntas de teste e análise manual da relevância dos documentos recuperados.

---

## Análise de Erros

As falhas observadas durante os experimentos foram categorizadas em diferentes grupos, incluindo:

* Erros de classificação;
* Ambiguidades dos dados;
* Ruídos textuais;
* Falhas de validação do schema;
* Limitações inerentes aos modelos de linguagem.

Para cada categoria foi registrada uma hipótese de causa e uma possível ação corretiva, permitindo documentar limitações do sistema e oportunidades de evolução futura.

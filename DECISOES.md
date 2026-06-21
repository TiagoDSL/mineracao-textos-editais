# Decisões

## Corpus
Escolhido o corpus de editais de licitação do PNCP (Portal Nacional de 
Contratações Públicas) por ter API pública bem documentada e dados já 
estruturados com categorização.

## Ambiente
Python 3.11, ambiente virtual isolado via venv para garantir reprodutibilidade.

## Notebooks
Para não ficar um unico notebook muito grande, decidir dividir em 3, coforme etapas. Ficando da seguinte forma:
- 1_dados_eda_ML.ipynb: coleta dos dados via API do site do PNCP, EDA, pré-processamento dos dados, embeddings, divisão dos dados 15/15/70 e modelo basline (Logistic Regression)+ Randon Forest e LightGBM.
- 2_llm_pydantic.ipynb: schema pydantic, prompt versionado, tratamento de falha de validação, classificação zero/few-shot via Gemini e comparação.
- 3_rag_erros.ipynb: indexação no ChromaDB, busca semântica, avaliação de relevância, análise de erros categorizada e estimativa de custos.

## Dados
### Fonte de Dados
Utilizado o endpoint /v1/contratacoes/publicacao da API de Consulta do PNCP, que não exige autenticação e permite filtrar por data e modalidade de contratação. Escolhido por ser o endpoint que retorna o objeto da contratação em texto livre, base para a tarefa de classificação por NLP do projeto.

### API
Defini o tamamnhoPagina=50,  a documentaçao indica até 500 registros por página, mas depois de fazer alguns testes a API rejeita qualquer valor acima de 50.

### Escolha das Categorias
Selecionei apenas as modalidades relevantes do ponto de vista de uma empresa fornecedora: 
- Pregão Eletrônico (6);
- Pregão Presencial (7);
- Concorrência Eletrônica (4);
- Concorrência Presencial (5); 
- Dispensa de Licitação (8);
- Inexigibilidade (9). 

Excluídas modalidades de venda de bens públicos (Leilão) e processos preparatórios sem contrato direto (Manifestação de Interesse, Pré-qualificação, Credenciamento), por não 
representarem oportunidade de negócio para uma empresa interessada.

### Estratégia de coleta de dados
Resolvi utilizar o endpoint /contratacoes/publicacao (histórico) para coletar o corpus de treino do classificador, garantindo volume e diversidade. Para a camada de RAG e a interface de demonstração, utilizei o endpoint /contratacoes/proposta (propostas em aberto), simulando o caso de uso real de uma empresa que busca oportunidades de participação imediata.

### Escolhas das datas para coleta
Ao fazer um teste inicial para saber a quantidade de volume de dados que iriar vir, constatei que o volume era muito alto de editais por dia por ter escolhido 3 meses de janela. Então para não tem um volume extraordinário e sem essa necessidade, resovi continuar os testes reduzindo o tempo de 3 meses para 1 mes e deu um volume extraordinário, de 3 meses para 1 mês também foi muito alto, 1 mês para 15 dias também foi alto, 15 dias para 5 dias também foi alto e entao ficou reduzido para 3 dias. Sendo suficiente para superar o minimo exigido, economizar tempo de execução desnecessário e evitando o risco de rate limit da API. Foi entao decidido a coleta dos primeiros dias do mês de Junho de 2026, de 01-06-26 ate 03-06-26.

### Classificação dos dados
Como ja vem um rotulo de classificação, "modalidade_nome", sendo o tipo de processo jurídico e não o que esta sendo comprado. Optei por aproveitar e usar como "categoria", para rotular os dados no dataset bruto.

### Distribuição da categorias coletadas
A coleta resultou em apenas 3 das 6 modalidades pretendidas, pois o limite global de 3.000 registros foi atingido antes de alcançar as demais modalidades. A distribuição final ficou da seguinte forma: 
- Pregão Eletrônico 63%, 
- Concorrência Eletrônica 35%, 
- Concorrência Presencial 2% 
Ocorrendo um desbalanceamento natural do mercado, preservado propositalmente. A classe minoritária, Concorrência Presencial de 50 registros (2%), exigirá atenção no na separação estratificado e na escolha de métricas, como F1 macroe não apenas accuracy.
# Prevenção de Inadimplência de Empréstimos

Um dos principais problemas das instituições financeiras é balancear o risco de inadimplência. Por um lado a instituição quer diminuir o risco do cliente entrar em Default (não conseguir pagar os juros ou o principal de uma dívida no prazo acordado) pra não ter prejuízos. Por outro lado, quer diminuir a probabilidade de perder um cliente que seja bom pagador por conta de um algoritmo mal ajustado.

Nesse problema iremos tratar as variáveis e treinar um modelo de CatBoost com diferentes hiperparâmetros usando o Pipeline e o GridSearch do Sklearn pra deixar um código mais limpo e decidir qual modelo será usado pra ser testado. Após isso será feito um ajuste das faixas de score pra melhorar a decisão a ser tomada pela área de negócio.

## Objetivo

O principal objetivo do projeto é diminuir o **Risco de Inadimpência** sem perder muitos clientes potencialmente bons. Para isso, após o modelo ser treinado e validado, os clientes serão ordenados em faixas de acordo com seu score e após a análise das faixas, serão agrupados em 3 grupos: 

*Empréstimo Aprovado* - Clientes das faixas mais altas, serão aprovados automaticamente pelo sistema

*Empréstimo Negado* - Clientes das faixas mais baixas, terão seus empréstimos negados pelo sistema

*Enviar para um analista* - Clientes das faixas intermediárias, seus casos serão analisados por um analista, ou será sugerido alguma manobra pra aumentar o score, por exemplo, indicar um fiador, ou juntar renda com cônjuge.

O objetivo técnico foi usar o GridSearch e o Pipeline da lib Sklearn pra treinar modelos de machine learning de forma rápida e limpa.

## Fonte de Dados

Os dados são oriundos de uma competição no [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/) que tinha o objetivo de garantir que os clientes capazes de pagar não sejam rejeitados e que os empréstimos sejam concedidos a clientes que sejam capaz de pagar no prazo determinado. A métrica que eles utilizaram pra avaliar os melhores modelos foi a *Área da Curva ROC*.
Foram disponibilziados 2 arquivos principais que serão usados nesse projeto:
*application_train.csv* que será usado pra treinar e validar o modelo;
*application_test.csv* que originalmente na competição serviu pra avaliar os melhores modelos, mas aqui usei pra fazer predições.

Outros 7 arquivos auxiliares com dados históricos e de bureau dos clientes também foram disponibilizados mas não serão usados nesse projeto. 

**application_train.csv** 
- Dados estáticos para todas as aplicações. Uma linha representa um empréstimo na amostra de dados.
- 121 variáveis explicativas e 1 variável alvo
- 307511 registros

**application_test.csv**
- Dados estáticos para todas as aplicações. Uma linha representa um empréstimo na amostra de dados.
- 121 variáveis explicativas
- 61503 registros

## Arquivos

- *Treino com GridSearch e Teste.ipynb* é o código principal, onde contém o dataprep, o treino e os testes do modelo, o ajustes das faixas de scores e o teste pra regra de negócio;
- *Metadados.ipynb* é código onde é extraído o metadados que serão usados no código principal;
- *tela.py* é o código feito na lib **Streamlit* para construção da tela do sistema;
- A pasta *abt* contém os dados de treino e teste;
- A pasta *inputs* contém registros de clientes pra previsão (extraídos dos dados de teste);
- A pasta *outputs* contém todas as saídas geradas pelo código principal, incluindo o modelo serializado em .pkl;

# Metadados

Antes de começar a treinar o modelo, preparei os metadados, para fazer a preparação necessária para cada tipo de dados. a variável alvo (TARGET) indiquei que na role como 'target', o id dos clientes (SK_ID_CURR) indiquei na role como 'id'. Nas outras variáveis foram indicadas automaticamente pelo seu tipo, 'nominal' para as variáveis categóricas, 'ordinal' para as variáveis discretas e 'interval' para as variáveis contínuas. Também foi indicado, o tipo e a cardinalidade de cada variável

# Treino

## DataPrep

O primeiro passo foi separar a base para teste. Usei 20% dos dados da base para testes, usando o *StratifiedShuffleSplit* da Sklearn para manter a distribuição das classes da variável alvo.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/strat_split.png?raw=true)

Após o split, foi feito separado a variável alvo das variáveis explicativas

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/var_alvo.png?raw=true)

O próximo passo foi selecionar as variáveis numéricas, colocar em uma lista e salvar essa lista no arquivo *lista_vars_numericas.pkl*

O mesmo foi feito com as variáveis categóricas com cardinalidade <= 10, que foram salvas no arquivo *lista_vars_dummif.pkl*

Também foi feito com as variáveis categóricas com cardinalidade > 10 e foram salvas no arquivo *lista_vars_le.pkl*

Após alguns testes decidi não usar as variáveis de alta cardinalidade pois não acrescentavam muito ao modelo e uma limitação do *ColumnTransformer* do Sklearn que por algum motivo que eu desconheço só permitiu usar 2 transformadores, então optei pelos trasnformadores das variáveis numéricas e os trasnformadores das variáveis categóricas de baixa cardinalidade, como veremos a seguir.

## Pipeline de Treino do Modelo

Com os dados já preparados, chegou a hora de treinar o modelo, para isso usei o Pipeline da *Sklearn*. Primeiramente fiz um pipeline para as variáveis numéricas, na qual chamei de *num_transformer*. Nele pedi pra tratar os nulos com a média  de cada coluna usando *SimpleImputer* e após isso normalizar os dados usando *StandardScaler*.

Para as variáveis categóricas fiz outro pipeline chamada *ohc_transformer*. Pedi para o *ohc_transformer* tratar os nulos com a moda de cada coluna usando *SimpleImputer e após isso criar variáveis Dummys usando *OneHotEncoder* e dropando o primeiro dummy de cada classe para diminuir o processamento e a multicolinearidade (alguns autores discordam da multicolinearidade nesse caso, mas todos concordam que aumenta a performance do processamento).
Próximo passo foi incluir os trasnformadores dentro da lib *ColumnTransformer* que chamei de *preprocessador1*. Essa lib permite o fazer transformações em colunas usando 2 pipelines diferentes ao mesmo tempo.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/pipeline.png?raw=true)

O modelo a ser treinado será o CatBoost, que é um algoritmo de gradient boosting que foi projetado para lidar com dados categóricos de forma eficiente. Foi projetado para ser rápido e eficiente, com algoritmos otimizados e implementações paralelas e tem sido amplamente utilizado em competições de ciência de dados, onde se destacou por seu desempenho e capacidade de lidar com dados categóricos. O algoritmo utiliza a abordagem de boosting, onde um conjunto de modelos de aprendizado de máquina relativamente fracos é combinado para formar um modelo forte e preciso. Ele constrói os modelos em uma sequência, onde cada modelo sucessivo é treinado para corrigir os erros cometidos pelos modelos anteriores. O treinamento é realizado por meio do gradiente descendente, minimizando uma função de perda que mede a diferença entre as previsões e os rótulos reais do conjunto de treinamento.

O Pipeline do Sklearn tem um mecanismo muito útil. Permite adicionar pipeline dentro de pipeline, e isso foi feito aqui. No Pipeline chamado *model* usei o *SelectKBest* que é uma lib do Sklearn que seleciona as 'k' variáveis mais importantes para o modelo usando algum critério, nesse caso usando a Informação Mútua. Além disso, foi incluído os passos anteriores : *preprocessador1* e o *classificador*

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/catboost.png?raw=true)

## Tuning do modelo

Para um melhor desempenho do modelo, algo recomendável é modificar os hiperparâmetros e ir testando vários tipos de configurações de hiperparâmetros para decidir qual a melhor combinação. Algo cansativo, mas podemos usar o *GridSearch* da Sklearn, uma lib em que indicamos uma lista de hiperparâmetros que escolhe a melhor combinação de acordo com uma métrica, nesse caso usei a *Área da Curva ROC* (roc_auc).

Os hiperparâmetros selecionados pra teste:

'classificador__learning_rate': [0.02, 0.03] - A taxa de aprendizado do algoritmo;

'classificador__iterations': [750, 1000]  - A quantidade de iterações que o algoritmo fará;

'classificador__depth': [5, 6] - A profundidade de cada ramo;

'preprocessador__num__num_null__strategy': ['mean', 'median'] - O imputer de registros nulos das variáveis numéricas (média ou mediana)

Também foi utilizado a validação cruzada com k-fold, criando 5 folds.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/grid.png?raw=true)

Após testar todas as combinações, podemos ver os melhores hiperparâmetros dentre os que foram testados:

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/best_hiper.png?raw=true)

## Variáveis mais importantes para o modelo

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/var_import.png?raw=true)

## Algumas Métricas do treinamento

- Acurácia da validação cruzada: 92.03278
- Área da Curva ROC da validação cruzada 78.319
- Gini da validação cruzada 56.638
- KS da validação cruzada 42.014

# Teste

após a carga dos dados de testes, foram excluídas as variáveis de alta cardinalidade e a variável alvo foi separada das explicativas. 
O Modelo treinado anteriormente salvo em um arquivo serializado *grid_search.pkl* foi carregado e as previsões foram feitas.

## Métricas 

              precision    recall  f1-score   support

           0       0.93      0.99      0.96     56538
           1       0.43      0.10      0.16      4965

    accuracy                           0.92     61503

- Acurácia da base de Teste: 91.60854
- Área da Curva ROC da base de Teste 75.698
- Gini da base de Teste 51.396
- KS da base de Teste 38.478

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/roc.png?raw=true)

**O Modelo foi validado e está pronto pra fase de ajuste de Score**

# Ajuste do Score

Após a Validação do modelo, foi feito o ajuste do Score. Primeiro calculamos o Score de cada cliente, depois agrupamos os clientes por faixa, de acordo com cada Score.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/faixas_score.png?raw=true)

Decidi separar os clientes em 3 grupos de acordo com as faixas de Score. As 2 faixas menores, decidi negar o empréstimo automaticamente, as 5 maiores decidi aprovar automaticamente e as intermediárias deixo pra área de negócio decidir. Então criei 3 dataframe usando só os clientes de cada grupo para fazer alguns testes.

**Faixas Maiores**

- Nas faixas de cima, em 30751 empréstimos com default previstas pelo modelo, em 894 ele afirma errôneamente que não é Default;
- Aprova automaticamente 52.81 % dos empréstimos 'bons';
- Aproximadamente 18.01 % dos empréstimos 'ruins' estão nas faixas maiores

**Faixas Menores**

- Nas faixas de baixo, possui 9706 Operações sem Default;
- Aproximadamente 17.17 % dos empréstimos 'bons' serão recusados;
- Evita prejuizos em cerca de 52.27 % dos empréstimos 'ruins'.

**Faixas Intermediárias**

- Aproximadamente 29.73 % dos empréstimos 'ruins' estão nas faixas intermediárias
- Aproximadamente 30.02 % dos empréstimos 'bons' estão nas faixas intermediárias

**Como o sistema aprova apenas 7.31% dos empréstimos ruins, aprova 1/3 dos empréstimos bons, evita mais da metade dos empréstimos ruins e só recusa 17.21% dos empréstimos bons, decidi aprovar esse sistema pra entrar em produção.**

*Regras do sistema*

- Scores acima de 0.898090 serão aprovados automaticamente;
- Scores iguais ou abaixo de 0.788517 serão negados automaticamente;
- Scores acima de 0.788517 e abaixo de 0.898090 serão enviados para um analista.

# Sistema com Streamlit

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/tela_inicial.png?raw=true)

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/aprovado.png?raw=true)

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/negado.png?raw=true)


# Sugestões

Como o objetivo técnico desse projeto foi criar um processe rápido e limpo usando o Pipeline e o GridSearch do Sklearn, é recomendado fazer uma análise exploratória dos dados para um melhor dataprep. Também recomendo testar outros algoritmos como Regressão Logistíca, Randomforest, LightGBM, XGBoost e hiperparâmetros diferentes. Fiz todo processamento no meu computador pessoal, então não testei tantos hiperparâmetros, seria uma boa ideia utilizar algum serviço de nuvem para testar mais algoritmos e mais hiperparâmetros. Usar mais variáveis também pode ajudar a melhorar o desempenho do modelo, recomendo usar as bases históricas e de bureau disponíveis na página da competição.

Outra sugestão é fazer mais ajustes nas faixas de Scores, se o objetivo é diminuir o risco de inadimpência, o recomendado é diminuir a quantidade de faixas de score no grupo de cima. Se o objetivo é não perder potenciais clientes bons, é recomendado aumentar o número de faixas do grupo de baixo. Tudo depende do objetivo do negócio.











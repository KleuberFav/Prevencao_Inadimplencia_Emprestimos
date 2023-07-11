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
Foram disponibilizados 2 arquivos principais que serão usados nesse projeto:
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

O primeiro passo foi separar a base para teste. Usei 20% dos dados da base para testes, usando o *StratifiedShuffleSplit* da Sklearn para manter a distribuição das classes da variável alvo. É importante separar dados para teste antes de qualquer tratamento para evitar que o modelo "trapaceie", então os dados de testes precisam ser dados que o modelo nunca viu. Esse procedimento aumenta a confiança do modelo.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/strat_split.png?raw=true)

Após o split, foi feito separado a variável alvo das variáveis explicativas

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/var_alvo.png?raw=true)

É importante fazer o tratamento dos dados, por exemplo:
- tratar os valores nulos pois alguns modelos não funcionam ou não lidam bem com valores ausentes;
- Padronizar as variáveis numéricas, ou seja, deixar as variáveis em uma escala parecida, pois alguns modelos usam metódos como cálculo de distância, então escalas diferentes aumentam o processamento e o modelo pode entender que variáveis com escalas maiores possuem mais peso que variaveis com escalas menores;
- Criar Variáveis Dummy's para variáveis categóricas com baixa cardinalidade pois muitos algoritmos de Machine Learning só aceitam valores numéricos e a criação de dummy's preserva a informação original e elimina o risco de criar uma hierarquia dos domínios das variáveis, coisa que pode ocorrer quando usamos Label Encoder.

O próximo passo foi selecionar as variáveis numéricas, colocar em uma lista e salvar essa lista no arquivo *lista_vars_numericas.pkl*

O mesmo foi feito com as variáveis categóricas com cardinalidade <= 10, que foram salvas no arquivo *lista_vars_dummif.pkl*

Também foi feito com as variáveis categóricas com cardinalidade > 10 e foram salvas no arquivo *lista_vars_le.pkl*

Após alguns testes decidi não usar as variáveis de alta cardinalidade pois não acrescentavam muito ao modelo e uma limitação do *ColumnTransformer* do Sklearn que por algum motivo que eu desconheço só permitiu usar 2 transformadores, então optei pelos transformadores das variáveis numéricas e os transformadores das variáveis categóricas de baixa cardinalidade, como veremos a seguir.

## Pipeline de Treino do Modelo

Com os dados já preparados, chegou a hora de treinar o modelo, para isso usei o Pipeline da *Sklearn*. Primeiramente fiz um pipeline para as variáveis numéricas, na qual chamei de *num_transformer*. Nele pedi pra tratar os nulos com a média  de cada coluna usando *SimpleImputer* e após isso normalizar os dados usando *StandardScaler*.

Para as variáveis categóricas fiz outro pipeline chamada *ohc_transformer*. Pedi para o *ohc_transformer* tratar os nulos com a moda de cada coluna usando *SimpleImputer* e após isso criar variáveis Dummys usando *OneHotEncoder* e dropando o primeiro dummy de cada classe para diminuir o processamento e a multicolinearidade (alguns autores discordam da multicolinearidade nesse caso, mas todos concordam que aumenta a performance do processamento).
Próximo passo foi incluir os transformadores dentro da lib *ColumnTransformer* que chamei de *preprocessador1*. Essa lib permite o fazer transformações em colunas usando 2 pipelines diferentes ao mesmo tempo.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/pipeline.png?raw=true)

O modelo a ser treinado será o CatBoost, que é um algoritmo de gradient boosting que foi projetado para lidar com dados categóricos de forma eficiente. Foi projetado para ser rápido e eficiente, com algoritmos otimizados e implementações paralelas e tem sido amplamente utilizado em competições de ciência de dados, onde se destacou por seu desempenho e capacidade de lidar com dados categóricos. O algoritmo utiliza a abordagem de boosting, onde um conjunto de modelos de aprendizado de máquina relativamente fracos é combinado para formar um modelo forte e preciso. Ele constrói os modelos em uma sequência, onde cada modelo sucessivo é treinado para corrigir os erros cometidos pelos modelos anteriores. O treinamento é realizado por meio do gradiente descendente, minimizando uma função de perda que mede a diferença entre as previsões e os rótulos reais do conjunto de treinamento.

O Pipeline do Sklearn tem um mecanismo muito útil que permite adicionar pipeline dentro de outro pipeline, e isso foi feito aqui. No Pipeline chamado *model* usei o *SelectKBest* que é uma lib do Sklearn que seleciona as 'k' variáveis mais importantes para o modelo pois na maioria dos casos, menos é mais. Essa técnica de seleção de variáveis necessita de algum critério, nesse caso usando a Informação Mútua e você informa quantas variáveis serão usadas. Além disso, foi incluído os passos anteriores do pipeline: *preprocessador1* e o *classificador*

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/catboost.png?raw=true)

## Tuning do modelo

Para um melhor desempenho do modelo, algo recomendável é modificar os hiperparâmetros e ir testando vários tipos de configurações de hiperparâmetros para decidir qual a melhor combinação. Algo cansativo, mas podemos usar o *GridSearch* da Sklearn, uma lib em que informamos uma lista de hiperparâmetros que escolhe a melhor combinação de acordo com uma métrica, nesse caso usei a *Área da Curva ROC* (roc_auc).

Os hiperparâmetros selecionados pra teste:

'classificador__learning_rate' - A taxa de aprendizado do algoritmo;

'classificador__iterations': - A quantidade de iterações que o algoritmo fará;

'classificador__depth': - A profundidade de cada ramo;

'preprocessador__num__num_null__strategy': - O imputer de registros nulos das variáveis numéricas (média ou mediana)

Também foi utilizado a validação cruzada com k-fold, criando 5 folds. A validação cruzada com k-folds é uma técnica utilizada para avaliar o desempenho de um modelo de aprendizado de máquina de forma mais robusta e confiável. Ela envolve dividir o conjunto de dados em k partes (ou "folds") de tamanho aproximadamente igual. O modelo é treinado e avaliado k vezes, cada vez usando k-1 folds como conjunto de treinamento e o fold restante como conjunto de teste.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/grid.png?raw=true)

Após testar todas as combinações, podemos ver os melhores hiperparâmetros dentre os que foram testados:

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/best_hiper.png?raw=true)

## Variáveis mais importantes para o modelo

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/var_import.png?raw=true)

## Algumas Métricas do treinamento

- Acurácia da validação cruzada: 92.05392
- Área da Curva ROC da validação cruzada 78.476
- Gini da validação cruzada 56.952
- KS da validação cruzada 42.341

**Acurácia**
Mede a proporção de instâncias classificadas corretamente em relação ao total de instâncias. É expressa como um valor entre 0 e 1, onde 1 indica uma classificação perfeita, em que todas as instâncias são classificadas corretamente, e 0 indica uma classificação completamente incorreta. Embora a acurácia seja uma métrica simples e fácil de entender, ela pode ser enganosa em certos casos, principalmente quando há um desequilíbrio significativo entre as classes, que é o caso desse projeto, onde temos poucos casos de inadimplência.

**Área da Curva ROC**
A área sob a curva ROC é uma medida da capacidade discriminativa do modelo. Ela representa a probabilidade de que um exemplo positivo escolhido aleatoriamente seja classificado corretamente em relação a um exemplo negativo escolhido aleatoriamente. Quanto maior a área sob a curva ROC, melhor é o desempenho do modelo em distinguir entre as classes. A área sob a curva ROC varia entre 0.0 e 1.0. Uma área de 0.5 indica um modelo que possui um desempenho semelhante ao de um classificador aleatório, enquanto uma área de 1.0 indica um classificador perfeito, que é capaz de separar perfeitamente as classes positiva e negativa.

**Gini**

Ele mede a impureza das classes em um determinado ponto de divisão, também conhecido como critério de divisão de Gini. Mede a probabilidade de um exemplo escolhido aleatoriamente ser incorretamente classificado ao ser atribuído aleatoriamente a uma das classes. Quanto menor o valor do índice de Gini, melhor é a pureza da divisão. O índice de Gini varia de 0 a 1, onde 0 indica uma divisão perfeitamente pura, onde todos os exemplos pertencem a uma única classe, e 1 indica uma divisão impura, onde a distribuição das classes é uniforme.

**KS**
O KS (Kolmogorov-Smirnov) é uma métrica comumente usada em machine learning para avaliar a qualidade de um modelo preditivo, principalmente em problemas de classificação binária. Essa métrica é usada para medir a capacidade do modelo de distinguir entre duas classes, geralmente a classe positiva e a classe negativa. O valor do KS varia entre 0 e 1. Quanto maior o valor do KS, melhor o modelo é em distinguir as duas classes. Um valor de KS próximo a 1 indica uma separação clara entre as classes, enquanto um valor próximo a 0 indica que o modelo não é capaz de diferenciar adequadamente as classes.

# Teste

após a carga dos dados de testes, foram excluídas as variáveis de alta cardinalidade e a variável alvo foi separada das explicativas. 
O Modelo treinado anteriormente salvo em um arquivo serializado *grid_search.pkl* foi carregado e as previsões foram feitas.

## Métricas 

              precision    recall  f1-score   support

           0       0.93      0.99      0.96     56538
           1       0.43      0.10      0.16      4965

    accuracy                           0.92     61503

- Acurácia da base de Teste: 91.66382
- Área da Curva ROC da base de Teste 75.797
- Gini da base de Teste 51.594
- KS da base de Teste 38.897

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/roc.png?raw=true)

Sicsu (2010) afirma que um KS entre 30% e 40% está próximo ao excelente quando se trata da eficácia de modelos de *Credit Score*.

Sicsu (2010) também afirma que um modelo cuja AUC ROC é igual ou superior a 70 é considerado satisfatório quando se trata de modelos de *Credit Score*.

**O Modelo foi validado e está pronto pra fase de ajuste de Score**

# Ajuste do Score

Após a Validação do modelo, foi feito o ajuste do Score. Primeiro calculamos o Score de cada cliente, depois agrupamos os clientes por faixa, de acordo com cada Score.

![](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/faixas_score.png?raw=true)

Decidi separar os clientes em 3 grupos de acordo com as faixas de Score. As 2 faixas menores, decidi negar o empréstimo automaticamente, as 5 maiores decidi aprovar automaticamente e as intermediárias (zona cinza) deixo pra área de negócio decidir. Então criei 3 dataframe usando só os clientes de cada grupo para fazer alguns testes.

**Faixas 5 a 9**

- Nessas faixas, em 30751 empréstimos com default previstas pelo modelo, em 888 ele afirma errôneamente que não é Defaul;
- Aprova automaticamente 52.82 % dos empréstimos 'bons';
- Aproximadamente 17.89 % dos empréstimos 'ruins' estão nas faixas maiores

**Faixas 0 e 1**

- Nessas faixas, possui 9718 Operações sem Default;
- Aproximadamente 17.19 % dos empréstimos 'bons' serão recusados;
- Evita prejuizos em cerca de 52.02 % dos empréstimos 'ruins'.

**Faixas 2 a 4**

- Aproximadamente 30.09 % dos empréstimos 'ruins' estão nas faixas intermediárias
- Aproximadamente 29.99 % dos empréstimos 'bons' estão nas faixas intermediárias

**Como o sistema aprova menos de 1/5 dos empréstimos ruins, aprova mais da metade dos empréstimos bons, evita mais da metade dos empréstimos ruins e só recusa 17.19% dos empréstimos bons, decidi aprovar esse sistema pra entrar em produção.**

*Regras do sistema*

- Scores acima de 0.788625 serão aprovados automaticamente;
- Scores iguais ou abaixo de 0.788625 serão negados automaticamente;
- Scores acima de 0.788625 e abaixo de 0.788625 serão enviados para um analista.

# Sistema com Streamlit

![Tela Inicial](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/tela_inicial.png?raw=true)

![Aprovado](https://github.com/KleuberFav/Prevencao_Inadimplencia_Emprestimos/blob/main/outputs/aprovado.png?raw=true)


# Resultados

- Cerca de 52% dos empréstimos bons foram aprovados automaticamente. Isso ajuda a fidelizar clientes que cumprem o prazo;
- Cerca de 52% dos empréstimos que entrariam em Default foram rejeitados automaticamente. Ou seja, evita um pouco mais da metade dos prejuízos causados pela inadimplência;
- Aprova menos de 20% dos empréstimos ruins, algo que a área de negócio deve definir se é um número aceitável;
- Rejeita cerca de 17% dos empréstimos que seriam pagos no tempo certo, algo que a área de negócio deve definir se é aceitável;
- Em torno de 30% dos empréstimos bons serão analisados por um especialista, então são negócios bons que ainda podem ser aprovados.


# Sugestões

Como o objetivo técnico desse projeto foi criar um processe rápido e limpo usando o Pipeline e o GridSearch do Sklearn, é recomendado fazer uma análise exploratória dos dados para um melhor dataprep. Também recomendo testar outros algoritmos como Regressão Logistíca, Randomforest, LightGBM, XGBoost e hiperparâmetros diferentes. Fiz todo processamento no meu computador pessoal, então não testei tantos hiperparâmetros, seria uma boa ideia utilizar algum serviço de nuvem para testar mais algoritmos e mais hiperparâmetros. Testar outras variáveis também pode ajudar a melhorar o desempenho do modelo, recomendo usar as bases históricas e de bureau disponíveis na página da competição.

Outra sugestão é fazer mais ajustes nas faixas de Scores, se o objetivo é diminuir o risco de inadimpência, o recomendado é diminuir a quantidade de faixas de score no grupo de cima. Se o objetivo é não perder potenciais clientes bons, é recomendado aumentar o número de faixas do grupo de baixo. Tudo depende do objetivo do negócio.











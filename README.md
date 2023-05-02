
# Evolução Diferencial para otimização de funções multimodais

Este código implementa um algoritmo de Evolução Diferencial (DE) para otimização de funções multimodais simples. A biblioteca `inspyred` é utilizada para implementar a DE. Para avaliação das soluções, são utilizadas funções de sua categoria do benchmark CEC2017.

A função de aptidão é definida pela função `evaluator`, que recebe como parâmetros um conjunto de candidatos a solução e um dicionário de argumentos contendo o número da função que deve ser avaliada. A avaliação é feita para cada candidato a solução e é adicionada em uma lista de aptidões. 

As funções dos problemas analizados estão disponivei na pasta cec2017. Este codigo foi obtido no repositório:  https://github.com/tilleyd/cec2017-py.

A função `generator` é utilizada para gerar a população inicial, recebendo como parâmetro um gerador aleatório e um dicionário de argumentos contendo a dimensão da solução.

O critério de parada é definido pela função `terminador`, que recebe como parâmetros a população atual, o número de gerações e o número de avaliações de função realizadas. O critério de parada adotado é que o número máximo de avaliações de função seja atingido ou que a aptidão da melhor solução seja menor que 10^-8.

A DE é implementada pela função `differential_evolution`, que recebe como parâmetros a dimensão da solução e o número da função que deve ser otimizada. A função utiliza a biblioteca `random` para inicializar a semente do gerador aleatório. A população inicial é gerada pela função `generator`. A avaliação das soluções é realizada pela função `evaluator`. O operador de crossover utilizado é o `blend_crossover` com o parâmetro `blx_alpha` igual a 0.1. Os operadores de mutação utilizados são o `nonuniform_mutation` e o `offspring`, que são responsáveis por realizar mutações em uma solução. A função `evolve` é responsável por realizar a evolução da população.

Por fim, o código executa a DE para cada uma das sete funções multimodais do benchmark CEC2017 e imprime os resultados na tela. Cada resultado consiste na aptidão da melhor solução encontrada e na solução em si.

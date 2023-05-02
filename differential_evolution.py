"""[DCC067]Function 3.ipynb
"""

import numpy as np
from math import cos, sin, pi, sqrt, pow
import inspyred 
from inspyred import ec
import random 
from time import time
import pandas as pd
from scipy.linalg import orth
import multimodal_functions
import cec2017.functions as functions
import multiprocessing
import csv
import os

"""Função de Aptidão"""

def evaluator(candidates, kwargs):
  n = kwargs.get('function', float('inf'))
  fitness = [] 
  if n == 3: 
    for i in range(0, len(candidates)): #Shifted and Rotated Rosenbrock’s Function - Problema 3
      fitness.append(abs(n*100 - functions.f3([candidates[i]])))
  elif n == 4: 
    for i in range(0, len(candidates)): #Shifted and Rotated Rastrigin’s Function - Problema 4
      fitness.append(abs(n*100 - functions.f4([candidates[i]])))
  elif n == 5: 
    for i in range(0, len(candidates)): # Shifted and Rotated Expanded Scaffer’s F6 Function - Problema 5
      fitness.append(abs(n*100 - functions.f5([candidates[i]])))
  elif n == 6: 
    for i in range(0, len(candidates)): # Lunacek bi-Rastrigin Function - Problema 6
      fitness.append(abs(n*100 - functions.f6([candidates[i]])))
  elif n == 7: 
    for i in range(0, len(candidates)): #Non-continuous Rotated Rastrigin's Function - Problema 7
      fitness.append(abs(n*100 - functions.f7([candidates[i]])))
  elif n == 8: 
    for i in range(0, len(candidates)): #Levy Function - Problema 8
      fitness.append(abs(n*100 - functions.f8([candidates[i]])))
  elif n == 9: 
    for i in range(0, len(candidates)): #Modified Schwefel's Function - Problema 9
      fitness.append(abs(n*100 - functions.f9([candidates[i]])))
      
  return fitness

"""Geradores População Inicial"""


def generator(random, kwargs):
  d = kwargs.get('dimention', float('inf'))
  x = []
  for i in range(0,d):
    x.append(random.uniform(-100, 100))
  return x

"""Criterio de paradada"""

def terminador(population, num_generations, num_evaluations, kwargs):
    best = max(population)
    max_evolutions = kwargs.get('max_evolutions', float('inf'))
    if num_evaluations == max_evolutions or best.fitness < 10**(-8):
        return True
    else: 
        return False

"""Evolução Diferencial"""

def differential_evolution(n_dimentions, caso):
  prng = random.Random()
  prng.seed(time()) 

  pop_size = 20*n_dimentions
  max_evolutions = n_dimentions*10000

  ea = ec.DEA(prng)
  ea.selector = inspyred.ec.selectors.rank_selection
  ea.terminator = ea.terminator = lambda population, num_generations, num_evaluations, args: terminador(population, num_generations, num_evaluations, {'max_evolutions': max_evolutions})
  ea.evaluator = ea.evaluator = lambda candidates, args: evaluator(candidates, {'function': caso})
  ea.generator = ea.generator = lambda random, args: generator(prng, {'dimention': n_dimentions})
  offspring = ec.variators.blend_crossover = lambda random, mom, dad, args: ec.variators.blend_crossover(random, mom, dad, args= {'crossover_rate' : 1, 'blx_alpha': 0.1})
  final_pop = ea.evolve(generator=ea.generator, evaluator=ea.evaluator, pop_size=pop_size, maximize=False, bounder=inspyred.ec.Bounder(-100, 100), 
                        max_evaluations=max_evolutions, mutation_rate=0.5, num_crossover_points=2,num_elites=round((n_dimentions+2)/3 + 1),variator=[offspring,  
                        ec.variators.nonuniform_mutation])

  best = max(final_pop)
  if best.fitness < 10**(-8):
    return [0], best.candidate
  else:
    return best.fitness, best.candidate


"""Resultados"""

for k in range(0,7):
  print("Processando função {}...".format(k+3))
  data2 = {'indice': [], 'melhor_resultado': [], 'melhor_candidato': []}
  data10 = {'indice': [], 'melhor_resultado': [], 'melhor_candidato': []}
  for i in range(0,1):

    fitness, candidate = differential_evolution(2, k+3)
    data2['indice'].append(i)
    data2['melhor_resultado'].append(fitness[0])
    data2['melhor_candidato'].append(candidate)


    fitness, candidate = differential_evolution(10, k+3)
    data10['indice'].append(i)
    data10['melhor_resultado'].append(fitness[0])
    data10['melhor_candidato'].append(candidate)

    df_2 = pd.DataFrame(data2)
    df_10 = pd.DataFrame(data10)

  df_2.to_csv('data2_' + str(k+3) + '.csv', index=False)
  df_10.to_csv('data10_' + str(k+3) + '.csv', index=False)



"""###Dataframe Resultados"""

def processa_dataframes(*dfs):
    resultados = []
    for df in dfs:
        # Obter melhor e pior candidato
        print(df['melhor_resultado'])
        melhor = df.loc[df['melhor_resultado'].idxmin()]
        pior = df.loc[df['melhor_resultado'].idxmax()]
        # Adicionar na lista de resultados
        resultados.append({
            'melhor_candidato': melhor['melhor_candidato'],
            'melhor_resultado': melhor['melhor_resultado'],
            'pior_candidato': pior['melhor_candidato'],
            'pior_resultado': pior['melhor_resultado'],
            'mediana': df['melhor_resultado'].median(),
            'media': df['melhor_resultado'].mean(),
            'desvio_padrao': df['melhor_resultado'].std()
        })
    # Criar e retornar o dataframe com os resultados
    resultado_df = pd.DataFrame(resultados, columns=[
        'melhor_candidato', 'melhor_resultado', 'mediana', 'media', 'desvio_padrao', 'pior_candidato', 'pior_resultado'])
    resultado_df.sort_values('melhor_resultado', ascending=False, inplace=True)
    resultado_df.reset_index(drop=True, inplace=True)
    return resultado_df


df_3 = pd.read_csv('data2_3.csv')
df_4 = pd.read_csv('data2_4.csv')
df_5 = pd.read_csv('data2_5.csv')
df_6 = pd.read_csv('data2_6.csv')
df_7 = pd.read_csv('data2_7.csv')
df_8 = pd.read_csv('data2_8.csv')
df_9 = pd.read_csv('data2_9.csv')
resultado1 = processa_dataframes(df_3, df_4, df_5, df_6, df_7, df_8, df_9)
resultado1.to_excel('resultados_2.xls', index=False)


df1_3 = pd.read_csv('data10_3.csv')
df1_4 = pd.read_csv('data10_4.csv')
df1_5 = pd.read_csv('data10_5.csv')
df1_6 = pd.read_csv('data10_6.csv')
df1_7 = pd.read_csv('data10_7.csv')
df1_8 = pd.read_csv('data10_8.csv')
df1_9 = pd.read_csv('data10_9.csv')
resultado2 = processa_dataframes(df1_3, df1_4, df1_5, df1_6, df1_7, df1_8, df1_9)
resultado2.to_excel('resultados_10.xls', index=False)

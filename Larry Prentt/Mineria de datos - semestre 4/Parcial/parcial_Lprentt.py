# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:51:37 2021

@author: lprentt
"""

#import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir("D:/4to_Semestre/Mineria de datos/python\Parcial")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

df=pd.read_csv('LifeExpect.csv')


# 1


for i in range(4,22):
    df.plot.scatter(x=df.columns[i], y='Life expectancy ')
    plt.title(str(df.columns[i])+ 'vs Life expectancy', fontsize = 22)
    plt.xlabel(str(df.columns[i]), fontsize = 20)
    plt.ylabel('Life expectancy', fontsize = 20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)

# Todas las variables explican la expectativa de vida, excepto tamaño
# de poblacion

sns.heatmap(df.corr(),annot=True,cmap="RdYlGn") ######## CLAVE
  
# 2

# grafico de total invertido vs expectiva de vida
df.plot.scatter(x='Total expenditure', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title('Total expenditure vs Life expectancy', fontsize = 22)
plt.xlabel('Total expenditure', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# el grafico anterior se filtra por edades menores a 65 años
df2 = df.loc[(df['Life expectancy '] < 65)]

sns.scatterplot(data=df2, x='Total expenditure', y='Life expectancy ', hue='Country')

# el nuevo grafico muestra que la mayoria de paises africanos
# invierten muy poco en salud y por eso su expectativa de vida es baja
# por lo anterior los paises que quieran mejorar su expectativa de vida deben
# mejorar su inversion/gasto sanitario

# 3

a=df['Country'].unique()

df.plot.scatter(x='Adult Mortality', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title('Adult Mortality vs Life expectancy', fontsize = 22)
plt.xlabel('Adult Mortality', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x='infant deaths', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title('infant deaths vs Life expectancy', fontsize = 22)
plt.xlabel('infant deaths', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# 4

# alcohol versus expectativa de vida
# BMI vs expectativa de vida BMI es indice de masa corporal
# delgadez de personas entre 1-19 años vs expectativa de vida
# delgadez de personas entre 5- 9 años vs expectativa de vida

df.plot.scatter(x='Alcohol', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title('Alcohol vs Life expectancy', fontsize = 22)
plt.xlabel('Alcohol', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x=' BMI ', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title('BMI vs Life expectancy', fontsize = 22)
plt.xlabel('BMI', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x=' thinness  1-19 years', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title(' thinness  1-19 years vs Life expectancy', fontsize = 22)
plt.xlabel(' thinness  1-19 years', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x=' thinness 5-9 years', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title(' thinness 5-9 years vs Life expectancy', fontsize = 22)
plt.xlabel(' thinness 5-9 years', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)


# 5

# escolaridad versus expectativa de vida

df.plot.scatter(x='Schooling', y='Life expectancy ')
# pd.crosstab(df['Adult Mortality'],df['Life expectancy ']).plot(kind='bar')
plt.title('Schooling vs Life expectancy', fontsize = 22)
plt.xlabel('Schooling', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)


# 6


dfa1 = df.loc[((df['Life expectancy '] > 65) & (df['Alcohol'] < 5))]
dfa2 = df.loc[((df['Life expectancy '] <= 65) & (df['Alcohol'] > 5))]

sns.scatterplot(data=dfa1, x='Alcohol', y='Life expectancy ', hue='Country')

sns.scatterplot(data=dfa2, x='Alcohol', y='Life expectancy ', hue='Country')

# 7  Population

sorted_by_population = df.sort_values(['Population'], ascending=False)
print(sorted_by_population.head(20))

sns.scatterplot(data=sorted_by_population[0:100], x='Population', y='Life expectancy ', hue='Country')

# 8

# Hepatitis B
# Measles 
# Polio
# Diphtheria 

df.plot.scatter(x='Hepatitis B', y='Life expectancy ')
plt.title('Hepatitis B vs Life expectancy', fontsize = 22)
plt.xlabel('Hepatitis B', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# sorted_by_HepB = df.sort_values(['Hepatitis B'], ascending=False)
# sns.scatterplot(data=sorted_by_HepB[0:500], x='Hepatitis B', y='Life expectancy ', hue='Country')

df.plot.scatter(x='Measles ', y='Life expectancy ')
plt.title('Measles  vs Life expectancy', fontsize = 22)
plt.xlabel('Measles ', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x='Polio', y='Life expectancy ')
plt.title('Polio  vs Life expectancy', fontsize = 22)
plt.xlabel('Polio', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x='Diphtheria ', y='Life expectancy ')
plt.title('Diphtheria  vs Life expectancy', fontsize = 22)
plt.xlabel('Diphtheria ', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df.plot.scatter(x=' HIV/AIDS', y='Life expectancy ')
plt.title(' HIV/AIDS  vs Life expectancy', fontsize = 22)
plt.xlabel(' HIV/AIDS', fontsize = 20)
plt.ylabel('Life expectancy', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)


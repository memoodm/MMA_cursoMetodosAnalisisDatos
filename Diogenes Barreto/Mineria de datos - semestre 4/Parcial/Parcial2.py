# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:34:44 2021

@author: LENOVO
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:33:00 2021

@author: LENOVO
"""
import pandas as pd # Libreria para analisis de Datos#
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split# divide en DF en subconjuntos de entrenamientos aleatorio
from sklearn.utils import resample

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import seaborn as sns # graficas y estadisticas , maquillaje#
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)

import os # Operative System #
import matplotlib.pyplot as plt
plt.rc("font", size=14)

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)
from countryinfo import CountryInfo

os.chdir("C:/Users/LENOVO/Documents/Clase9/")

#leer los datos
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
csv1_file='LifeExpect.csv' #asignacion dle nombre del archivo a una bandeja #

df1=pd.read_csv(csv1_file)



#Guarda el documento
df1.to_excel('./DFregresion.xls',index=False)



#%%% 1 punto. De los diversos factores de predicción elegidos inicialmente afectan realmente a la esperanza de vida? ¿Cuáles son las variables de predicción que realmente afectan la esperanza de vida?

# # #Perfilaje de Variables y Miramos las quemas afecten a la variable de esperanza de vida
from pandas_profiling import ProfileReport
ProfileReport(df1)
profile = ProfileReport(df1, title="Profiling Report",explorative=True)

profile = ProfileReport(df1)
profile.to_file('profile_report.html')


# se filtran los valores que no se encuentren superando el rango de 0.4
# #Correlacion de variables numericas
df1.corr(method='pearson')
df1.corr()
# #Mapa de calor para las variables numericas y encontrar el grado de importancia de cada variable

sns.heatmap(df1.corr(),annot=True,cmap="RdYlGn")
sns.pairplot(df1)

## Respuesta: Todas las variables afectan la "expectativa de vida" ya que ninguna de sus correlaciones es de 0 
# sin embargo, alguas afectan mas que otra, impactan mas las quetienen un factor de correlacion mayor a .6
#Negativamente: thinness 5-9 years, thinness 1-19 years, HIV/AIDS, Adult Mortality
#Positivamente: Schooling, Income composition of resources, GDP, Diphtheria, Polio, BMI, Alchool
#%%%%  punt 2. ¿Debería un país con un valor de esperanza de vida inferior (<65) aumentar su gasto sanitario para mejorar su esperanza de vida media?

Menor65= df1.loc[(df1['Life expectancy '] < 65)]['Total expenditure'].mean()
Mayor65 = df1.loc[(df1['Life expectancy '] >= 65)]['Total expenditure'].mean()

print("Gasto sanitario promedio de los paises que tienen un promedio de vida inferior a 65: %.6f"%(Menor65))
print("Gasto sanitario promedio de los paises que tienen un promedio de vida superior a 65: %.6f"%(Mayor65))

#RESPUESTA: Si un pasis deberia aumentar su gasto para mejorar su esperanza de vida enn la poblacion.
# Gasto sanitario promedio de los paises que tienen un promedio de vida inferior a 65: 5.349745
# Gasto sanitario promedio de los paises que tienen un promedio de vida superior a 65: 6.166988

#%%  3 Punto.  ¿Cómo afectan las tasas de mortalidad de niños y adultos a la esperanza de vida?

#llenamos los nan conn valores medios.
a= df1["Life expectancy "].mean()
b= df1["Adult Mortality"].mean()
c= df1["infant deaths"].mean()

df1 = df1.fillna( df1["Life expectancy "].mean())
df1 = df1.fillna(df1["Adult Mortality"].mean())
df1 = df1.fillna(df1["infant deaths"].mean())

corr = df1.corr()
print(corr)
#Matrzi de Correacion de las variables
#                                      Year  ...  Schooling
# Year                             1.000000  ...   0.209400
# Life expectancy                  0.170033  ...   0.751975
# Adult Mortality                 -0.079052  ...  -0.454612
# infant deaths                   -0.037415  ...  -0.193720
# Alcohol                         -0.052990  ...   0.547378
# percentage expenditure           0.031400  ...   0.389687
# Hepatitis B                      0.104333  ...   0.231117
# Measles                         -0.082493  ...  -0.137225
#  BMI                             0.108974  ...   0.546961
# under-five deaths               -0.042937  ...  -0.209373
# Polio                            0.094158  ...   0.417866
# Total expenditure                0.090740  ...   0.246384
# Diphtheria                       0.134337  ...   0.425332
#  HIV/AIDS                       -0.139741  ...  -0.220429
# GDP                              0.101620  ...   0.448273
# Population                       0.016969  ...  -0.031668
#  thinness  1-19 years           -0.047876  ...  -0.471652
#  thinness 5-9 years             -0.050929  ...  -0.460632
# Income composition of resources  0.243468  ...   0.800092
# Schooling                        0.209400  ...   1.000000
# Life_expectancy = 1

print("Correlacion entre 'life expectancy' y 'Adult Mortality': %.4f"%(corr.iloc[Life_expectancy]["Adult Mortality"]))
print("Correlacion entre 'life expectancy' y 'infant deaths': %.4f"%(corr.iloc[Life_expectancy]["infant deaths"]))

#RESPUESTA: La correlacion entre esperanza de vida y mortlidad de adultos es -0.69, y la correlacion  con muerte de niños es -0.19.
#La afectacion en ambas variables es negativa, hay una afectacion directa enn la esperanza de vida.

#%%  4 Punto. ¿Tiene la esperanza de vida una correlación positiva o negativa con los hábitos alimenticios, el estilo de vida, el ejercicio, el tabaquismo, el consumo de alcohol, etc.?
corr = df1.corr()
print(corr)
Life_expectancy = 1
for column in corr.columns:
    result = "Positiva" if corr.iloc[Life_expectancy][column] > 0 else "Negativa"
    print("%s -> %s"%(result,column))

#RESPUESTA: 
    
# Positiva -> Year
# Positiva -> Life expectancy 
# Negativa -> Adult Mortality
# Negativa -> infant deaths
# Positiva -> Alcohol
# Positiva -> percentage expenditure
# Positiva -> Hepatitis B
# Negativa -> Measles 
# Positiva ->  BMI 
# Negativa -> under-five deaths 
# Positiva -> Polio
# Positiva -> Total expenditure
# Positiva -> Diphtheria 
# Negativa ->  HIV/AIDS
# Positiva -> GDP
# Negativa -> Population
# Negativa ->  thinness  1-19 years
# Negativa ->  thinness 5-9 years
# Positiva -> Income composition of resources
# Positiva -> Schooling


#%%  5 Punto. ¿Cuál es el impacto de la escolarización en la vida útil de los seres humanos?
corr = df1.corr()
print(corr)
Life_expectancy= 1
print("Correlacion entre 'life expectancy' y 'Schooling': %.4f"%(corr.iloc[Life_expectancy]["Schooling"]))

#RESPUESTA: Una mejor escolarizacion mejora la expectativa de vida. la  correlacionn es del 0.7520

#%%  6 Punto. ¿Tiene la esperanza de vida una relación positiva o negativa con el consumo de alcohol?
corr = df1.corr()
Life_expectancy = 1
print("Correlacion entre 'life expectancy' y 'Alcohol': %.4f"%(corr.iloc[Life_expectancy]["Alcohol"]))

#Respuesta: La correlacion entre la esperanza de vida y alcoholismo es 0.409, segun este valor es positiva

#%%  7 Punto. ¿Los países densamente poblados tienden a tener una menor esperanza de vida?

corr = df1.corr()
print(corr)
Life_expectancy= 1
print("Correlacion entre 'life expectancy' y 'Population': %.4f"%(corr.iloc[Life_expectancy]["Population"]))

#Respuesta: La correlacion entre La poblacion y la esperanza de vida es de -0.0215. Implica que los pasises densamente poblados tienen maor afectacion en la esperzanza de vida

#%%  8 Punto. ¿Cuál es el impacto de la cobertura de vacunación en la esperanza de vida?
corr = df1.corr()
print(corr)
Life_expectancy= 1
print("Correlacion entre 'life expectancy' y 'percentage expenditure': %.4f"%(corr.iloc[Life_expectancy]["percentage expenditure"]))
#Respuesta: La correlacion es 0.3818, es positiva , implica que a mayor cobertura de vacunacion, mayor espezanza de vida.
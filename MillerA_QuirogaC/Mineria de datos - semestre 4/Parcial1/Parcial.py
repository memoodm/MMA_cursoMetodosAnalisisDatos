#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:23:48 2021


            ACTIVIDAD CLASE 7

Miller Alexander Quiroga Campos
@author: milleralexanderquirogacampos
"""

"""

¿Los diversos factores de predicción elegidos inicialmente afectan realmente a la esperanza de vida? 
¿Cuáles son las variables de predicción que realmente afectan la esperanza de vida?   

¿Debería un país con un valor de esperanza de vida inferior (<65) aumentar su gasto sanitario para mejorar su esperanza de vida media?

¿Cómo afectan las tasas de mortalidad de niños y adultos a la esperanza de vida?

¿Tiene la esperanza de vida una correlación positiva o negativa con los hábitos alimenticios, el estilo de vida, el ejercicio, el tabaquismo, el consumo de alcohol, etc.?

¿Cuál es el impacto de la escolarización en la vida útil de los seres humanos?

¿Tiene la esperanza de vida una relación positiva o negativa con el consumo de alcohol?

¿Los países densamente poblados tienden a tener una menor esperanza de vida?

¿Cuál es el impacto de la cobertura de vacunación en la esperanza de vida?

"""

#%%

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import os

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn import preprocessing # procesamiento de datos
from pandas_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#%% importamos archivos de excel y csv y los convertimos en df1

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Clase7/")
cwd = os.getcwd()

#df1 ColombiaCB
df = pd.read_csv('LifeExpect.csv')#, index_col = 0)
df.info()

ProfileReport(df)

df.hist()
df.corr()

pd.crosstab(df['Country'],df.Year).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
#plt.savefig('18_Last Funding Amount Currency')

for i in range (4,22):
    df.plot.scatter(x = df.columns[i], y = 'Life expectancy')
    plt.title('Grafica', fontsize = 22)
    plt.xlabel('', fontsize = 20)
    plt.ylabel('', fontsize = 20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    
correlations = df.corr()["Life expectancy"].sort_values()
print("Most positive correlations : \n\n",correlations.tail(8)*100 )
print("Most negative correlations : \n\n",correlations.head(8)*100 )

"""
¿Cuáles son las variables de predicción que realmente afectan la esperanza de vida?
Rta: 
    Las variables que afectan la esperanza de vida es la Mortalidad en Adultos,
    Alcohol, BIM, Polio, Diphtheria, GDP, thinness  1-19 years, thinness 5-9 years,
    Income composition of resources
"""

    
#%%

pd.crosstab(df['Country'], df['Life expectancy ']).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
    
df.plot.scatter(x = 'Country', y = 'Life expectancy ')
#pd.crosstab(df['Country'], df['Life expectancy ']).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)



df.plot.scatter(x = 'Country', y = 'Total expenditure')
#pd.crosstab(df['Country'], df['Life expectancy ']).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

"""
¿Debería un país con un valor de esperanza de vida inferior (<65) aumentar su gasto sanitario para mejorar su esperanza de vida media?
Rta:
    En los paises que aparecen en la gráfica, se indica los países si
    Deberían aumentar el gasto sanitario
"""

#%%

df.plot.scatter(x = 'Adult Mortality', y = 'Life expectancy ')
#pd.crosstab(df['Country'], df['Life expectancy ']).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

df['infant deaths'].hist()
df.plot.scatter(x = 'Life expectancy ', y = 'infant deaths')
#pd.crosstab(df['Country'], df['Life expectancy ']).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

pd.crosstab(df['infant deaths'], df['Life expectancy ']).plot(kind='bar')
plt.title('Grafica', fontsize = 22)
plt.xlabel('Country', fontsize = 20)
plt.ylabel('Year', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

sns.scatterplot(data=df, x='Adult Mortality', y='infant deaths', hue='Life expectancy ')

plt.bar(df['Adult Mortality'],df['Life expectancy '])
plt.show()

plt.bar(df['infant deaths'],df['Life expectancy '])
plt.show()

correlacion = df.corr()

"""
¿Cómo afectan las tasas de mortalidad de niños y adultos a la esperanza de vida?

Rta:
    Las tasas de mortalidad de niños y adultos Afecta a las esperanza de vida.



"""

#%%

df.corrwith(df['Life expectancy '])

"""
¿Tiene la esperanza de vida una correlación positiva o negativa con los hábitos 
alimenticios, el estilo de vida, el ejercicio, el tabaquismo, el consumo de alcohol, etc.?

Si.., hay variables que como se muestra en el codigo al ejecutar muestran su 
correlación negativa y otras positivas como: 
Year                               0.170033
Life expectancy                    1.000000
Adult Mortality                   -0.696359
infant deaths                     -0.196557
Alcohol                            0.404877
percentage expenditure             0.381864
Hepatitis B                        0.256762
Measles                           -0.157586
 BMI                               0.567694
under-five deaths                 -0.222529
Polio                              0.465556
Total expenditure                  0.218086
Diphtheria                         0.479495
 HIV/AIDS                         -0.556556
GDP                                0.461455
Population                        -0.021538
 thinness  1-19 years             -0.477183
 thinness 5-9 years               -0.471584
Income composition of resources    0.724776
Schooling                          0.751975

"""
    
#%%
sns.scatterplot(data = df, x = 'Alcohol', y = 'Life expectancy ')

"""
¿Tiene la esperanza de vida una relación positiva o negativa con el consumo de alcohol?
RTA:
    De acuerdo a la grafica si hay una esperanza de vida
    
"""    
    
    
    
    
    

"""
Created on Fri Dec  4 21:52:09 2020

@author: milleralexanderquirogacampos
"""

#comentario de la librería pandas
"""
pandas se basa en matrices numéricas para proporcionar ricas estructuras
de datos y herramientas de análisis de datos. La función pandas.DataFrame 
proporciona matrices etiquetadas de datos (potencialmente heterogéneos), 
similares al "data.frame" R. La función pandas.read_csv se puede utilizar 
para convertir un archivo de valores separados por comas en un objeto 
DataFrame.


Patsy es una biblioteca Python para describir modelos estadísticos 
y construir matrices de diseño usando fórmulas similares a la R.
"""

import pandas as pd #importamos librerías para analisis de datos
# import numpy as np #librería para análisis numérico
# import matplotlib.pyplot as plt #Librería de visualización
# import seaborn as sns #graficas más bonitas y estadística
import os #Librería para el sistema operativo
# from sklearn import preprocessing # procesamiento de datos
# from pandas_profiling import ProfileReport

# plt.rc("font", size=14)
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)


#ruta del archivo donde esta alojado
os.chdir('/Users/milleralexanderquirogacampos/Documents/Documentos - MacBook Pro de Miller/MaestriaMatematicas/3 - Metodos de Analisis de datos/LuzStellaGomez/Clase5')
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo
excel_file = "HW.xlsx" # Asignación del nombre del archivo a una bandera

df = pd.read_excel(excel_file)


import statsmodels.api as sm
from patsy import dmatrices
import statsmodels.formula.api as smf


#Seleccionamos las variables de interés y miramos las 3 filas:
vars = ["QuantySold", "Price", "Advertising"]
df = df[vars] 
df[-3:]

df = df.dropna()
df[-7:]

#El modelo se estima usando la regresión de mínimos cuadrados ordinarios (OLS).
#Usamos la función de dmatrices de Patsy para crear matrices de diseño:
y, X = dmatrices('QuantySold ~ Price + Advertising', data=df, return_type='dataframe')


#Las matrices/marcos de datos resultantes se ven así:
y[:3]
X[:3]


#Ajustar un modelo en los modelos estadísticos típicamente implica 3 pasos fáciles:
#Usar la clase de modelo para describir el modelo
#Ajustar el modelo usando un método de clase
#Inspeccione los resultados utilizando un método de resumen
#Para la OLS, esto se logra mediante:

mod = sm.OLS(y, X)    # Describe model

res = mod.fit()       # Forma el modelo

print(res.summary())   # Resumir el modelo estadísticamente


#El objeto res tiene muchos atributos útiles. Por ejemplo, 
#podemos extraer estimaciones de parámetros y r-cuadrado tecleando:
res.params
#Escriba dir(res) para obtener una lista completa de atributos.
res.rsquared


#statsmodels le permite realizar una serie de útiles diagnósticos 
#de regresión y pruebas de especificación. Por ejemplo, aplicar 
#la prueba de Rainbow para la linealidad (la hipótesis nula es que 
#la relación se modela adecuadamente como lineal)
sm.stats.linear_rainbow(res)

#statsmodels también proporciona funciones gráficas. Por ejemplo, 
#podemos dibujar una gráfica de regresión parcial para un prueba
#conjunto de regresores por:
sm.graphics.plot_partregress('QuantySold', 'Price', 'Advertising', data = df, obs_labels=False)



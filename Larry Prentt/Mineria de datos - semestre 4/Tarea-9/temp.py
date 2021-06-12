# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:10:12 2021

@author: lprentt
"""
# https://www.ncdc.noaa.gov/cdo-web/

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler

os.chdir("D:/4to_Semestre/Mineria de datos/python/clase-5")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

#%%

df=pd.read_csv('temps.csv')

df.head()         # muestra los 5 primeros registros del DataFrame
df.tail()         # muestra los 5 ultimos  registros del DataFrame
df.year.unique()  # determina cuantos años hay en database
df.month.unique() # determina cuantos meses hay en database
df.day.unique()   # determina cuantos dias hay en database
df.week.unique()   # determina nombres de los dia de la semana en database
df['temp_2'].isnull().sum()  # determina cuantos datos nulos hay en temp_2
df['temp_1'].isnull().sum()  # determina cuantos datos nulos hay en temp_1
df['average'].isnull().sum()  # determina cuantos datos nulos hay en average
df['actual'].isnull().sum()  # determina cuantos datos nulos hay en actual
df['friend'].isnull().sum()  # determina cuantos datos nulos hay en friend

df.describe()
df.describe().to_excel("describe.xlsx")

# sns.pairplot(df)  # matriz de correlacion con graficos entre variables

df.loc[(df["temp_2"] == 117) ]
df.loc[(df["temp_1"] == 117) ]

df.loc[(df.temp_2 == 117),'temp_2']= 92  # reemplaza datos anomalos
df.loc[(df.temp_1 == 117),'temp_1']= 92  # reemplaza datos anomalos

df.describe().to_excel("describe2.xlsx")

# Grafico de las 4 variables de Temperatura
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df.index, df.actual)
axs[0, 0].set_title("actual= Max Temp")
axs[1, 0].plot(df.index, df.temp_1)
axs[1, 0].set_title("Max temp in prior day")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df.index, df.temp_2)
axs[0, 1].set_title("Max temp in two prior day")
axs[1, 1].plot(df.index, df.friend)
axs[1, 1].set_title("Friend Temp")
axs[0, 0].set_xlabel('dias')
axs[0, 1].set_xlabel('dias')
axs[1, 0].set_xlabel('dias')
axs[1, 1].set_xlabel('dias')
fig.tight_layout()


#%%

# preparando data

df2 = df.copy()
df2= pd.get_dummies(df2)

# borrando las columnas:
    
# forecast_noaa
# forecast_acc	
# forecast_under
df2 = df2.drop(['forecast_noaa', 'forecast_acc', 'forecast_under'], axis=1) 

# Mapa de Calor
sns.heatmap(df2.corr(),annot=True,cmap="RdYlGn") ######## CLAVE

# display de las 5 primeras filas y las ultimas doce columnas
# para verificar lel resultado del get-dummies
df2.iloc[:,5:].head(5).to_excel("dummies.xlsx")
df2.iloc[:,5:].head(5)

y = np.array(df2['actual'])  # definiendo la variable objetivo o dependiente como array
df2 = df2.drop('actual', axis = 1)  # removiendo variable objetivo del df2
x_list = list(df2.columns)  # creando lista con nombre de variables independientes
df2 = np.array(df2) # conviertiendo df a array de variables independientes

# preparando data para regresion
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2, y, test_size = 0.35, random_state = 50)

#%%

# La linea base de predicciones son los promedios historicos x dia 'average'
linea_base = test_x[:, x_list.index('average')]

# error = average - y (test)
error_linea_base = abs(linea_base - test_y)

print('Error_linea_base promedio: ', round(np.mean(error_linea_base), 3), 'grados.')
# Error_linea_base promedio:  4.905

######################################################
# ajuste del modelo con datos de entrenamiento

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 50)
regressor.fit(train_x, train_y)
print(regressor.score(train_x, train_y))

####################################################
# predicción con los datos de testing
forecast = regressor.predict(test_x)

# Metricas de desempeño
####################################
# Calculo del valor absoluto del error
errors = abs(forecast - test_y)
error2 = (forecast - test_y)**2
np.mean(error2) # error cuadratico medio
np.sqrt(np.mean(error2)) # std error cuadratico medio

# Error Promedio
print('Mean Absolute Error:', round(np.mean(errors), 3), 'grados.')
# Mean Absolute Error: 3.575 grados.

####################################

# Calculo de vector de errores relativos del valor absoluto del error
error_mean = 100 * (errors / test_y)
# Calculo de la precision del modelo
accuracy = 100 - np.mean(error_mean)
print('Accuracy:', round(accuracy, 3), '%.')

#%%
###########################################
######  Analisis de sensibilidad

# conda install python-graphviz
import pydotplus
# Importando librerias para visualizacion
from sklearn.tree import export_graphviz

arbol = regressor.estimators_[15]
# definiendo ruta para graphviz
os.environ['PATH'] = os.environ['PATH']+';' + r'C:\Users\lpren\anaconda3\Library\bin\graphviz'

# Exportando archivo *.dot
export_graphviz(arbol, out_file="tree2" + ".dot", feature_names =  x_list)
# Generando archivo *.dot
dot_data = export_graphviz(arbol, out_file=None, feature_names = x_list)
graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
# Generando archivo *.png para visualizar todo el arbol
graph.write_png("tree2" + "_gv.png")

############################################

# Limitando profundidad del arbol a 3 niveles
regressor_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
regressor_small.fit(train_x, train_y)
# Extrayendo el arbol pequeño
arbol_small = regressor_small.estimators_[5]

# Exportando archivo *.dot
export_graphviz(arbol, out_file="small-tree" + ".dot", feature_names =  x_list)
# Generando *.dot
dot_data = export_graphviz(arbol_small, out_file=None, feature_names = x_list)
graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
# Generando *.png
graph.write_png("small-tree" + "_gv.png")


#%%

# obtencion de variables independientes mas importantes /influyentes en el modelo
importances = list(regressor.feature_importances_)
# Listado de una tupla que contiene Var. Ind y su grado de importancia
feature_importances = [(feature, round(importance, 2)) \
                       for feature, importance in zip(x_list, importances)]
# ordenamiento de variables de acuerdo a importancia
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# imprimiendo Var. Ind. y su importancia
[print('Variable: {:20} Importance: {}'.format(*pair)) \
 for pair in feature_importances];
    
#%%

# Nuevo random forest con las 2 variables mas importantes
regressor_most_important = RandomForestRegressor(n_estimators= 1000, random_state=50)
# Sacando de lista de var X, a temp_1 y average
important_indices = [x_list.index('temp_1'), x_list.index('average')]
train_important = train_x[:, important_indices]
test_important = test_x[:, important_indices]
# Entrenando RF con temp_1 y Average
regressor_most_important.fit(train_important, train_y)
# Realizando predicciones y calculando el error
predictions = regressor_most_important.predict(test_important)
errors_mi = abs(predictions - test_y)
# Mostrando metricas de predicción
print('Mean Absolute Error:', round(np.mean(errors_mi), 2), 'degrees.')
mape = np.mean(100 * (errors_mi / test_y))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

#%%
# grafica de variables de importancia
plt.figure()
# Definiendo el estilo
plt.style.use('fivethirtyeight')
# Listado de variables X
x_values = list(range(len(importances)))
# Haciendo un  bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, x_list, rotation='vertical')
# Definiendo label del eje x y titulo del grafico
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

#%%

# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = df2[:, x_list.index('month')]
days = df2[:, x_list.index('day')]
years = df2[:, x_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': y})
# Dates of predictions
months = test_x[:, x_list.index('month')]
days = test_x[:, x_list.index('day')]
years = test_x[:, x_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.figure()
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');

#%%

# Make the data accessible for plotting
true_data['temp_1'] = df2[:, x_list.index('temp_1')]
true_data['average'] = df2[:, x_list.index('average')]
true_data['friend'] = df2[:, x_list.index('friend')]
# Plot all the data as lines
plt.figure()
plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)
# Formatting plot
plt.legend(); plt.xticks(rotation = '60');
# Lables and title
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');


#%%
Y=y.reshape(len(y),1)

sc_X = StandardScaler()     # define a sc_X como una variable tipo Clase StandardScaler
sc_Y = StandardScaler() 
    # define a sc_Y como una variable tipo Clase StandardScaler

X = sc_X.fit_transform(df2)   # Normaliza la escala de X mediante estandarizacion X = (X - Xmean)/Xstd
Y = sc_Y.fit_transform(Y)   # Normaliza la escala de Y mediante estandarizacion Y = (Y - Ymean)/Ystd

Y2 = Y.ravel()
#%%

regressorA = RandomForestRegressor(n_estimators = 1000, random_state = 50)
regressorA.fit(X, Y2)
#%%
from sklearn.svm import SVR

c=1
epsi = 0.01
regressor1 = SVR(kernel='rbf', C=c, epsilon=epsi) # Kernel function = Radial basis function
grado = 3                                        # Define el grado del polinomio
grado2 = 5                                       # Define el grado del polinomio
regressor2 = SVR(kernel ='poly', degree=grado) # Kernel function = Polinomio grado de acuerdo a variable grado
regressor3 = SVR(kernel ='poly', degree=grado2) # Kernel function = Polinomio grado de acuerdo a variable grado2
regressor4 = SVR(kernel ='linear')
# Entrenamiento de los modelos SVR con todo el dataset de level y salario
# regressor.fit(X,Y) # Warning para que se use Ravel
regressor1.fit(X,Y.ravel())
regressor2.fit(X,Y.ravel())
regressor3.fit(X,Y.ravel())
regressor4.fit(X,Y.ravel())
#%%
from sklearn.tree import DecisionTreeRegressor
regressor5 = DecisionTreeRegressor(random_state = 0)
regressor5.fit(X, Y)

#%%
# regresion multilineal
from sklearn.linear_model import LinearRegression
regressor6 = LinearRegression().fit(X, Y)

#%%

import scipy.stats

suma=0
for i in range(len(y)):
    suma+=y[i]
suma /= len(Y)
Y_prom = suma
print(Y_prom)

# Suma total de cuadrados
def STCC(y,y_prom):
    rango = len(y)
    suma = 0
    for i in range(rango):
        suma+=(y[i]-y_prom)**2
    return suma

def SCE(y, y_cal):
    rango = len(y_cal)
    suma = 0
    for i in range(rango):
        suma+=(y[i]-y_cal[i])**2
    return suma
# Creando la tabla resumen para ANOVA TEST

n = 10  # Filas
m = 8  # Columnas
lista = [0] * n
for i in range(n):
    lista[i] = [0] * m

# Titulos 1ra Fila
lista[0][0]="Param. / Modelo"
lista[0][1]="Random Forest"
lista[0][2]="rbf"
lista[0][3]="Pol grado 3"
lista[0][4]="Pol grado 5"
lista[0][5]="svr-linear"
lista[0][6]="DTR"
lista[0][7]="Multilineal"

# Titulos 1ra Columna
lista[1][0]="STCC"
lista[2][0]="SCE"
lista[3][0]="SCR"
lista[4][0]="n"
lista[5][0]="k"
lista[6][0]="S^2"
lista[7][0]="F"
lista[8][0]="p-Value"
lista[9][0]="r^2"


y_cal1=sc_Y.inverse_transform(regressorA.predict(X))
y_cal2=sc_Y.inverse_transform(regressor1.predict(X))
y_cal3=sc_Y.inverse_transform(regressor2.predict(X))
y_cal4=sc_Y.inverse_transform(regressor3.predict(X))
y_cal5=sc_Y.inverse_transform(regressor4.predict(X))
y_cal6=sc_Y.inverse_transform(regressor5.predict(X))
y_cal7=sc_Y.inverse_transform(regressor6.predict(X))

def llenar_lista(i,Y_real,Y_PROM,Y_CAL,N,n,k):

    global X
    global Y
    global lista

    for j in range(1,N):
        
        if j == 1:
            lista[j][i]=STCC(Y_real,Y_PROM)
        if j == 2:
            lista[j][i]=SCE(Y_real, Y_CAL)
        if j == 3:
            lista[j][i]=lista[j-2][i]-lista[j-1][i]
        if j == 4:
            lista[j][i]=n            
        if j == 5:
            lista[j][i]=k
        if j == 6:
            lista[j][i]=lista[j-4][i]/(lista[j-2][i]-lista[j-1][i]-1)
        if j == 7:
            lista[j][i]=lista[j-4][i]/lista[j-2][i]/lista[j-1][i]
        if j == 8:
            p_value = 1-scipy.stats.f.cdf(lista[j-1][i], k, (n-k-1))
            lista[j][i]=p_value
        if i == 1:
            lista[9][i]= regressorA.score(X,Y2)
        elif i == 2:
            lista[9][i]= regressor1.score(X,Y)
        elif i == 3:
            lista[9][i]= regressor2.score(X,Y)
        elif i == 4:
            lista[9][i]= regressor3.score(X,Y)
        elif i == 5:
            lista[9][i]= regressor4.score(X,Y)
        elif i == 6:
            lista[9][i]= regressor5.score(X,Y)
        else:
            lista[9][i]= regressor6.score(X,Y)
    print("lista llena")

llenar_lista(1,y,Y_prom,y_cal1,10,len(y),1)
llenar_lista(2,y,Y_prom,y_cal2,10,len(y),1)
llenar_lista(3,y,Y_prom,y_cal3,10,len(y),1)
llenar_lista(4,y,Y_prom,y_cal4,10,len(y),1)
llenar_lista(5,y,Y_prom,y_cal5,10,len(y),1)
llenar_lista(6,y,Y_prom,y_cal6,10,len(y),1)
llenar_lista(7,y,Y_prom,y_cal7,10,len(y),1)

# Genera tabla resumen con analisis estadistico de metodos de regresion

from tabulate import tabulate
print(tabulate(lista, headers='firstrow', tablefmt='fancy_grid'))


# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:00:00 2021

@author: lprentt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler

os.chdir("D:/4to_Semestre/Mineria de datos/python/4_archivo")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

#%%
df=pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:-1].values  # Extrae del Dataframe df la variable level
Y = df.iloc[:,-1].values    # Extrae del Dataframe df la variable Salario

Y=Y.reshape(len(Y),1)

sc_X = StandardScaler()     # define a sc_X como una variable tipo Clase StandardScaler
sc_Y = StandardScaler()     # define a sc_Y como una variable tipo Clase StandardScaler

X = sc_X.fit_transform(X)   # Normaliza la escala de X mediante estandarizacion X = (X - Xmean)/Xstd
Y = sc_Y.fit_transform(Y)   # Normaliza la escala de Y mediante estandarizacion Y = (Y - Ymean)/Ystd

#%%

from sklearn.svm import SVR

c=1
epsi = 0.01
regressor = SVR(kernel='rbf', C=c, epsilon=epsi) # Kernel function = Radial basis function
grado = 3                                        # Define el grado del polinomio
grado2 = 5                                       # Define el grado del polinomio
regressor1 = SVR(kernel ='poly', degree=grado) # Kernel function = Polinomio grado de acuerdo a variable grado
regressor2 = SVR(kernel ='poly', degree=grado2) # Kernel function = Polinomio grado de acuerdo a variable grado2

# Entrenamiento de los modelos SVR con todo el dataset de level y salario
# regressor.fit(X,Y) # Warning para que se use Ravel
regressor.fit(X,Y.ravel())
regressor1.fit(X,Y.ravel())
regressor2.fit(X,Y.ravel())

# Reescala un nivel de 6.5
sc_X.transform([[6.5]])

# predice el salario rescalado del nivel reescalado
regressor.predict(sc_X.transform([[6.5]]))

# Prediccion de un nuevo resultado usando transformacion inversa
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualizando los resultados

# Visualizando los resultados. RBF vs Polinomios
label1A = "SVR con C ="+str(c)
label2A = 'Polinomio grado '+str(grado)
label3A = 'Polinomio grado '+str(grado2)
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color = 'blue', label =label1A)
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor1.predict(X)), color = 'green', label =label2A )
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor2.predict(X)), color = 'orange', label =label3A )

plt.title('(SVR)', fontsize = 22)
plt.xlabel('Nivel o Cargo Laboral', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 22)
plt.ylabel('Salario', fontsize = 20)
#plt.show()

#%%
# Uso de Regresion Tree
from sklearn.tree import DecisionTreeRegressor
regressor3 = DecisionTreeRegressor(random_state = 0)
regressor3.fit(X, Y)

#agregando plot de regression tree
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor3.predict(X)), color = 'black', label ="regression-tree" )
plt.legend(fontsize = 22)

# Prediccion de un nuevo resultado usando transformacion inversa
sc_Y.inverse_transform(regressor3.predict(sc_X.transform([[6.5]])))

#%%

#####################################################################################

### Generando estadisticas ANOVA TEST

import scipy.stats

y=sc_Y.inverse_transform(Y)

suma=0
for i in range(len(y)):
    suma+=y[i][0]
suma /= len(Y)
Y_prom = suma
print(Y_prom)

# Suma total de cuadrados
def STCC(y,y_prom):
    rango = len(y)
    suma = 0
    for i in range(rango):
        suma+=(y[i][0]-y_prom)**2
    return suma

def SCE(y, y_cal):
    rango = len(y_cal)
    suma = 0
    for i in range(rango):
        suma+=(y[i][0]-y_cal[i])**2
    return suma


# Creando la tabla resumen para ANOVA TEST

n = 10  # Filas
m = 5  # Columnas
lista = [0] * n
for i in range(n):
    lista[i] = [0] * m

# Titulos 1ra Fila
lista[0][0]="Param. / Modelo"
lista[0][1]="RBF"
lista[0][2]="Pol grado 3"
lista[0][3]="Pol grado 5"
lista[0][4]="Regression Tree"

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


y_cal1=sc_Y.inverse_transform(regressor.predict(X))
y_cal2=sc_Y.inverse_transform(regressor1.predict(X))
y_cal3=sc_Y.inverse_transform(regressor2.predict(X))
y_cal4=sc_Y.inverse_transform(regressor3.predict(X))

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
            lista[9][i]= regressor.score(X,Y)
        elif i == 2:
            lista[9][i]= regressor1.score(X,Y)
        elif i == 3:
            lista[9][i]= regressor2.score(X,Y)
        else:
            lista[9][i]= regressor3.score(X,Y)
    print("lista llena")

llenar_lista(1,y,Y_prom,y_cal1,10,len(y),1)
llenar_lista(2,y,Y_prom,y_cal2,10,len(y),1)
llenar_lista(3,y,Y_prom,y_cal3,10,len(y),1)
llenar_lista(4,y,Y_prom,y_cal4,10,len(y),1)

# Genera tabla resumen con analisis estadistico de metodos de regresion

from tabulate import tabulate
print(tabulate(lista, headers='firstrow', tablefmt='fancy_grid'))

# print(regressor.score(X,Y))

# from sklearn.metrics import r2_score
# r2_score(X,Y)

# lista =[[0]*5]*9

# import scipy.stats
# alpha = 0.05 #Or whatever you want your alpha to be.
# 1- p_value = scipy.stats.f.cdf(0.647472036, 9, 6)
# if p_value > alpha:
#     # Reject the null hypothesis that Var(X) == Var(Y)
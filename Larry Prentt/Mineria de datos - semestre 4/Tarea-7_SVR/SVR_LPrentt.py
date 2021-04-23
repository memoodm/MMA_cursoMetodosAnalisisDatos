# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:00:00 2021

@author: lprentt
"""


#import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler


os.chdir("D:/4to_Semestre/Mineria de datos/python/4_archivo")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

df=pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:-1].values  # Extrae del Dataframe df la variable level
Y = df.iloc[:,-1].values    # Extrae del Dataframe df la variable Salario

Y=Y.reshape(len(Y),1)

sc_X = StandardScaler()     # define a sc_X como una variable tipo Clase StandardScaler
sc_Y = StandardScaler()     # define a sc_Y como una variable tipo Clase StandardScaler

X = sc_X.fit_transform(X)   # Normaliza la escala de X mediante estandarizacion X = (X - Xmean)/Xstd
Y = sc_Y.fit_transform(Y)   # Normaliza la escala de Y mediante estandarizacion Y = (Y - Ymean)/Ystd

from sklearn.svm import SVR

regressor = SVR(kernel='rbf') # Kernel function = Radial basis function
grado = 3
grado2 = 5                  # Define el grado del polinomio
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
label1 = 'Polinomio grado '+str(grado)
label2 = 'Polinomio grado '+str(grado2)
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color = 'blue', label ='rbf')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor1.predict(X)), color = 'green', label =label1 )
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor2.predict(X)), color = 'orange', label =label2 )
plt.title('Comparacion de modelos (SVR)', fontsize = 22)
plt.xlabel('Nivel o Cargo Laboral', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Salario', fontsize = 20)
plt.legend(fontsize = 22)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:03:06 2020

@author: lprentt
"""

import pandas as pd
import matplotlib.pyplot as plt

ventas = {'Cant_Vend': [8500,4700,5800,7400,6200,7300,5600],
                'Precio': [2,5,3,2,5,3,4],
                'Publicidad': [2800,200,400,500,3200,1800,900],
                }

df = pd.DataFrame(ventas,columns=['Cant_Vend','Precio','Publicidad'])

print(df)

####### Grafica Sencilla

plt.scatter(df['Precio'], df['Cant_Vend'], color='red')
plt.title('Cantidades Vendidas Vs Precio', fontsize=14)
plt.xlabel('Precio del Producto', fontsize=14)
plt.ylabel('Cantidad Vendida', fontsize=14)
plt.grid(True)
plt.show()

####### Dos Grafica en un solo plot o figura

fig = plt.figure()

ax1 = fig.add_subplot(121) # 1 row, 2 columns, 1st position
ax2 = fig.add_subplot(122) # 1 row, 2 columns, 2nd position

ax1.scatter(df['Precio'],df['Cant_Vend'], color='red')
ax2.scatter(df['Publicidad'],df['Cant_Vend'])  # color azul por default

ax1.set_title('Cantidades Vendidas Vs Precio')
ax1.set_xlabel('Precio del Producto')
ax1.set_ylabel('Cantidad Vendida')

ax2.set_title('Cantidades Vendidas Vs Publicidad')
ax2.set_xlabel('publicidad')
ax2.set_ylabel('Cantidad Vendida')
#plt.grid(True)
plt.show()

#########################

from sklearn import linear_model

# X tiene 2 variables 
# Y tiene 1 variable 

X = df[['Precio', 'Publicidad']] 
Y = df['Cant_Vend']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
New_Precio = 2.75
New_Publicidad = 3000
print ('Prediccion de Candidad Vendida: \n', regr.predict([[New_Precio ,New_Publicidad]]))

# with statsmodels.api

#################################################

import statsmodels.api as sm
# Genera estadisticas del modelo ajustado
# with statsmodels
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()

print_model = model.summary()
print(print_model)

######################################################
# se obtienen los valores calculados
predictions = model.predict(X)

#######################################################

#create residual vs. predictor plot for 'precio'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'Precio', fig=fig)

#create residual vs. predictor plot for 'Publicidad'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'Publicidad', fig=fig)

##################################################
# Normal Prob plot

import scipy.stats
import numpy as np

counts, start, dx, _ = scipy.stats.cumfreq(df['Cant_Vend'], numbins=df.shape[0])
x = np.arange(counts.size) * dx + start

plt.plot(x, counts, 'ro')
plt.xlabel('Value')
plt.ylabel('Cumulative Frequency')

plt.show()

# https://stackoverflow.com/questions/46127030/how-to-use-python-to-draw-a-normal-probability-plot-by-using-certain-column-data

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html

# https://pythonhealthcare.org/tag/pp-plot/


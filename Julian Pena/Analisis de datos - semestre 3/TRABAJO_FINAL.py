# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:10:08 2020

@author: Julian
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
import seaborn as sns #graficas y estadistica



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os #sistema op


os.chdir('C:/Users/Julian/Documents/MMA/PYTHONLUZ')
cwd=os.getcwd() # asigan variable swd al directorio
xls_file = "Winequality.xls"


# TOMANDO LOS DATOS DE LOS EQUIPOS

df = pd.read_csv(xls_file, header= 0, sep=';')

# Cluster, agrupar en datos parecidos se usan cuando los datos no estan agrupados.

# Se puede hacer segmentacion El vecino mas cewrcano K-NN
# K= numero de vecinos si es 1 solo hay un vecino cercano y si es dos hay 2

print(df.head(20))

# hago clusters para calidad de vino , Puedo usar k means

#***** TAREA, VER UN NUEVO VINO A QUE CALIDAD PERTENECE (cluster)***** 8 CLUSTERS de calidades
# hacer el analisis con todo lo que hemos visto 
# importo in key neigborhs clasifier y hago un predict

print(df.describe())

#sns.countplot(x= "quality", data = df)
#plt.title("Quality")
#plt.show()


print(df["quality"].value_counts(normalize = True))

	
df.hist()
#plt.show()

# density, ph y fixed acidity distribuc normal 


# Modelo


X = df.loc[:,df.columns != 'quiality']
y = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler() # necesito escalamiento del dato max al min 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#k-Nearest Neighbor

n_neighbors = 6
 
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Accuracy training: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy test: {:.2f}'
     .format(knn.score(X_test, y_test)))

# Precisión del modelo


pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred)) 
print(classification_report(y_test, pred)) # f1 score 57 % bajito 
# 3 y 8 no tienen precision 


 
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

# Prcision en graficpo 

k_range = range(1, 8)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])



clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)


# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

#print(clf.predict([]))

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:30:00 2020

@author: lprentt
"""
#import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import scipy.stats
import numpy as np

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns

os.chdir("D:/Analisis-Datos/python")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

text_file = 'Winequality.csv'  # objeto string con el nombre del archivo a revisar # Es una bandera

# df = pd.read_csv(text_file, sep=';')
# df=pd.read_csv(text_file, index_col =0)
# df = pd.read_csv('text_file', header=0)

df=pd.read_csv(text_file)

df.head(20) # Muestra los primeros 20 datos
df.tail(20) # Muestra los ultimos 20 datos
df.shape  # Muestra dimensiones del dataframe

print("Rows, columns: " + str(df.shape)) # imprime el # de filas y columnas

print(df.isna().sum()) # Imprime si hay datos faltantes para cada variable

print(df.isnull().sum()) # Imprime si hay datos nulos para cada variable

df.describe() # estadistica descriptiva de los datos

sns.countplot(x='quality',data=df)  # Countplot para la variable quality
plt.title('Clusters of Wine Quality') # Countplot = Histograma de 6 bines
plt.show()                            # con valores de bines de de 3 a 8

df.hist(bins=6) # Hace histogramas para todas las variables, con 6 bins para cada una

df['quality'].value_counts(normalize=True)*100 # % de datos para cada calidad de vino

# sorted_by_quality = df.sort_values(['alcohol', 'quality'], ascending=(True, True))
# sorted_by_quality['alcohol'].head(10).plot(,kind="barh")

# Doble sort con alcohol y quality
sorted_by_quality=df.sort_values('alcohol', ascending=False)
sorted_by_quality[['alcohol','quality']].head(10).plot.bar(stacked=True)
plt.ylabel("Alcohol and Quality counting")
plt.xlabel("# of Index")
plt.title("Wine quality - Alcohol - Top 10")
plt.show()

# Profile report
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile.to_file('profile_report_wine.html')

# Matriz de Correlacion
sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")

# Deteccion de posibles outliers usando:
# Z = (x-mhu)/(desv_standard)
# valores Z>3 y Z<-3 pueden ser datos anomalos

############################################
############################################
from scipy import stats
df2=df.copy()
z = np.abs(stats.zscore(df2))
df2 = df2[(z < 3).all(axis=1)]
df2.shape

plt.subplots(figsize=(15, 10))
sns.heatmap(df2.corr(),annot=True,cmap="RdYlGn")
sns.heatmap(df2.corr(),annot = True,cmap = "coolwarm")

# no se aplica metodo Z
##############################
##############################

# se aplica Drop para sacar variables fixed acidity
# residual sugar, free sulfur dioxide, pH
df3=df.copy()
df3.drop(df3.columns[[0, 3, 5, 8]], axis=1, inplace=True) 

# Balanceando la data, sin hacer drop de variables

X = df3.loc[:,df3.columns != 'quality']
y = df3.loc[:,df3.columns == 'quality']

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


smt = SMOTE(random_state=0)  # Muestra de entrenamiento balanceado con variable binaria
data_X,data_y=smt.fit_sample(X_train, y_train) # en misma proporcion

sns.countplot(x='quality',data=data_y)
plt.title('Wine Quality')
plt.show()

print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)

#################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# Se pone a variar k (Numero de vecinos alrededor)
k_range = range(1, 30)
k_scores = []
# Calcula puntajes de validacion-cruzada para cada valor de k = 1 to 26
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # validacion cruzada usando como scoring = accuracy y fold = 10
    scores = cross_val_score(knn, X ,y, cv=10, scoring='accuracy') 
    k_scores.append(scores.mean())

# Plot accuracy para cada k = 1 to 30
plt.plot(k_range, k_scores)
plt.xlabel('Valor de K para metodo KNN')
plt.ylabel('Cross-validated accuracy')

#####################################################
# Entrenando el modelo y prediciendo con k=15
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# lineas de la 154 a la 168 es para implementar roc curve, no funciona
# en roc_curve multiclass format no es permitido
y_scores = knn.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()
######################################################################

# Reporte de clasificacion para datos de prueba
print(metrics.classification_report(y_test, y_pred, digits=3, zero_division = 1))

# matriz de confusion
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Calculando cv score con puntaje = 'accuracy' y 10 folds
accuracy = cross_val_score(knn, X, y, scoring = 'accuracy',cv=10)
print('cross validation score',accuracy.mean())

# Calculando cv score con 'roc_auc_ovr' scoring y 10 folds
#accuracy = cross_val_score(knn, X, y, scoring = 'roc_auc_ovr',cv=10)
#print('cross validation score with roc_auc',accuracy.mean())

# Calculando roc_auc score con multiclass parameter
#print('roc_auc_score',roc_auc_score(y_test,knn.predict_proba(X_test), multi_class='ovr'))
###################################################################

# probar precision de valores de k

k_values = list(range(1,30))
accuracy_list =[]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = cross_val_score(knn, X, y, scoring = 'accuracy',cv=10)
    accuracy_list.append(accuracy.mean())
    
sns.barplot(k_values, accuracy_list)

#################################################################

datos= {'volatile acidity':[0.51],'citric acid':[0.13],
        'chlorides':[0.076],'total sulfur dioxide':[40],'density':[0.98],
'sulphates':[0.75],'alcohol':[4]}

X_1 = pd.DataFrame(datos,columns=['volatile acidity', 'citric acid',
       'chlorides', 'total sulfur dioxide', 'density',
       'sulphates', 'alcohol'])
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train) 
y1 = knn.predict(X_1)
print(y1)




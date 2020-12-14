# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:30:06 2020

@author: Carlos moreno
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir('C:\\Users\\Carlos Moreno\\Documents')
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

text_file = 'Winequality1.csv'  
df= pd.read_csv(text_file,sep=';')
pd.read_csv('Winequality1.csv',sep=';', header=0)


df.head()
df.hist(bins=8)

from pandas_profiling import ProfileReport
reporte = ProfileReport(df)
reporte.to_file('reporte_vinos.html')

plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(),annot = True,cmap = "coolwarm")

X = df.loc[:,df.columns != 'quality']
y = df.loc[:,df.columns == 'quality']

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

smt = SMOTE(random_state=0)
data_X,data_y=smt.fit_sample(X_train, y_train)

sns.countplot(x='quality',data=data_y)
plt.title('Calidad del Vino')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_range = range(1, 40)
k_values = []
# Calcula puntajes de validacion-cruzada para cada valor de k = 1 to 26
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # validacion cruzada usando como scoring = accuracy y fold = 10
    scores = cross_val_score(knn, X ,y, cv=10, scoring='accuracy') 
    k_values.append(scores.mean())

# Plot accuracy para cada k = 1 to 30
plt.plot(k_range, k_values)
plt.xlabel('Valor de K para metodo KNN')
plt.ylabel('Cross-validated accuracy')

from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn import metrics
# Reporte de clasificacion para datos de prueba
print(metrics.classification_report(y_test, y_pred, digits=2, zero_division = 1))

# matriz de confusion
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


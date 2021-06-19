
"""
Created on Sat Jun  5 12:52:33 2021

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
import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
# plt.style.use('ggplot')
import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score, mean_squared_error, r2_score,classification_report)
from countryinfo import CountryInfo
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.inspection import permutation_importance
from sklearn import svm, linear_model, model_selection
from sklearn.svm import SVR

os.chdir("C:/Users/LENOVO/Documents/Clase #11/")

#lectura de datos
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
csv_file='temps.csv' #asignacion dle nombre del archivo a una bandeja #


#lectura de datos
df =pd.read_csv(csv_file)

df.shape
df.info()
df.describe()#estadistica descriptiva de los datos
sns.pairplot(df)
df.plot(kind = 'box', subplots = True, figsize = (30,6), fontsize = 15);

# Remplazo los dos valores atípicos por el valor de la moda-Frecuencia

df.loc[(df.temp_2 == 117),'temp_2'] = int(df['temp_2'].mode())  # reemplaza datos mayor frecuencia
df.loc[(df.temp_1 == 117),'temp_1'] = int(df['temp_2'].mode())  # reemplaza datos mayor frecuencia
df['temp_1'].mode()
df.describe()#estadistica descriptiva de los datos

#preparamos los datos para el modelo

df2=pd.get_dummies(df, columns =['week'])
df2.corr()
df2.shape
df2.info()
# sns.pairplot(df2)

# #Analizamos el aporte de las variables
# from pandas_profiling import ProfileReport
# ProfileReport(df2)
# profile = ProfileReport(df2, title="Profiling Report",explorative=True)
# profile = ProfileReport(df2)
# profile.to_file('profile_report.html')

# display de las 5 primeras filas y las ultimas doce columnas para verificar lel resultado del get-dummies
df2.iloc[:,5:].head()
df2.head()

#df2.to_excel('./DFtem.xls',index=False)

# preparando data para regresion
y = df2.actual
X = df2[["year","month","day","year","temp_2","temp_1","average","actual","forecast_noaa","forecast_acc","forecast_under","friend", "week_Fri","week_Mon","week_Sat","week_Sun","week_Sun","week_Thurs", "week_Tues","week_Wed"]]

#%% RandonForest para 1000 arboles
model = RandomForestRegressor(
            n_estimators = 1000,
            criterion    = 'mse',
            max_depth    = None,
            max_features = 'auto',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 42
         )
model.fit(X,y)
predictions = model.predict(X)
print(model.score(X,y))

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation =train_test_split(X, y, test_size=validation_size, random_state=seed)



# Error de test del modelo inicial
# ==============================================================================
predictions = model.predict(X=X_validation)

rmse = mean_squared_error(
        y_true  = Y_validation,
        y_pred  = predictions,
        squared = False
       )
print(f"El error (rmse) de test es: {rmse}")


#Numero de arboles
# Validación empleando el Out-of-Bag error
# ==============================================================================
train_scores = []
oob_scores   = []

# Valores evaluados
estimator_range = range(1, 1000, 5)

# Bucle para entrenar un modelo con cada valor de n_estimators y extraer su error
# de entrenamiento y de Out-of-Bag.
for n_estimators in estimator_range:
   model = RandomForestRegressor(
                n_estimators = n_estimators,
                criterion    = 'mse',
                max_depth    = None,
                max_features = 'auto',
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123
             )
   model.fit(X_train, Y_train)
   train_scores.append(model.score(X_train, Y_train))
   oob_scores.append(model.oob_score_)
    
# Gráfico con la evolución de los errores
fig, ax = plt.subplots(figsize=(6, 3.84))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, oob_scores, label="out-of-bag scores")
ax.plot(estimator_range[np.argmax(oob_scores)], max(oob_scores),
        marker='o', color = "red", label="max score")
ax.set_ylabel("R^2")
ax.set_xlabel("n_estimators")
ax.set_title("Evolución del out-of-bag-error vs número árboles")
plt.legend();
print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(oob_scores)]}")


#%%% ANALISIS COMPARATIVO

#%%%SVR/POLINOMIAL/DECISION TREE

df2["Y"] = df2.actual

# y = df2.actual
# X = df2[["year","month","day","year","temp_2","temp_1","average","actual","forecast_noaa","forecast_acc","forecast_under","friend", "week_Fri","week_Mon","week_Sat","week_Sun","week_Sun","week_Thurs", "week_Tues","week_Wed"]]
X = df2.iloc[:,1:2].values  
Y = df2.iloc[:,-1].values    

Y=Y.reshape(len(Y),1)

sc_X = StandardScaler()     # define a sc_X como una variable tipo Clase StandardScaler
sc_Y = StandardScaler() 
    # define a sc_Y como una variable tipo Clase StandardScaler

X = sc_X.fit_transform(X)   # Normaliza  X = (X - Xmean)/Xstd
Y = sc_Y.fit_transform(Y)   # Normaliza  Y = (Y - Ymean)/Ystd  

regressor = SVR(kernel='rbf') 
regressor1 = SVR(kernel ='poly', degree=3) 
regressor2 = SVR(kernel ='poly', degree=5)
regressor3 = DecisionTreeRegressor(random_state = 0)

regressor.fit(X,Y.ravel())
regressor1.fit(X,Y.ravel())
regressor2.fit(X,Y.ravel())
regressor3.fit(X, Y.ravel())

#Realizamos la prediccion

#SRV
sc_X.transform([[6.5]])
regressor.predict(sc_X.transform([[6.5]]))
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Polinomial
sc_X.transform([[6.5]])
regressor1.predict(sc_X.transform([[6.5]]))
sc_Y.inverse_transform(regressor1.predict(sc_X.transform([[6.5]])))

sc_X.transform([[6.5]])
regressor2.predict(sc_X.transform([[6.5]]))
sc_Y.inverse_transform(regressor2.predict(sc_X.transform([[6.5]])))

label1 = 'Polinomio grado '+str(3)
label2 = 'Polinomio grado '+str(5)

#Decision tree
sc_X.transform([[6.5]])
regressor3.predict(sc_X.transform([[6.5]]))
sc_Y.inverse_transform(regressor3.predict(sc_X.transform([[6.5]])))


plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y),color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)),  color = 'black', label ='rbf')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor1.predict(X)), color = 'blue', label =label1 )
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor2.predict(X)), color = 'green', label =label2 )
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor3.predict(X)), color = 'Yellow', label = "tree")
plt.title("Poly Vs SVR Vs Tree",fontsize = 20)
plt.xticks(fontsize = 20)
plt.ylabel('Temperatura Maxima', fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 10)
plt.show()

#Calulos de R2
a=regressor.score(X,Y)
#Modelo Polinomial, grado 3
b=regressor1.score(X,Y)
#Modelo Polinomial, grado 5
c=regressor2.score(X,Y)
#Modelo DT
d=regressor3.score(X,Y)

print(a,b,c,d)

#%% Regresion Lineal/multilineal
regr = linear_model.LinearRegression()
 
# Entrenamos nuestro modelo

regr.fit(X_train, Y_train)
 
# Hacemos las predicciones 
y_pred = regr.predict(X_train)

# Calculo de Coeficientes
print('Coefficients: \n', regr.coef_)

# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)

# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(Y_train, y_pred))

# Puntaje de Varianza. El mejor puntaje es un 1.0
print('R2 score: %.2f' % r2_score(Y_train, y_pred))

# with statsmodels
X = sm.add_constant(X_train) # adding a constant
 
model = sm.OLS(Y_train, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

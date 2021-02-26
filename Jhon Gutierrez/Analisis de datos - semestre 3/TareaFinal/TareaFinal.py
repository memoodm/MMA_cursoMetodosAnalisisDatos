import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

white_wines = pd.read_csv('Winequality1.csv',sep=";") # Cargar Datos en Dataframe

print(white_wines.shape)
white_wines.head()

white_wines.isnull().any()

# Eliminar datos erroneos del DataFrame "Alcohol".
filas = []
for i in range(len(white_wines)):
  try:
    n = float(white_wines['alcohol'][i])
  except:
    filas.append(i)

white_wines = white_wines.drop(filas,axis=0)
white_wines['alcohol'] = pd.to_numeric(white_wines['alcohol'])
white_wines.describe()

sns.boxplot(white_wines["residual sugar"])
plt.show()

#La variable de "residual sugar" posee demasiados datos atipicos, los cuales se alejan de la media


plt.subplots(figsize=(15, 10))
sns.heatmap(white_wines.corr(), annot = True, cmap = "coolwarm")
plt.show()

# en el gráfico de correlación, podemos notar que las variables que presentan una correlación significativa son:
# 1. "fixed acidity" y "citric acid" con correlación positiva, son directamente proporcionales 
# 2. "fixed acidity" y "Ph" con correlación negativa, son inversamente proporcionales.
# 3. "citric acid" y "Ph" con correlación negativa, consecuencia de la observación 1.
# 4. "citric acid" y "volatile acidity" con correlación negativa.
# 5. la variable de "quality" no muestra una correlación significante con las variables, siendo la mayor a esta
#    con la variable de alcohol con un valor de 0.48.


# Define features X
X = np.asarray(white_wines.iloc[:,:-1])
# Define target y
y = np.asarray(white_wines["quality"])

#Normalizamos los datos
from sklearn import preprocessing
SC = preprocessing.StandardScaler().fit(X)
X = SC.transform(X)

#Dividimos el set de entrenamiento y validación. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
print ("Train set:", X_train.shape, y_train.shape)
print ("Test set:", X_test.shape, y_test.shape)

#Creamos el modelo de Regresión lineal múltiple para predecir la valoración de la calidad. 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

#Realizamos la predicción de la calidad para cada set. 
quality_y_pred_train = regr.predict(X_train)
quality_y_pred_test = regr.predict(X_test)

#Error de predicción sobre la calidad del vino.
print('Mean squared error Train: {:.3}'.format(mean_squared_error(quality_y_pred_train,y_train)))
print('Mean squared error Test: {:.3}'.format(mean_squared_error(quality_y_pred_test,y_test)))
import pandas as pd #importamos librer­as para analisis de datos
import numpy as np #librerÃ­a para anÃ¡lisis numÃ©rico
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
from patsy import dmatrices
os.chdir('C:\\Users\\Carlos Moreno\\Documents')#ruta del archivo donde esta alojado
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo
excel_file = "ADIC5.xlsx"
ad= pd.read_excel(excel_file)
vars=["QuantySold","Price","Advertising"]
ad=ad[vars]
ad[-3:]
ad = ad.dropna()
ad[-7:]
y,X= dmatrices("QuantySold ~ Price + Advertising", data=ad, return_type="dataframe")
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())
res.params
res.rsquared
sm.stats.linear_rainbow(res)
sm.graphics.plot_partregress('QuantySold', 'Price', 'Advertising', data = ad, obs_labels=False)

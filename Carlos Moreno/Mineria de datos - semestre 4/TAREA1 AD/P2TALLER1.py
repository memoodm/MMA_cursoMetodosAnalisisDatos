import pandas as pd #importamos librerÃ­as para analisis de datos
import numpy as np #librerÃ­a para anÃ¡lisis numÃ©rico
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from pandas_profiling import ProfileReport
from seaborn import lmplot
os.chdir("C:\\Users\\Casa\\Documents\PROGRMASCARLOS")#ruta del archivo donde esta alojado
cwd=os.getcwd() #Asign
excel_file = "Colombia-Feb21.xlsx" #asigno archivo
df = pd.read_excel(excel_file, header= 0)
df=df.replace({"BogotÃ¡, Distrito Especial, Colombia":"Bogotá","MedellÃ­n, Antioquia, Colombia":"Medellín", "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén"})
df = df.drop(['Organization Name URL'], axis=1)
df.head()
df.count()
df.describe()
df1=df.dropna()
pd.crosstab(index=df1["Headquarters Location"],columns=df1["Last Funding Amount"].sum)
plot = pd.crosstab(index=df1['Headquarters Location'].head(20),columns=df1['Last Funding Amount']).plot(kind='bar')
lmplot('Headquarters Location', 'Last Funding Amount', data=df1, ci=None)#se observa la forma en que se acumulan los datos
lmplot(x="Headquarters Location", y="Last Funding Amount", data=df1)
fig, ax = plt.subplots()
df1.plot(x = 'Headquarters Location', y = 'Last Funding Amount', ax = ax,kind="bar")
plt.show()
sns.swarmplot(x="Headquarters Location", y="Last Funding Amount", data=df1)#grafico de dispersión de las variablesdf1.groupby(["Headquarters Location"])["Last Funding Amount"].nunique().plot(kind='barh')

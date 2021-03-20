import pandas as pd #importamos librerÃ­as para analisis de datos
import numpy as np #librerÃ­a para anÃ¡lisis numÃ©rico
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
import seaborn as sns
os.chdir("C:\\Users\\Casa\\Documents\PROGRMASCARLOS")#ruta del archivo donde esta alojado
cwd=os.getcwd() #Asign
text_file = "Col.csv" 
Col=pd.read_csv(text_file)
Col
excel_file = "Top100.xlsx"
tp=pd.read_excel(excel_file)
Col['Bandera'] = 0#se coloca una columna bander
Col = Col.join(tp, rsuffix='_right')#se inserta la data de l top 100 a la derecha
print(Col)
df1=set(Col['Organization Name']).intersection(set(tp['Organization Name']))#se intersectan las dos para saber  que tienen de comun
for j in df1:#se relializa un ciclo para colocar 1 en l bandera que representa las que estan en las 2
    Col.loc[Col['Organization Name'] == j, ['Bandera']] = 1 
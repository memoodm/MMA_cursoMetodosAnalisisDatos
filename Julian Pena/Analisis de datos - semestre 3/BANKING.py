# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:13:52 2020

@author: Julian
"""

import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from sklearn import preprocessing # lib para preprocesar los datos
import os 
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns #graficas y estadistica
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


os.chdir('C:/Users/Julian/Documents/MMA/PYTHONLUZ')
cwd=os.getcwd() # asigan variable swd al directorio
text_file = "Banking.csv"

from sklearn.linear_model import LinearRegression #model resolve question 
from sklearn.linear_model import LogisticRegression #Regre logistica


data_frane = pd.read_csv(text_file)





print(data_frane.head(15))
print(data_frane.tail(15))

# ************clientes potenciales de un cdt ***********

# REG LOGISTICA PARA DEFINIR SI COMPRA O NO EL CDT 

plt.hist(data_frane["duration"])

print(data_frane["duration"].mean())
print(data_frane["duration"].describe())

print(data_frane.describe())

# Y variable binaria 0 si no lo compro 1 si lo compro 

## PERFILACION 


ProfileReport(data_frane)

print(data_frane["education"].unique()) #permite obtener categorias unicas

#agrupo los basic en una sola categoria 

data_frane["education"]=np.where(data_frane['education'] =='basic.9y', 'Basic', data_frane['education'])
data_frane["education"]=np.where(data_frane['education'] =='basic.6y', 'Basic', data_frane['education'])
data_frane["education"]=np.where(data_frane['education'] =='basic.4y', 'Basic', data_frane['education'])

print(data_frane["education"].unique() )

#cuento los que en la anterior campana dijero que si o no  al cdt

count_no_sub = len(data_frane[data_frane["y"]==0])
count_sub = len(data_frane[data_frane["y"]==1])
unt_no_sub =  count_no_sub / (count_no_sub+count_sub)
print("% no", unt_no_sub*100 )

unt_sub = count_sub / (count_no_sub+count_sub)
print("% si", unt_sub*100 )

#sns.countplot("y",data_frane=data_frane, palette="Blues")

#sns.countplot("job", data_frane=data_frane, palette='GnBu_d')


# ********* JOB **************
#pd.crosstab(data_frane.job,data_frane.y).plot(kind='bar')

# ********* MARITAL **********
#pd.crosstab(data_frane.marital,data_frane.y).plot(kind='bar')

# ******** EDUCATION ********
pd.crosstab(data_frane.previous,data_frane.y).plot(kind='bar')



plt.title('Purchase Frequency for PREVIOUS Title')

plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_VAR')








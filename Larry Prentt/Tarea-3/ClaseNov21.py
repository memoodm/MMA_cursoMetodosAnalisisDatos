# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:20:18 2020

@author: lprentt
"""
#pip install pandas_profiling

from pandas_profiling import ProfileReport
 
import pandas as pd # Importando libreria pandas para analisis de datos
import numpy as np  # Importando libreria numpy  para analisis numerico

import os # libreria para uso de comandos DOS - Sistema operativo
from sklearn import preprocessing

import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


os.chdir("D:/Analisis-Datos/python")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

text_file = 'Banking.csv'  # objeto string con el nombre del archivo a revisar # Es una bandera

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


data=pd.read_csv(text_file) # metodo read lee el excel como un dataframe. Por default
# lee la primera hoja

bank = ProfileReport(data)
bank.to_file('output.html')

# Exploraciones sobre el dataframe data

print(data.head(15))
print(data.tail(15))

# movies.fillna(0, inplace=True) # Asigna cero a todos los datos del dataframe que fueron cargados como NaN
data.shape  # imprime las dimensiones del dataframe

# organizando df por ingreso gross

data['education'].unique()

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])
# se puede usar replace en vez de where
data['education'].unique()

count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

sns.countplot(x='y',data=data, palette='Blues')

sns.countplot(y="job", data=data, palette='GnBu_d')

sns.countplot(y="marital", data=data, palette='cubehelix')


# Grafica 1 Y vs Job
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_job')

# Grafica 2 Y vs Job
pd.crosstab(data.age,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Age')
plt.xlabel('age')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_Age')

# Grafica 3 Y vs Marital
pd.crosstab(data.marital,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Marital status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_marital')

# Grafica 4 Y vs Education
pd.crosstab(data.education,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Education')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_education')

# Grafica 5 Y vs default
pd.crosstab(data.default,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Default')
plt.xlabel('Default')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_default')

# Grafica 6 Y vs Housing
pd.crosstab(data.housing,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Housing')
plt.xlabel('Housing')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_housing')

# Grafica 7 Y vs Loan
pd.crosstab(data.loan,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Loan')
plt.xlabel('Loan')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_loan')

# Grafica 8 Y vs contact
pd.crosstab(data.contact,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Contact Type')
plt.xlabel('contact')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_contact')

# Grafica 9 Y vs Month
pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Month of Contact')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_month')

# day_of_week
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs day_of_week to Contact')
plt.xlabel('day_of_week')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_day_of_week')

#campaign
# Grafica 11 Y vs campaign
pd.crosstab(data.campaign,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs # of campaign')
plt.xlabel('campaign')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_campaign')

#pdays
pd.crosstab(data.pdays,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs pdays')
plt.xlabel('pdays')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_pdays')

# previous
pd.crosstab(data.previous,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs Previous')
plt.xlabel('previous')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_previous')

# poutcome
pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs poutcome')
plt.xlabel('poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_poutcome')


# duration
data.plot.scatter(x='duration', y='y')
# emp_var_rate
data.plot.scatter(x='emp_var_rate', y='y')

pd.crosstab(data.emp_var_rate,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs emp_var_rate')
plt.xlabel('emp_var_rate')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_emp_var_rate')

# cons_price_idx
data.plot.scatter(x='cons_price_idx', y='y')

pd.crosstab(data.cons_price_idx,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs cons_price_idx')
plt.xlabel('cons_price_idx')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_cons_price_idx')

# cons_conf_idx
data.plot.scatter(x='cons_conf_idx', y='y')

pd.crosstab(data.cons_conf_idx,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs cons_conf_idx')
plt.xlabel('cons_conf_idx')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_cons_conf_idx')



# euribor3m
data.plot.scatter(x='euribor3m', y='y')

# nr_employed
data.plot.scatter(x='nr_employed', y='y')

pd.crosstab(data.nr_employed,data.y).plot(kind='bar')
plt.title('Purchase Frequency vs nr_employed')
plt.xlabel('nr_employed')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_for_nr_employed')

# %matplotlib qt


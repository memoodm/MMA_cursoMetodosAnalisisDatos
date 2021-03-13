import pandas as pd #importamos librerÃ­as para analisis de datos
import numpy as np #librerÃ­a para anÃ¡lisis numÃ©rico
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import altair as alt
from pandas_profiling import ProfileReport
os.chdir("C:\\Users\\Casa\\Documents\PROGRMASCARLOS")#ruta del archivo donde esta alojado
cwd=os.getcwd() #Asign
text_file = "Col.csv" 
Col=pd.read_csv(text_file)
Col
text_file = "Arg.csv" 
Arg=pd.read_csv(text_file)
Braz=pd.read_csv(text_file)
Chi=pd.read_csv(text_file)
Uru=pd.read_csv(text_file)
Mex=pd.read_csv(text_file)
print(Col.isnull().sum())
dt=[Col,Braz,Chi,Uru,Mex,Arg]
dt=pd.concat([Col,Braz,Chi,Uru,Mex,Arg])
dt
for i in dt:
    print(i.isnull().sum())
for i in dt:
    print(i.columns)

df=dt.drop(columns=["Organization Name URL","Website","Facebook","Twitter","Contact Email","Phone Number","LinkedIn"])


df1=df["Headquarters Location"].str.split(",", n = 2, expand = True)
df["Ciudad"]= df1[0]
df["Departamento"]= df1[1]
df["Pais"]= df1[2]
df.shape
a=df.describe()
df.count()
df.info()
#Grafico
df['Last Funding Date'] = pd.to_datetime(df['Last Funding Date'])
df['year']= df['Last Funding Date'].dt.year
tabla2 = pd.pivot_table(df, 'Last Funding Amount','year','Headquarters Location', aggfunc=np.sum )
tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('LFA USD')
plt.title('Last Funding Amount - year - Headquarters Location')
plt.show()


# plot = df['Headquarters Location'].value_counts(10).plot(kind='bar',title='Ciudades')
# plot = df['Industries'].value_counts(10).plot(kind='bar',title='Ciudades')
# pd.crosstab(index=df['Headquarters Location'],columns=df['Industries'], margins=True)#tabal de contingencia de dos variables categoricas
# plot = pd.crosstab(index=df['Headquarters Location'],columns=df['Industries']).apply(lambda r: r/r.sum() *100,axis=1).plot(kind='bar')#grafica de variables categoricasdef distribucion_variable_categorica(col):
sns.scatterplot(df['Industries'].head(50), df['Headquarters Location'].head(50))#grafico de dispersion
def distribucion_variable_categorica(col): 
    df[col].value_counts(ascending=True,normalize=True).tail(20).plot.bar()
    plt.xlabel('ciudades')
    plt.ylabel('empresas')
    plt.title('Industrias')
    plt.show()
distribucion_variable_categorica('Industries')#grafico de barras     

def distribucion_variable_categorica(col): 
    df[col].value_counts(ascending=True,normalize=True).tail(20).plot.bar()
    plt.xlabel('Nombre de la ciudad')
    plt.ylabel('Distribución')
    plt.title('Ciudades')
    plt.show()
distribucion_variable_categorica('Headquarters Location')#grafico de barras     

def distribucion_variable_categoric(col): 
    df[col].value_counts(ascending=True,normalize=True).tail(20).plot.bar()
    plt.xlabel('Nombre de la ciudad')
    plt.ylabel('Distribución')
    plt.title('Ciudades')
    plt.show()
distribucion_variable_categorica('Last Funding Amount')#gr
df1=pd.concat([Col,Braz,Chi,Uru,Mex,Arg])
df1.hist()
df1_sorted_by_Number_of_Employees=df1.sort_values(["Number of Employees"], ascending=True)
                                                                                                                               df1_sorted_by_CB_Rank = df1.sort_values(["CB Rank (Company)"], ascending=True)
df1_sorted_by_Estimated_Revenue_Range = df1.sort_values(["Estimated Revenue Range"], ascending=True)
df1_sorted_by_Number_of_Articles = df1.sort_values(["Number of Articles"], ascending=True)
df1_sorted_by_Investment_Stage = df1.sort_values(["Investment Stage"], ascending=True)
df1_sorted_by_Number_of_Funding_Rounds = df1.sort_values(["Number of Funding Rounds"], ascending=True)
df1_sorted_by_Last_Funding_Amount= df1.sort_values(["Last Funding Amount"], ascending=True)
df1_sorted_by_Last_Funding_Type= df1.sort_values(["Last Funding Type"], ascending=True)
df1_sorted_by_Last_Equity_Funding_Amount= df1.sort_values(["Last Equity Funding Amount"], ascending=True)
df1_sorted_by_Number_of_Investments= df1.sort_values(["Number of Investments"], ascending=True)



pd.crosstab(df1_sorted_by_CB_Rank["CB Rank (Company)"].head(10),df1_sorted_by_CB_Rank["Headquarters Location"].head(10)).plot(kind="bar")
plt.xlabel('CB Rank (Company)')
plt.ylabel('Headquarters Location')
plt.title('Compañias')
plt.show()

pd.crosstab(df1_sorted_by_Number_of_Articles["Number of Articles"].head(10),df1_sorted_by_Number_of_Articles ["Last Funding Type"].head(10)).plot(kind="bar")
plt.xlabel('Estimated Revenue Range')
plt.ylabel('Headquarters Location')
plt.title('')
plt.show()


pd.crosstab(df1_sorted_by_Last_Equity_Funding_Amount["Last Equity Funding Amount"].head(10),df1_sorted_by_Last_Equity_Funding_Amount["Last Funding Type"].head(10)).plot(kind="bar")
plt.xlabel('Last Equity Funding Amount')
plt.ylabel('Last Funding Type')
plt.title('Comparación de dos variables')
plt.show()


pd.crosstab(df1_sorted_by_Number_of_Employees["Number of Employees"].head(10),df1_sorted_by_Number_of_Employees["Last Funding Amount"].head(10)).plot(kind="bar")
plt.xlabel('Number of Employees')
plt.ylabel('Last Funding Amount')
plt.title('Comparación de dos variables')
plt.show()

pd.crosstab(df1_sorted_by_Number_of_Funding_Rounds["Number of Funding Rounds"].head(50),df1_sorted_by_Number_of_Funding_Rounds["Number of Employees"].head(50)).plot(kind="bar")
plt.title('Number_of_Funding_Rounds Vs Number of Employees ')
plt.xlabel("Number_of_Funding_Rounds")
plt.ylabel("Number of Employees")
plt.show()


pd.crosstab(df1_sorted_by_Number_of_Articles["Number of Articles"].head(50),df1_sorted_by_Number_of_Articles["Number of Funding Rounds"].head(50)).plot(kind="bar")
plt.title('Number of Articles vs Number of Funding Rounds ')
plt.xlabel("Number of Articles")
plt.ylabel("Number of Funding Rounds")
plt.show()

pd.crosstab(df1_sorted_by_Number_of_Investments["Number of Articles"].head(50),df1_sorted_by_Number_of_Investments["Number of Investments"].head(50)).plot(kind="bar")
plt.title('Number of Articles vs Numberv of Investments ')
plt.xlabel("Number of Articles")
plt.ylabel("Numberv of Investments")
plt.show()
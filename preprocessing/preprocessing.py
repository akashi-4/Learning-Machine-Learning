import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the data
base_credit = pd.read_csv('credit_data.csv')

#print(base_credit.head()) # Show the first 5 rows of the dataset

#print(base_credit.tail()) # Show the last 5 rows of the dataset

#print(base_credit.describe()) # Show the statistics of the dataset

#print(base_credit.info()) # Show the information of the dataset

#print(np.unique(base_credit['default'], return_counts=True)) # Show how many people pay and not pay the credit

#sns.countplot(x = base_credit['default']) # Show the countplot of the default column

#plt.hist(base_credit['age']) # Show the histogram of the age column



#Treating wrong ages
#print(base_credit.loc[base_credit['age']<0]) # Show the rows with age less than 0
# or base_credit[base_credit['age']<0]

#Delete the columns with age less than 0
#base_credit.drop(base_credit[base_credit['age']<0].index, inplace=True)


#print(base_credit.loc[base_credit['clientid']==16])

def showPlot():
    graphic = px.scatter_matrix(base_credit, dimensions=['income', 'age', 'loan'], color='default') # Create a scatter matrix of the income, age and loan columns, differentiating by the default column
    graphic.show() # Show the scatter matrix of the age column

#Fill the missing values manually or with the mean
def correctNegativeAges():
    meanAges = base_credit['age'][base_credit['age']>0].mean()
    base_credit.loc[base_credit['age']<0,'age'] = meanAges

#Count the missing values
def countMissingValues():
    print(base_credit.isnull().sum())

def correctMissingValues():
    base_credit['age'].fillna(base_credit['age'].mean(), inplace=True) # Fill the missing values in the age column with the mean of the column

correctNegativeAges()
countMissingValues()

print(base_credit.loc[pd.isnull(base_credit['age'])])# Show the rows with missing values in the age column

correctMissingValues()

#print(base_credit.loc[base_credit['clientid'].isin([29,31,32])])  # Show the rows with clientid 29, 31 and 32  

X_credit = base_credit.iloc[:,1:4].values # Create a matrix with the columns 1, 2 and 3 of the dataset, unique values shouldnt be included, eg: clientid

Y_credit = base_credit.iloc[:,4].values # Create a matrix with the column 4 of the dataset, which is default.

# Escalonamento de variáveis, padronização(x - média(x))/desvio padrão(x)) ou normalização(x - min(x))/(max(x) - min(x))
#                              Standardization                                Normalization
# 
from sklearn.preprocessing import StandardScaler

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
# Agora as variáveis estão padronizadas.

base_census = pd.read_csv('census.csv')

#sns.countplot(x = base_census['income'])

#plt.hist(base_census['age'])
#plt.show()

#grafico = px.parallel_categories(base_census, dimensions=['workclass','occupation', 'income'])
#grafico.show()

X_census = base_census.iloc[:,0:14].values # Create a matrix with the columns 0 to 13 of the dataset, unique values shouldnt be included, eg: clientid
Y_census = base_census.iloc[:,14].values # Create a matrix with the column 14 of the dataset, which is income.

# Tratar variáveis categóricas, geralmente são strings, logo vamos transformar em números
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder() # É preciso criar um label encoder para cada variável categórica
label_encoder_relationship = LabelEncoder() 
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

# print(X_census[0]) conseguimos ver que as variáveis categóricas foram transformadas em números!

# Uma das desvantagens de usar só o label encoder é que o algoritmo pode entender que um número maior é mais importante que um número menor, o que não é verdade.
# Para resolver isso, vamos usar o OneHotEncoder, que cria uma coluna para cada valor da variável categórica, e atribui 0 ou 1 para cada uma delas.

from sklearn.compose import ColumnTransformer

onehotenconder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
# Agora cada valor da variável categórica foi transformado em uma coluna, e cada linha tem 0 ou 1 para cada uma delas.
# O hot encoder cria uma matriz esparsa, que é uma matriz com muitos zeros, para economizar memória.
# O column transformer é usado para transformar várias colunas ao mesmo tempo.
X_census = onehotenconder_census.fit_transform(X_census).toarray()
# Agora transformamos para uma matriz do numpy, para que possamos usar no algoritmo.
#print(X_census)
# Agora vamos ter que fazer o escalonamento das variáveis, para que elas tenham o mesmo peso.

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

#print(X_census) # Agora temos as variáveis categóricas transformadas em números e escalonadas.

# Avaliação de algoritmos: 

# Dividir a base de dados em treino e teste
from sklearn.model_selection import train_test_split

X_credit_train, X_credit_test, Y_credit_train, Y_credit_test = train_test_split(X_credit, Y_credit, test_size=0.25, random_state=0) # Dividir a base de crédito em treino e teste
# Random state é para que sempre tenhamos o mesmo resultado, e o test size é o tamanho da base de teste.

X_census_train, X_census_test, Y_census_train, Y_census_test = train_test_split(X_census, Y_census, test_size=0.15, random_state=0) # Dividir a base de censo em treino e teste

# Pré-processamento feito. Agora vamos salvar as bases de dados pré-processadas.
import pickle # Biblioteca para salvar os dados
with open('credit.pkl', 'wb') as f:
    pickle.dump([X_credit_train, Y_credit_train, X_credit_test, Y_credit_test], f)

with open('census.pkl', 'wb') as f:
    pickle.dump([X_census_train, Y_census_train, X_census_test, Y_census_test], f)

# Naive Bayes

# Base de dados de risco de crédito
base_risco_credito = pd.read_csv('risco_credito.csv')
X_risco_credito = base_risco_credito.iloc[:,0:4].values
Y_risco_credito = base_risco_credito.iloc[:,4].values

#Converter strings para números

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantia.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_historia.fit_transform(X_risco_credito[:,3])
#Criar ficheiro da base de dados
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, Y_risco_credito], f)

from sklearn.naive_bayes import GaussianNB

naive_risco_credito = GaussianNB()
# Treinar o modelo ou seja, gerar a tabela de probabilidades
naive_risco_credito.fit(X_risco_credito, Y_risco_credito)

previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
print(previsao) # [1 0] significa que o primeiro cliente é bom e o segundo é mau.

#naive_risco_credito.classes_ # Mostra as classes do modelo, ou seja, os valores que a variável target pode assumir.


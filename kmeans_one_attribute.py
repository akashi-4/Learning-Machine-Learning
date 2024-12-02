import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Agrupamento de dados geralmente necessita de normalização por isso usamos o StandardScaler

from sklearn.cluster import KMeans

def basic_kmeans():
    # Gerando dados aleatórios
    x = [20, 22, 25, 27, 21, 37, 45, 53, 55, 52, 48, 50, 5, 70, 65, 72, 73, 75]
    y = [1000, 1200, 1500, 1520, 2200, 2300, 2000, 2200, 2500, 2700, 3000, 3200, 3500, 3700, 4000, 4200, 4500, 4700]


    #grafico = px.scatter(x=x, y=y, title='Dados Originais') # Gráfico com os dados originais
    #grafico.show()

    # Normalizando os dados
    salary_data = np.array(list(zip(x, y)))
    scaler = StandardScaler()
    salary_data = scaler.fit_transform(salary_data)

    # KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(salary_data)

    centroids = kmeans.cluster_centers_

    inversed_centroids = scaler.inverse_transform(centroids)

    rotulos = kmeans.labels_

    grafico = px.scatter(x=salary_data[:, 0], y=salary_data[:, 1], color=rotulos)

    grafico2 = px.scatter(x=centroids[:, 0], y=centroids[:, 1], color=[0, 1, 2], size=[12, 12, 12])
    grafico3 = go.Figure(data=grafico.data + grafico2.data)
    grafico3.show()

def random_kmeans():
    from sklearn.datasets import make_blobs
    # Gerando dados aleatórios, simulando 5 clusters e 200 amostras como se fossem salários e idades
    X, y = make_blobs(n_samples=200, centers=5, random_state=42)
    
    grafico = px.scatter(x=X[:, 0], y=X[:, 1], color=y)
    #grafico.show()

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)

    rotulos = kmeans.labels_
    centroids = kmeans.cluster_centers_

    grafico = px.scatter(x=X[:, 0], y=X[:, 1], color=rotulos)
    graficoCENTROIDS = px.scatter(x=centroids[:, 0], y=centroids[:, 1], color=[0, 1, 2, 3, 4], size=[10, 10, 10, 10, 10])
    graficoGO = go.Figure(data=grafico.data + graficoCENTROIDS.data)
    graficoGO.show()

def kmeans_alg(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    rotulos = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    grafico = px.scatter(x=X[:, 0], y=X[:, 1], color=rotulos)
    graficoCENTROIDS = px.scatter(x=centroids[:, 0], y=centroids[:, 1], color=[0, 1, 2, 3, 4], size=[10, 10, 10, 10, 10])
    graficoGO = go.Figure(data=grafico.data + graficoCENTROIDS.data)
    return graficoGO, rotulos

def kmeans_clust(X):
    print("KMeans")
    print("Sabe quantos clusters temos? (S/N)")
    resposta = input()
    if resposta == 'S' or resposta == 's':
        print("Quantos clusters quer?")
        n_clusters = int(input())
        graficoGO, rotulos = kmeans_alg(X, n_clusters)
        return graficoGO, rotulos
    else:
        print("Vamos calcular o número de clusters")
        print("usando o elbow method")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        grafico = px.line(x=range(1, 11), y=wcss)
        grafico.show()
        print("Com base no gráfico, quantos clusters quer?")
        n_clusters = int(input())
        graficoGO, rotulos = kmeans_alg(X, n_clusters)
        return graficoGO , rotulos

def credit_kmeans():
    credit_base = pd.read_csv('content\credit_card_clients.csv')
    # Criando um novo atributo que é a soma de todas as faturas
    credit_base['FaturaTotal'] = credit_base['BILL_AMT1'] + credit_base['BILL_AMT2'] + credit_base['BILL_AMT3'] + credit_base['BILL_AMT4'] + credit_base['BILL_AMT5'] + credit_base['BILL_AMT6']
    X = credit_base.iloc[:, [1, 25]].values # Limite e fatura total
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # X já está normalizado
    grafico , rotulos = kmeans_clust(X)
    print("Quer mostrar o gráfico? (S/N)")
    resposta = input()
    if resposta == 'S' or resposta == 's':
        grafico.show()
    else:
        print("Ok, sem gráfico")

    clients_list = np.column_stack((credit_base, rotulos))
    clients_list = clients_list[clients_list[:, 26].argsort()]

credit_kmeans()
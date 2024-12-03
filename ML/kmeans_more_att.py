import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans

def elbow_method(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    grafico = px.line(x=range(1, 11), y=wcss)
    grafico.show()

def kmeans_calc():
    credit_data = pd.read_csv('content/credit_card_clients.csv')
    credit_data['BILL_TOTAL'] = credit_data['BILL_AMT1'] + credit_data['BILL_AMT2'] + credit_data['BILL_AMT3'] + credit_data['BILL_AMT4'] + credit_data['BILL_AMT5'] + credit_data['BILL_AMT6']
    X = credit_data.iloc[:,[1,2,3,4,5,25]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    elbow_method(X)
    print("How many clusters do you want?")
    n_clusters = int(input())
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    from sklearn.decomposition import PCA
    # Reduzindo a dimensionalidade para 2D
    pca = PCA(n_components=2) # Enviamos a base para 2D
    X_pca = pca.fit_transform(X)
    grafico = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=labels)
    grafico.show()
    
kmeans_calc()


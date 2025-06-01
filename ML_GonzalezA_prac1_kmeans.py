# -*- coding: utf-8 -*-

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import random as rd

"""
coord= [[1,1],[2,1],[2,2],[1,2],[0.5,1.5]]

coord.append(coord[0]) #repeat the first point to createa closed loop
xs, ys = zip(*coord) # takes the list coord direction and create lists of x and y values

plt.figure()
plt.plot(xs, ys)
plt.scatter(xs, ys)
plt.show()
"""
#Generar Bases de datos artificiales 
from sklearn.datasets import make_blobs

plt.figure()
X, y_true = make_blobs(n_samples=300, centers = 3, cluster_std=.90, random_state=3)
plt.scatter(X[:, 0], X[:, 1], s=50)


#X es un vector de posiciones (x,y), primero escojemos tres puntos aleatorios 
k=3 #definimos el número de clusters y centroides
N=len(X) #tamaño del arreglo

random_indices = np.random.choice(X.shape[0], size=k, replace=False) # elegimos aleatoriamente k índices del arreglo X
centroids = X[random_indices] #Extraemos la información de los índices 

###para asignar a qué cluster pertenecen calculamos distancias 

def compute_dist(a,b): #función que calcula la norma al cuadrado entre puntos a y b
    dist = pow(b[0]-a[1],2) + pow(b[0]-a[1], 2)
    return dist


iteraciones = 100

for n in range(iteraciones):
    distances = []
    for point in X:
        distances_centroids = [compute_dist(point, c) for c in centroids] #calcula la distancia de los puntos con cada centroide
        distances.append(distances_centroids)
        
    distances = np.array(distances) 
    
    clusters = []
    for i, point in enumerate(X):
        closest_centroid_index = np.argmin(distances[i])# Encuentra el índice del centroide más cercano al punto
                                                        #análogo a la tabla de 1 y 0 de membresía a un cluster
        clusters.append(closest_centroid_index)
    
    clusters = np.array(clusters)
    
    new_centroids = []
    for cluster_index in range(len(centroids)):
        puntos_clusters = X[clusters == cluster_index] # Filtra los puntos que pertenecen al cluster actual
        new_centroid = np.mean(puntos_clusters, axis=0)# Calcula el nuevo centroide como el promedio de los puntos en el cluster
        new_centroids.append(new_centroid)
    
    new_centroids = np.array(new_centroids)
    
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], color='red') #grafica los nuevos centroides óptimos 

plt.show()

print("Centroides: \n", new_centroids)



"""
#clustering algoritmo K-means
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=3)
"""


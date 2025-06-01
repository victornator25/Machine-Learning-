# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:12:01 2023

@author: TUF-PC8

@author: Víctor Manuel González Aguayo
NUA: 314509
Machine Learning, Práctica #2 PCA
Catedrático: Luis Carlos Padierna
"""


import numpy as np
import matplotlib.pyplot as plt

#Dataset (D)
D = np.array([[10,30], [20,40], [30,35], [40,45], [50,50]])

#Paso 1. Calculando Vector de medias
mean = np.mean(D, axis =0)

#Paso 2. Calculando Matriz de covarianza de la matriz D
C = np.cov(D, rowvar=False, bias=True)

#Paso 3. Calculando eigenvalores y eigenvectores
eigval, eigvec = np.linalg.eig(C)

# Grafica del conjunto de datos
plt.figure()
plt.scatter(D[:,0],D[:,1])


# Gráfica de eigenvectores centrados en la media, escalados por la raíz cuadrada de sus eigenvalores
plt.arrow(*mean, *eigvec[:, 0]*np.sqrt(eigval[0]), width=0.1, color="k", lw=2,
          overhang=0.1)
plt.arrow(*mean, *eigvec[:, 1]*np.sqrt(eigval[1]), width=0.1, color="r", lw=2,
          overhang=0.1)
plt.grid(True)
plt.axis('equal')
#plt.show()

#Proyectando a la nueva dimensión
#De Tarea...

#nos quedamos con el eigenvector de mayor dimensión dado que es el de mayor varianza de los puntos
max_eigvec_index = np.argmax(np.abs(eigvec[:, 0]))  # Índice del eigenvector con la magnitud más grande
max_eigvec = eigvec[:, max_eigvec_index] #vector de mayor magnitud


# Proyectar puntos en D sobre el eigenvector con mayor eigenvalor

"""
projection = (np.dot(D, max_eigvec)/(np.linalg.norm(max_eigvec))**2)*max_eigvec

###   Marcó error de broadcasting:
ValueError: operands could not be broadcast together with shapes (5,) (2,)     
"""

#solución encontrada:
"""
projection = np.dot(D, max_eigvec)  # Proyección en el eigenvector más grande
projection = np.outer(projection, max_eigvec)  

plt.scatter(projection[:, 0], projection[:, 1], c='b', marker='x', label='Proyección')
"""

#se le resta la media a la matriz de covarianzas para centrar la nueva proyección
  
projection = np.dot(D - mean, max_eigvec)  # Proyección en el eigenvector más grande

projection = np.outer(projection, max_eigvec) #multiplicación elemento a elemento resulta en matriz de 10x2(producto externo) 
plt.scatter(projection[:, 0] + mean[0], projection[:, 1] + mean[1], c='b', marker='x', label='Proyección')

plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()



"""
#Implementación de PCA en sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(D)

print(pca.components_)
print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.figure()
plt.scatter(D[:, 0], D[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector *  np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
    print('printing  vector....',v)
plt.grid(True)
plt.axis('equal');

pca = PCA(n_components=1)
pca.fit(D)
X_pca = pca.transform(D)
print("original shape:   ", D.shape)
print("transformed shape:", X_pca.shape)
print(pca.explained_variance_)
X_new = pca.inverse_transform(X_pca)
plt.scatter(D[:, 0], D[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');

"""



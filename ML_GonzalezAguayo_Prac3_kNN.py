# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:06:31 2023

@author: Víctor Manuel González Aguayo
NUA: 314509
Machine Learning, Práctica #3 (Autónoma) K-NN
Catedrático: Luis Carlos Padierna
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
"""
1. Mediante la función make_blobs del paquete sklearn.datasets, genere 250 puntos divididos en 3 
clases con desviación estándar de 0.5. Utilize un random_state = 209 para que podamos comparar resultados.
Guarde las coordenadas de los primeros 200 puntos en una variable llamada training_data y los indicadores de
clase en una variable llamada training_labels. Grafique un scatter plot usando un color distinto para los puntos 
de entrenamiento en cada una de las 3 clases. Los 50 puntos restantes los guardará en una variable llamada test_data 
y sus indicadores de clase en una variable llamada test_labels
"""
#definimos k
k=5
n_clases=3#número de cúmulos de la generación de los datos
X, y = make_blobs(n_samples=250, n_features=2, centers=n_clases, cluster_std=0.5, random_state=209)
#n clases son los centros, en este caso =3

n_training_samples = 200
training_data = X[:n_training_samples]
training_labels = y[:n_training_samples]
test_data = X[n_training_samples:]
test_labels = y[n_training_samples:]

# Graficamos el scatter plot
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b'] #graficar de diferentes colores
class_labels = ['Clase 0', 'Clase 1', 'Clase 2']

for i in range(n_clases):
    plt.scatter(training_data[training_labels == i][:, 0],
                training_data[training_labels == i][:, 1],
                c=colors[i], label=class_labels[i])

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.title('Scatter de datos')
plt.show()


"""
2.Implemente el algoritmo k-NN siguiendo el pseudocódigo visto en clase y usando una distancia Euclidiana
"""
# Función que calcula la distancia euclidiana
def euclid_dist(a, b):
    dist = np.sum((b - a) ** 2) #como aparece en el artículo de IBM
    #vimos en clase qu eno afecta si es la norma al cuadrado para 
    #ahorrar cósto de cómputo
    return dist


"""
3. Entrene su clasificador k-NN, con k=5 y training_data y grafique en el
 mismo scatter plot del paso 1 las predicciones de este clasificador sobre el conjunto test_data.
"""
 


def KNN(test_data, training_data, training_labels, k): 
    
    for sample in test_data:
        distances = [euclid_dist(sample, x) for x in training_data] #calcula la distancia euclidiana entre los datos de entrenamiento con los de test
        k_indices = np.argsort(distances)[:k] #irdena los índices de las distancias de menor a mayor
        k_nearest_labels = [training_labels[i] for i in k_indices] #extrae las etiquetas de las k distancias más cercanas
        most_common = np.bincount(k_nearest_labels).argmax() #extraemos las etiquetas que más se repiten
        predictions.append(most_common) #regresa las etiquetas más comunes

    return predictions

predictions = []
predictions = KNN(test_data, training_data, training_labels, k)



"""
# Crear y entrenar el clasificador KNN con biblioteca sklearn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(training_data, training_labels)

# Realizar predicciones en el conjunto de prueba
predictions = knn.predict(test_data)

"""




# Scatter plot de todos los puntos de entrenamiento
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
class_labels = ['Clase 0', 'Clase 1', 'Clase 2']

for i in range(n_clases):
    plt.scatter(training_data[training_labels == i][:, 0],
                training_data[training_labels == i][:, 1],
                c=colors[i], label=class_labels[i], alpha=0.4) #los hacemos más ténues

# Scatter plot de todos los puntos de predicciones
for i in range(n_clases):
    plt.scatter(test_data[np.array(predictions) == i][:, 0],
                test_data[np.array(predictions) == i][:, 1],
                c=colors[i], marker='^', s=50, label=f'Predicciones Clase {i}',alpha=1) #scatter triángular 

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.title('Puntos de Datos y Predicciones KNN')
plt.show()


# Generar y mostrar un informe de clasificación
report = classification_report(test_labels, predictions)
print("Informe de Clasificación:")
print(report)









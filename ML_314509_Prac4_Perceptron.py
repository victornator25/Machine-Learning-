# -*- coding: utf-8 -*-
"""
Victor Manuel Gonzalez Aguayo
Machine Learning ago-dic 2023
Catedratico: Luis Carlos Padierna

Práctica Autónoma 4: Clasificación AND con Neurona McCulloch-Pitts entrenada con Perceptron
"""


"""
1.-Construya un dataset artificial de 40 puntos usando el método make_blobs de sklearn.datasets.
El dataset tendrá como centros los 4 puntos de valores posibles a una compuerta lógica AND. Esto es 
centers = [(0, 0), (0, 1), (1, 0), (1,1)]. Use una media  de 0 y desviación estándar de 0.1 en cada cluster

"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

# Definir los centros de los clusters
centers = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])

# Crear el dataset artificial
X, y = make_blobs(n_samples=40, centers=centers, cluster_std=0.1)

# Ajustar las etiquetas para que representen la compuerta lógica AND
y[y == 0] = 1
y[y == 1] = 1
y[y == 2] = 1
y[y == 3] = 0

"""
#plot del dataset
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Clase 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Clase 1')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.title('Dataset Artificial - Compuerta lógica AND')
plt.show()
"""


"""
2. Implemente una Neurona de McCulloch-Pitts que reciba dos valores de entrada (sin bias) y utilice
 una función de activación de tipo escalón (0 o 1). Entrene la neurona usando el algoritmo
 Perceptron y el dataset de 40 puntos

"""

# Función de activación tipo escalón
def step_function(x):
    return 1 if x >= 0 else 0

# Inicialización de pesos y sesgo
w = np.random.rand(2)  # Inicializamos pesos aleatorios
b = 0  # Sin sesgo

# Hiperparámetros
learning_rate = 0.01
epochs = 1000

# Entrenamiento del Perceptrón
for epoch in range(epochs):
    for i in range(len(X)):
        input_data = X[i]
        target = y[i]

        # Calcular la salida de la neurona
        output = step_function(np.dot(w, input_data)) #+b dentro del paréntesis

        # Actualizar los pesos y sesgo
        w += learning_rate * (target - output) * input_data
        #b += learning_rate * (target - output)

# Scatter plot del dataset
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Clase 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Clase 1')

# Trazar la recta que separa las clases
x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
y_vals = (-w[0] / w[1]) * x_vals - b / w[1]
plt.plot(x_vals, y_vals, 'g--', label='Recta Separadora')


plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.title('Clasificación compuerta AND con bias')
plt.show()


#reporte del desempeño en la clasificación


#lista de predicciones
predictions = [step_function(np.dot(w, input_data)) for input_data in X]

# Generar el informe de clasificación
report = classification_report(y, predictions)

print("Informe de Clasificación:")
print(report)




# Crear la matriz de confusión
cm = confusion_matrix(y, predictions)

# Configurar el gráfico de la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")

# Mostrar el gráfico
plt.show()
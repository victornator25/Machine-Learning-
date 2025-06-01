import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd

# Generar un dataset de ejemplo con 600 puntos en R^2
#dataset1, _ = make_blobs(n_samples=600, centers=5, random_state=42)

ruta = r'C:\Users\victo\OneDrive\Escritorio\9_sem_DCI\Machine Learning\Exam_Prac_1\dataset1.xlsx'

data = pd.read_excel(ruta)
df = pd.DataFrame(data=data, columns=["X", "Y"])
 
# Aplicar K-Means con 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(df)

# Obtener los centros de los grupos
centros = kmeans.cluster_centers_

# Imprimir los centroides
print("Centroides de los grupos:")
for i, centro in enumerate(centros):
    print(f"Grupo {i+1}: {centro}")

# Graficar los puntos originales y los centros encontrados
plt.scatter(df['X'], df['Y'], c=kmeans.labels_, cmap='viridis')
plt.scatter(centros[:, 0], centros[:, 1], c='red', marker='x', s=200)
plt.title('Puntos y Centroides encontrados')
plt.show()



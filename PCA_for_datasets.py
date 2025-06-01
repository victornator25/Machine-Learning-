import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ruta = r'C:\Users\victo\OneDrive\Escritorio\9_sem_DCI\Machine Learning\Exam_Prac_1\dataset1.xlsx'

data = pd.read_excel(ruta)
df = pd.DataFrame(data=data, columns=["X", "Y"])

# Paso 1: Preprocesamiento de datos (Estandarización)
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(df)

# Paso 2: Aplicar PCA para encontrar las dos primeras componentes principales
pca = PCA(n_components=2)
principal_components = pca.fit_transform(dataset_scaled)

# Crear un nuevo DataFrame con las dos primeras componentes principales
principal_df = pd.DataFrame(data=principal_components, columns=['Componente 1', 'Componente 2'])

# Paso 3: Graficar el scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(principal_df['Componente 1'], principal_df['Componente 2'], c='b', marker='o')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Scatterplot de las 2 Componentes Principales')
plt.grid()
plt.show()
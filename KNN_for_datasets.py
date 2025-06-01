import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Cargar el dataset de entrenamiento
ruta1 = r'C:\Users\victo\OneDrive\Escritorio\9_sem_DCI\Machine Learning\Exam_Prac_1\dataset3_train.xlsx'
datos_train = pd.read_excel(ruta1, header=None, names=['Feature1', 'Feature2', 'Label'])

# Cargar el dataset de prueba
ruta2 = r'C:\Users\victo\OneDrive\Escritorio\9_sem_DCI\Machine Learning\Exam_Prac_1\dataset3_test.xlsx'
datos_test = pd.read_excel(ruta2, header=None, names=['Feature1', 'Feature2', 'Label'])

df_train = pd.DataFrame(data=datos_train)
df_test = pd.DataFrame(data=datos_test)


# Separar características (features) y etiquetas (labels) para entrenamiento y prueba
X_train = df_train[['Feature1', 'Feature2']]
y_train = df_train['Label']
X_test = df_test[['Feature1', 'Feature2']]
y_test = df_test['Label']

# Crear el clasificador k-NN con k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Realizar predicciones en el dataset de prueba
y_pred = knn_classifier.predict(X_test)

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)

# Imprimir la matriz de confusión
print("Matriz de Confusión:")
print(confusion)

# Calcular el reporte de clasificación
classification_rep = classification_report(y_test, y_pred)

# Imprimir el reporte de clasificación
print("Reporte de Clasificación:")
print(classification_rep)

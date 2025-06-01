import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Cargar el dataset de entrenamiento
ruta1 = r'C:\Users\victo\OneDrive\Escritorio\9_sem_DCI\Machine Learning\Exam_Prac_1\dataset3_train.xlsx'
data_train = pd.read_excel(ruta1, header=None, names=['Feature1', 'Feature2', 'Label'])
df_train=df = pd.DataFrame(data=data_train)
# Cargar el dataset de prueba
ruta2 = r'C:\Users\victo\OneDrive\Escritorio\9_sem_DCI\Machine Learning\Exam_Prac_1\dataset3_test.xlsx'
data_test = pd.read_excel(ruta2, header=None, names=['Feature1', 'Feature2', 'Label'])
df_train=df = pd.DataFrame(data=data_test)
# Filtrar solo las etiquetas de clase 1 y 2 en el dataset de entrenamiento
df_train = df_train[df_train['Label'].isin([1, 2])]

# Filtrar solo las etiquetas de clase 1 y 2 en el dataset de prueba
df_test = df_test[df_test['Label'].isin([1, 2])]

# Separar características (features) y etiquetas (labels) para entrenamiento y prueba
X_train = df_train[['Feature1', 'Feature2']]
y_train = df_train['Label']
X_test = df_test[['Feature1', 'Feature2']]
y_test = df_test['Label']

# Crear el clasificador SVC con kernel RBF y hiperparámetros C=10 y gamma=0.25
svc_classifier = SVC(kernel='rbf', C=10, gamma=0.25)

# Entrenar el clasificador SVC
svc_classifier.fit(X_train, y_train)

# Realizar predicciones en el dataset de prueba
y_pred = svc_classifier.predict(X_test)

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

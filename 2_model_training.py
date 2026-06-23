import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import pickle
import os

# Crear carpetas si no existen
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)


# CARGAR FEATURES
print("ENTRENAMIENTO CON SVM")

df = pd.read_csv('my_features.csv')
print(f"\nDataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")


# PREPARAR DATOS
X = df.drop(['filename', 'genre'], axis=1).values
y = df['genre'].values

# Codificar labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {len(X_train)} muestras")
print(f"Test set: {len(X_test)} muestras")

# ENTRENAMIENTO
print("\nEntrenando SVM con kernel RBF...")
model = SVC(kernel='rbf', random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# EVALUACION
print("\n" + "="*60)
print("EVALUACION DEL MODELO")
print("="*60)

print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_,
    digits=3
))


# MATRIZ DE CONFUSION
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar_kws={'label': 'Numero de predicciones'}
)
plt.title('Matriz de Confusion - SVM', fontsize=16)
plt.ylabel('Etiqueta Real', fontsize=12)
plt.xlabel('Etiqueta Predicha', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()


# GUARDAR MODELO
pickle.dump(model, open('models/svm_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))

print(f"\nModelo: SVM con kernel RBF")
print(f"Accuracy: {accuracy*100:.2f}%")
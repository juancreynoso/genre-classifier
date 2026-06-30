"""
Entrenamiento de Red Neuronal utilizando GTZAN + FOLKLORE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

print("="*60)
print("Entrenamiento de RED NEURONAL utilizando GTZAN + FOLKLORE")
print("="*60)

# CARGAR DATOS (GTZAN + FOLKLORE)
df = pd.read_csv('my_features_combined.csv')
print(f"\nDataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")

X = df.drop(['filename', 'genre'], axis=1).values
y = df['genre'].values

# CARGAR LABEL ENCODER Y SCALER
scaler = pickle.load(open('models/scaler_with_folklore.pkl', 'rb'))
label_encoder = pickle.load(open('models/label_encoder_with_folklore.pkl', 'rb'))

y_encoded = label_encoder.transform(y)
n_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {len(X_train)} muestras")
print(f"Test set:  {len(X_test)} muestras")
print(f"Clases ({n_classes}): {list(label_encoder.classes_)}")

# RED NEURONAL
y_train_cat = keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=n_classes)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# EVALUACION
y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=3))

# GUARDAR MODELO
model.save('models/nn_model_with_folklore.keras')
print("\nModelo guardado: models/nn_model_with_folklore.keras")


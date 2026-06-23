# 6_neural_network.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('my_features.csv')
X = df.drop(['filename', 'genre'], axis=1).values
y = df['genre'].values

# Preparar
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear red neuronal SIMPLE
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(45,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 géneros
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("="*60)
print("ENTRENAMIENTO RED NEURONAL")
print("="*60)
model.summary()

# Entrenar
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluar
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {accuracy:.2%}")

# Comparar con SVM
print("\n" + "="*60)
print("COMPARACIÓN SVM vs RED NEURONAL")
print("="*60)
print(f"SVM:           68%")
print(f"Red Neuronal:  {accuracy:.0%}")
print(f"Diferencia:    {(accuracy - 0.68)*100:+.1f}%")

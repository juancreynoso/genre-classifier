import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Keras/TensorFlow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Crear carpetas
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

DATA_PATH = "Data/genres_original/"
FOLKLORE_PATH = "Data/folklore_test/"

print("="*60)
print("CLASIFICACION CON CNN")
print("="*60)

# ============================================
# PARAMETROS
# ============================================

# Parametros de mel-spectrogram
N_MELS = 128          # Numero de bandas de frecuencia
MAX_TIME_STEPS = 130  # Frames temporales (3 segundos)
SAMPLE_RATE = 22050
DURATION = 3          # Segundos de audio a usar

# Parametros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

print(f"\nParametros:")
print(f"  Mel bands: {N_MELS}")
print(f"  Time steps: {MAX_TIME_STEPS}")
print(f"  Duration: {DURATION}s")
print(f"  Epochs: {EPOCHS}")

# ============================================
# FUNCIONES
# ============================================

def extract_melspectrogram(file_path, duration=DURATION):
    """
    Extrae mel-spectrogram de un archivo de audio
    
    Returns:
        numpy array de shape (N_MELS, MAX_TIME_STEPS)
    """
    try:
        # Cargar audio
        y, sr = librosa.load(file_path, duration=duration, sr=SAMPLE_RATE)
        
        # Crear mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=N_MELS,
            fmax=8000
        )
        
        # Convertir a dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ajustar a tamaño fijo (padding o truncado)
        if mel_spec_db.shape[1] < MAX_TIME_STEPS:
            # Pad con ceros si es muy corto
            pad_width = MAX_TIME_STEPS - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncar si es muy largo
            mel_spec_db = mel_spec_db[:, :MAX_TIME_STEPS]
        
        return mel_spec_db
    
    except Exception as e:
        print(f"Error en {file_path}: {e}")
        return None

def load_data(include_folklore=False, n_folklore_train=None):
    spectrograms = []
    labels = []
    
    # Cargar GTZAN
    print("\nCargando GTZAN...")
    genres = [g for g in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, g))]
    
    for genre in genres:
        files = sorted(glob(os.path.join(DATA_PATH, genre, "*.wav")))
        print(f"  {genre}: {len(files)} archivos")
        
        for file in files[:100]:  # Limitar a 100 por tiempo
            spec = extract_melspectrogram(file)
            if spec is not None:
                spectrograms.append(spec)
                labels.append(genre)
    
    # Cargar folklore si se requiere
    if include_folklore and n_folklore_train is not None:
        print("\nCargando folklore para entrenamiento...")
        folklore_files = sorted(glob(os.path.join(FOLKLORE_PATH, "*.wav")))
        folklore_files += sorted(glob(os.path.join(FOLKLORE_PATH, "*.mp3")))
        
        for file in folklore_files[:n_folklore_train]:
            spec = extract_melspectrogram(file)
            if spec is not None:
                spectrograms.append(spec)
                labels.append('folklore')
                print(f"  Agregado: {os.path.basename(file)}")
    
    # Convertir a arrays
    X = np.array(spectrograms)
    y = np.array(labels)
    
    # Normalizar espectrogramas (0-1)
    X = (X - X.min()) / (X.max() - X.min())
    
    # Añadir dimension de canal (para CNN)
    X = X[..., np.newaxis]  # Shape: (n_samples, N_MELS, MAX_TIME_STEPS, 1)
    
    print(f"\nDatos cargados:")
    print(f"  X shape: {X.shape}")
    print(f"  Distribucion de clases:")
    unique, counts = np.unique(y, return_counts=True)
    for genre, count in zip(unique, counts):
        print(f"    {genre}: {count}")
    
    return X, y

def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        # Capa 1: Convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Capa 2: Convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Capa 3: Convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ============================================
# EXPERIMENTO: CNN SIN FOLKLORE
# ============================================

print("\n" + "="*60)
print("EXPERIMENTO: CNN CON GTZAN (sin folklore)")
print("="*60)

# Cargar datos
X, y = load_data(include_folklore=False)

# Codificar labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"\nClases: {label_encoder.classes_}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# Crear modelo
model = create_cnn_model(
    input_shape=(N_MELS, MAX_TIME_STEPS, 1),
    num_classes=len(label_encoder.classes_)
)

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nARQUITECTURA DEL MODELO")
model.summary()

# Entrenar
print("\nENTRENAMIENTO CON CNN")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

# Evaluar
print("\n" + "="*60)
print("EVALUACION")
print("="*60)

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test_labels,
    y_pred,
    target_names=label_encoder.classes_,
    digits=3
))

# ============================================
# VISUALIZACIONES
# ============================================

# Curvas de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy durante entrenamiento', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss durante entrenamiento', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/cnn_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nGrafico guardado: results/cnn_training_history.png")

# Matriz de confusion
cm = confusion_matrix(y_test_labels, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Matriz de Confusion - CNN', fontsize=16)
plt.ylabel('Etiqueta Real', fontsize=12)
plt.xlabel('Etiqueta Predicha', fontsize=12)
plt.tight_layout()
plt.savefig('results/cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Grafico guardado: results/cnn_confusion_matrix.png")

# ============================================
# GUARDAR MODELO
# ============================================

model.save('models/cnn_model.keras')
pickle.dump(label_encoder, open('models/cnn_label_encoder.pkl', 'wb'))

print("\nModelo guardado:")
print("  - models/cnn_model.keras")
print("  - models/cnn_label_encoder.pkl")

# ============================================
# COMPARACION CON SVM
# ============================================

print("\n" + "="*60)
print("COMPARACION: CNN vs SVM")
print("="*60)

svm_accuracy = 0.68  # Tu accuracy de SVM

print(f"\nSVM Accuracy:  {svm_accuracy:.2%}")
print(f"CNN Accuracy:  {accuracy:.2%}")
print(f"Diferencia:    {(accuracy - svm_accuracy)*100:+.2f}%")


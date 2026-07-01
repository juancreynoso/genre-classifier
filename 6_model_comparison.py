import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import os

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from features import load_folklore_features

# Reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('results', exist_ok=True)

print("="*60)
print("COMPARACION SVM vs RED NEURONAL (con folklore, 11 clases)")
print("="*60)


# ============================================
# CARGAR DATASET AMPLIADO: GTZAN + FOLKLORE
# ============================================
df_gtzan = pd.read_csv('my_features.csv')
df_folklore = load_folklore_features()

df = pd.concat([df_gtzan, df_folklore], ignore_index=True)
print(f"\nGTZAN:    {len(df_gtzan)} muestras")
print(f"Folklore: {len(df_folklore)} muestras")
print(f"Total:    {len(df)} muestras, {df['genre'].nunique()} generos")
print("\nDistribucion:")
print(df['genre'].value_counts().sort_index())


# ============================================
# PREPARAR DATOS (un solo split 80/20, igual para ambos modelos)
# ============================================
X = df.drop(['filename', 'genre'], axis=1).values
y = df['genre'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = list(label_encoder.classes_)
n_classes = len(classes)

# MISMO split para SVM y red neuronal -> comparacion justa
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")


# ============================================
# MODELO 1: SVM
# ============================================
print("\n" + "="*60)
print("MODELO 1: SVM (kernel RBF)")
print("="*60)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"\nAccuracy global: {acc_svm:.2%}")
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm, target_names=classes, digits=3))


# ============================================
# MODELO 2: RED NEURONAL (MLP)
# ============================================
print("\n" + "="*60)
print("MODELO 2: RED NEURONAL (MLP)")
print("="*60)

y_train_cat = keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=n_classes)

nn = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')  # 11 generos
])

nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0  # silencioso; cambiar a 1 para ver el entrenamiento
)

y_pred_nn = np.argmax(nn.predict(X_test_scaled, verbose=0), axis=1)

acc_nn = accuracy_score(y_test, y_pred_nn)
print(f"\nAccuracy global: {acc_nn:.2%}")
print("\nClassification Report (Red Neuronal):")
print(classification_report(y_test, y_pred_nn, target_names=classes, digits=3))


# ============================================
# COMPARACION: foco en FOLKLORE
# ============================================
print("\n" + "="*60)
print("COMPARACION")
print("="*60)

rep_svm = classification_report(y_test, y_pred_svm, target_names=classes,
                                output_dict=True, zero_division=0)
rep_nn = classification_report(y_test, y_pred_nn, target_names=classes,
                               output_dict=True, zero_division=0)

print(f"\n{'Metrica':<28} {'SVM':>10} {'Red Neuronal':>14}")
print("-"*54)
print(f"{'Accuracy global':<28} {acc_svm:>9.1%} {acc_nn:>13.1%}")
print(f"{'Folklore - precision':<28} {rep_svm['folklore']['precision']:>9.1%} "
      f"{rep_nn['folklore']['precision']:>13.1%}")
print(f"{'Folklore - recall':<28} {rep_svm['folklore']['recall']:>9.1%} "
      f"{rep_nn['folklore']['recall']:>13.1%}")
print(f"{'Folklore - f1-score':<28} {rep_svm['folklore']['f1-score']:>9.1%} "
      f"{rep_nn['folklore']['f1-score']:>13.1%}")

# Cuantas canciones de folklore hay en el test y como las clasifico cada modelo
folklore_idx = label_encoder.transform(['folklore'])[0]
mask = (y_test == folklore_idx)
print(f"\nCanciones de folklore en el test: {mask.sum()}")
print(f"  SVM las reconoce: {(y_pred_svm[mask] == folklore_idx).sum()}/{mask.sum()}")
print(f"  Red las reconoce: {(y_pred_nn[mask] == folklore_idx).sum()}/{mask.sum()}")


# ============================================
# VISUALIZACION: matrices de confusion lado a lado
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, y_pred, title, cmap, acc in [
    (axes[0], y_pred_svm, 'SVM', 'Blues', acc_svm),
    (axes[1], y_pred_nn, 'Red Neuronal', 'Greens', acc_nn),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set_title(f'{title} (accuracy {acc:.1%})', fontsize=15, fontweight='bold')
    ax.set_xlabel('Predicho', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nGrafico guardado: results/model_comparison.png")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"  SVM           - accuracy: {acc_svm:.1%}  |  folklore f1: {rep_svm['folklore']['f1-score']:.1%}")
print(f"  Red Neuronal  - accuracy: {acc_nn:.1%}  |  folklore f1: {rep_nn['folklore']['f1-score']:.1%}")

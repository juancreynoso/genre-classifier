import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
import librosa
from glob import glob
import os

FOLKLORE_PATH = "Data/folklore_test/"
FOLKLORE_CACHE = "folklore_features.csv"
N_SPLITS = 5  # k-fold

print("="*60)
print("EXPERIMENTO B (v2): FOLKLORE CON CROSS-VALIDATION")
print("="*60)

df_gtzan = pd.read_csv('my_features.csv')
print(f"\nGTZAN cargado: {len(df_gtzan)} muestras")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30, sr=22050)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()
        else:
            tempo = float(tempo)

        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        features = {
            **{f'mfcc_mean_{i}': mfcc_mean[i] for i in range(13)},
            **{f'mfcc_std_{i}': mfcc_std[i] for i in range(13)},
            **{f'chroma_{i}': chroma_mean[i] for i in range(12)},
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_std': spectral_centroid_std,
            'spectral_rolloff_mean': spectral_rolloff_mean,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'zcr_mean': zcr_mean,
            'tempo': tempo,
            'rms_mean': rms_mean
        }
        return features
    except Exception as e:
        print(f"Error en {file_path}: {e}")
        return None


folklore_files = sorted(glob(os.path.join(FOLKLORE_PATH, "*.wav")))
folklore_files += sorted(glob(os.path.join(FOLKLORE_PATH, "*.mp3")))

if len(folklore_files) == 0:
    print(f"\nError: No se encontraron archivos en {FOLKLORE_PATH}")
    exit()

use_cache = os.path.exists(FOLKLORE_CACHE)
if use_cache:
    df_folklore = pd.read_csv(FOLKLORE_CACHE)
    # Si cambio la cantidad de canciones, re-extraer
    if len(df_folklore) != len(folklore_files):
        print("\nLa cantidad de canciones cambio -> re-extrayendo features...")
        use_cache = False

if use_cache:
    print(f"\nFeatures de folklore cargadas desde cache: {len(df_folklore)} muestras")
else:
    print(f"\nProcesando {len(folklore_files)} canciones de folklore...")
    folklore_data = []
    for i, file in enumerate(folklore_files, 1):
        print(f"  [{i}/{len(folklore_files)}] {os.path.basename(file)}")
        features = extract_features(file)
        if features is not None:
            features['filename'] = os.path.basename(file)
            features['genre'] = 'folklore'
            folklore_data.append(features)
    df_folklore = pd.DataFrame(folklore_data)
    df_folklore.to_csv(FOLKLORE_CACHE, index=False)
    print(f"Features guardadas en: {FOLKLORE_CACHE}")

print(f"Folklore listo: {len(df_folklore)} muestras")

if len(df_folklore) < N_SPLITS:
    print(f"\nError: Se necesitan al menos {N_SPLITS} canciones de folklore")
    exit()


# ============================================
# CROSS-VALIDATION (K-FOLD SOBRE EL FOLKLORE)
# ============================================
# Idea: en cada fold apartamos una porcion del folklore como TEST,
# entrenamos con GTZAN + el resto del folklore, y vemos si el modelo
# reconoce las canciones apartadas como 'folklore'.
# Asi cada cancion es evaluada exactamente una vez (out-of-fold) y
# obtenemos una metrica estable, no dependiente de una sola corrida.

feature_cols = [c for c in df_gtzan.columns if c not in ['filename', 'genre']]

X_folklore = df_folklore[feature_cols].values
folklore_names = df_folklore['filename'].values

X_gtzan = df_gtzan[feature_cols].values
y_gtzan = df_gtzan['genre'].values

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print("\n" + "="*60)
print(f"VALIDACION CRUZADA ({N_SPLITS}-FOLD)")
print("="*60)

fold_accuracies = []
# Predicciones out-of-fold: una por cada cancion de folklore
oof_predictions = np.empty(len(df_folklore), dtype=object)

# StratifiedKFold necesita 'y'; como el folklore es una sola clase,
# usamos un dummy del mismo largo.
dummy_y = np.zeros(len(df_folklore))

for fold, (train_idx, test_idx) in enumerate(kf.split(X_folklore, dummy_y), 1):
    # Folklore: parte para entrenar, parte para testear
    X_folk_train = X_folklore[train_idx]
    X_folk_test = X_folklore[test_idx]

    # Dataset de entrenamiento = GTZAN + folklore de entrenamiento
    X_train = np.vstack([X_gtzan, X_folk_train])
    y_train = np.concatenate([y_gtzan, ['folklore'] * len(train_idx)])

    # Codificar y normalizar (fit SOLO con train -> sin data leakage)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_folk_test_scaled = scaler.transform(X_folk_test)

    # Entrenar SVM
    model = SVC(kernel='rbf', random_state=42, probability=False)
    model.fit(X_train_scaled, y_train_enc)

    # Predecir el folklore apartado
    pred_enc = model.predict(X_folk_test_scaled)
    pred_genre = label_encoder.inverse_transform(pred_enc)

    # Guardar predicciones out-of-fold
    for j, idx in enumerate(test_idx):
        oof_predictions[idx] = pred_genre[j]

    # Accuracy del fold = fraccion de folklore reconocido como 'folklore'
    fold_acc = np.mean(pred_genre == 'folklore')
    fold_accuracies.append(fold_acc)
    print(f"  Fold {fold}: {len(test_idx):2d} canciones de test -> "
          f"reconocidas como folklore: {fold_acc:.1%}")

fold_accuracies = np.array(fold_accuracies)


# ============================================
# RESULTADOS AGREGADOS
# ============================================
print("\n" + "="*60)
print("RESULTADOS")
print("="*60)

folklore_recall = np.mean(oof_predictions == 'folklore')
print(f"\nFolklore reconocido correctamente (recall): {folklore_recall:.1%}")
print(f"  ({np.sum(oof_predictions == 'folklore')}/{len(df_folklore)} canciones)")
print(f"\nAccuracy promedio por fold: {fold_accuracies.mean():.1%} "
      f"(+/- {fold_accuracies.std():.1%})")

# A donde van a parar las canciones que NO se reconocen como folklore
print("\nDistribucion de predicciones (donde 'cae' cada cancion):")
pred_counts = pd.Series(oof_predictions).value_counts()
for genre, count in pred_counts.items():
    pct = count / len(df_folklore) * 100
    marca = "  <- correcto" if genre == 'folklore' else ""
    print(f"  {genre:12s}: {count:2d} ({pct:5.1f}%){marca}")

# Canciones mal clasificadas (utiles para el analisis)
mal = [(folklore_names[i], oof_predictions[i])
       for i in range(len(df_folklore)) if oof_predictions[i] != 'folklore']
if mal:
    print(f"\nCanciones NO reconocidas como folklore ({len(mal)}):")
    for name, pred in mal:
        print(f"  {name:35s} -> {pred}")


# ============================================
# VISUALIZACION
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy por fold
ax1.bar(range(1, N_SPLITS + 1), fold_accuracies * 100,
        color='steelblue', edgecolor='black', alpha=0.85)
ax1.axhline(folklore_recall * 100, color='red', linestyle='--',
            label=f'Promedio: {folklore_recall:.1%}')
ax1.set_title(f'Recall de folklore por fold ({N_SPLITS}-fold CV)',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('% reconocido como folklore', fontsize=12)
ax1.set_ylim([0, 105])
ax1.set_xticks(range(1, N_SPLITS + 1))
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(fold_accuracies * 100):
    ax1.text(i + 1, v + 1, f'{v:.0f}%', ha='center', fontweight='bold')

# Distribucion de predicciones out-of-fold
colors = ['green' if g == 'folklore' else 'coral' for g in pred_counts.index]
ax2.bar(pred_counts.index, pred_counts.values, color=colors,
        edgecolor='black', alpha=0.85)
ax2.set_title('A que genero se asignan las canciones de folklore',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Genero predicho', fontsize=12)
ax2.set_ylabel('Numero de canciones', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
for i, v in enumerate(pred_counts.values):
    ax2.text(i, v + 0.3, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/folklore_crossvalidation.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nGrafico guardado: results/folklore_crossvalidation.png")


# ============================================
# MODELO FINAL (entrenado con TODO el folklore)
# ============================================
# Para uso real, entrenamos el modelo definitivo con GTZAN + las 70
# canciones de folklore. La evaluacion confiable es la de arriba (CV).
print("\n" + "="*60)
print("ENTRENANDO MODELO FINAL (GTZAN + todo el folklore)")
print("="*60)

X_all = np.vstack([X_gtzan, X_folklore])
y_all = np.concatenate([y_gtzan, ['folklore'] * len(df_folklore)])

label_encoder_final = LabelEncoder()
y_all_enc = label_encoder_final.fit_transform(y_all)

# Split para reportar accuracy global del modelo de 11 clases
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all_enc, test_size=0.2, random_state=42, stratify=y_all_enc
)

scaler_final = StandardScaler()
X_tr_scaled = scaler_final.fit_transform(X_tr)
X_te_scaled = scaler_final.transform(X_te)

model_final = SVC(kernel='rbf', random_state=42, probability=True)
model_final.fit(X_tr_scaled, y_tr)

acc_global = accuracy_score(y_te, model_final.predict(X_te_scaled))
print(f"\nClases ({len(label_encoder_final.classes_)}): "
      f"{list(label_encoder_final.classes_)}")
print(f"Accuracy global del modelo de 11 clases: {acc_global:.2%}")

# Reentrenar con TODO (sin apartar test) para el modelo que se guarda
scaler_save = StandardScaler()
X_all_scaled = scaler_save.fit_transform(X_all)
model_save = SVC(kernel='rbf', random_state=42, probability=True)
model_save.fit(X_all_scaled, y_all_enc)

pickle.dump(model_save, open('models/svm_with_folklore.pkl', 'wb'))
pickle.dump(scaler_save, open('models/scaler_with_folklore.pkl', 'wb'))
pickle.dump(label_encoder_final, open('models/label_encoder_with_folklore.pkl', 'wb'))

print("\nModelo final guardado:")
print("  - models/svm_with_folklore.pkl")
print("  - models/scaler_with_folklore.pkl")
print("  - models/label_encoder_with_folklore.pkl")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"  Canciones de folklore:        {len(df_folklore)}")
print(f"  Recall de folklore ({N_SPLITS}-fold): {folklore_recall:.1%} "
      f"(+/- {fold_accuracies.std():.1%})")
print(f"  Accuracy global (11 clases):  {acc_global:.2%}")

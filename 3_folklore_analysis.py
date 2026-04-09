import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import pickle

FOLKLORE_PATH = "Data/folklore_test/"

print("="*60)
print("EXPERIMENTO A: FOLKLORE SIN ENTRENAMIENTO")
print("="*60)

# CARGAR MODELO
print("\nCargando modelo entrenado...")

model = pickle.load(open('models/svm_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

print("Modelo cargado: SVM")
print(f"Generos conocidos: {list(label_encoder.classes_)}")


# FUNCION DE EXTRACCION
def extract_features(file_path):
    """Extrae features de audio"""
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
        
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            [spectral_centroid_mean, spectral_centroid_std, 
             spectral_rolloff_mean, spectral_bandwidth_mean, 
             zcr_mean, tempo, rms_mean]
        ])
        
        return features
    
    except Exception as e:
        print(f"Error en {file_path}: {e}")
        return None


# PROCESAR FOLKLORE
folklore_files = glob(os.path.join(FOLKLORE_PATH, "*.wav"))
folklore_files += glob(os.path.join(FOLKLORE_PATH, "*.mp3"))
folklore_files = sorted(folklore_files)

if len(folklore_files) == 0:
    print(f"\nError: No se encontraron archivos en {FOLKLORE_PATH}")
    exit()

print(f"\nEncontradas {len(folklore_files)} canciones de folklore")
print("\nProcesando archivos...")

folklore_features = []
folklore_names = []

for i, file in enumerate(folklore_files, 1):
    name = os.path.basename(file)
    print(f"  [{i}/{len(folklore_files)}] {name}")
    
    features = extract_features(file)
    if features is not None:
        folklore_features.append(features)
        folklore_names.append(name)

if len(folklore_features) == 0:
    print("\nError: No se pudo procesar ningun archivo")
    exit()

X_folklore = np.array(folklore_features)
X_folklore_scaled = scaler.transform(X_folklore)

# Predecir
predictions = model.predict(X_folklore_scaled)
probabilities = model.predict_proba(X_folklore_scaled)


# RESULTADOS
print("\n" + "="*60)
print("PREDICCIONES DEL MODELO")
print("="*60)

results_df = pd.DataFrame({
    'Cancion': folklore_names,
    'Genero Predicho': label_encoder.inverse_transform(predictions),
    'Confianza': np.max(probabilities, axis=1)
})

print("\n", results_df.to_string(index=False))


# ESTADISTICAS
print("\n" + "="*60)
print("DISTRIBUCION DE PREDICCIONES")
print("="*60)

pred_counts = pd.Series(
    label_encoder.inverse_transform(predictions)
).value_counts()

print("\nEl modelo clasifico folklore argentino como:")
for genre, count in pred_counts.items():
    percentage = (count / len(predictions)) * 100
    bar = "█" * int(percentage / 5)
    print(f"  {genre:12s}: {count:2d} canciones ({percentage:5.1f}%) {bar}")


# CONFIANZA
print("\n" + "="*60)
print("ANALISIS DE CONFIANZA")
print("="*60)

max_probs = np.max(probabilities, axis=1)
avg_confidence = np.mean(max_probs)

print(f"\nConfianza promedio: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")

print(f"\nPrediccion mas confiada:")
max_idx = np.argmax(max_probs)
print(f"  {folklore_names[max_idx]}")
print(f"  {label_encoder.inverse_transform([predictions[max_idx]])[0]} ({max_probs[max_idx]:.1%})")

print(f"\nPrediccion mas ambigua:")
min_idx = np.argmin(max_probs)
print(f"  {folklore_names[min_idx]}")
print(f"  {label_encoder.inverse_transform([predictions[min_idx]])[0]} ({max_probs[min_idx]:.1%})")


# VISUALIZACION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Distribucion
pred_counts.plot(kind='bar', ax=ax1, color='coral', edgecolor='black')
ax1.set_title('Predicciones para Folklore Argentino', fontsize=14, fontweight='bold')
ax1.set_xlabel('Genero Predicho', fontsize=12)
ax1.set_ylabel('Numero de Canciones', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(pred_counts.values):
    ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# Heatmap
short_names = [n[:20] + '...' if len(n) > 20 else n for n in folklore_names]
sns.heatmap(
    probabilities.T, 
    xticklabels=short_names,
    yticklabels=label_encoder.classes_,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    ax=ax2,
    cbar_kws={'label': 'Probabilidad'}
)
ax2.set_title('Probabilidades por Cancion', fontsize=14, fontweight='bold')
ax2.set_xlabel('Canciones de Folklore', fontsize=12)
ax2.set_ylabel('Generos GTZAN', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/folklore_without_training.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nGrafico guardado: results/folklore_without_training.png")


# CONCLUSIONES
print("\n" + "="*60)
print("CONCLUSIONES - EXPERIMENTO A")
print("="*60)

most_common = pred_counts.index[0]
most_common_pct = (pred_counts.iloc[0] / len(predictions)) * 100

print(f"\nResultados:")
print(f"  - Genero mas predicho: {most_common} ({most_common_pct:.1f}%)")
print(f"  - Confianza promedio: {avg_confidence:.1%}")
print(f"  - Total de canciones: {len(folklore_files)}")

print(f"\nInterpretacion:")
if most_common in ['country', 'blues']:
    print(f"  El modelo clasifico folklore como {most_common}")
    print("  Esto tiene sentido por:")
    print("    - Instrumentos acusticos similares")
    print("    - Estructuras ritmicas comparables")
    print("    - Espectro frecuencial parecido")
else:
    print(f"  El modelo clasifico folklore como {most_common}")
    print("  Resultado interesante para analizar")

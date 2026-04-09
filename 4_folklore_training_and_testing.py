import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
import librosa
from glob import glob
import os

FOLKLORE_PATH = "Data/folklore_test/"

print("="*60)
print("EXPERIMENTO B: FOLKLORE CON ENTRENAMIENTO")
print("="*60)


# LOAD GTZAN
df_gtzan = pd.read_csv('my_features.csv')
print(f"\nGTZAN cargado: {len(df_gtzan)} muestras")


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
        print(f"Error: {e}")
        return None



# PROCESAR FOLKLORE
folklore_files = sorted(glob(os.path.join(FOLKLORE_PATH, "*.wav")))
folklore_files += sorted(glob(os.path.join(FOLKLORE_PATH, "*.mp3")))

if len(folklore_files) == 0:
    print(f"\nError: No se encontraron archivos en {FOLKLORE_PATH}")
    exit()

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
print(f"Folklore procesado: {len(df_folklore)} muestras")


# SPLIT DE FOLKLORE
if len(df_folklore) < 2:
    print("\nError: Se necesitan al menos 2 canciones de folklore")
    exit()

# SHUFFLE folklore
df_folklore_shuffled = df_folklore.sample(frac=1, random_state=None).reset_index(drop=True)

# Elegir 1 cancion random para test
df_folklore_test = df_folklore_shuffled.iloc[:1].copy()
df_folklore_train = df_folklore_shuffled.iloc[1:].copy()

print(f"\nSplit de folklore (RANDOM):")
print(f"  Train: {len(df_folklore_train)} canciones")
print(f"  Test:  1 cancion")
print(f"  Cancion de test: {df_folklore_test['filename'].values[0]}")

'''
n_train = min(37, len(df_folklore) - 1)
df_folklore_train = df_folklore.iloc[:n_train].copy()
df_folklore_test = df_folklore.iloc[n_train:n_train+1].copy()

print(f"\nSplit de folklore:")
print(f"  Train: {len(df_folklore_train)} canciones")
print(f"  Test:  {len(df_folklore_test)} canciones")
print(f"  Cancion de test: {df_folklore_test['filename'].values[0]}")
'''

# COMBINAR GTZAN + FOLKLORE

df_combined = pd.concat([df_gtzan, df_folklore_train], ignore_index=True)
print(f"\nDataset combinado: {len(df_combined)} muestras")
print(f"Generos: {df_combined['genre'].nunique()}")
print("\nDistribucion:")
print(df_combined['genre'].value_counts().sort_index())


# ENTRENAR CON 11 GENEROS
print("\n" + "="*60)
print("ENTRENAMIENTO CON FOLKLORE INCLUIDO")
print("="*60)

X = df_combined.drop(['filename', 'genre'], axis=1).values
y = df_combined['genre'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nClases: {list(label_encoder.classes_)}")

X_train, X_test_gtzan, y_train, y_test_gtzan = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_gtzan_scaled = scaler.transform(X_test_gtzan)

print(f"\nTrain: {len(X_train)} muestras")
print(f"Test (GTZAN): {len(X_test_gtzan)} muestras")

print("\nEntrenando SVM con 10 generos y Folklore...")
model = SVC(kernel='rbf', random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_gtzan_scaled)
accuracy = accuracy_score(y_test_gtzan, y_pred)

print(f"Accuracy en el entrenamiento: {accuracy:.2%}")


# PRUEBA CON CANCION
print("\n" + "="*60)
print("TEST CON CANCION DE FOLKLORE NO ENTRENADA")
print("="*60)

X_folklore_test = df_folklore_test.drop(['filename', 'genre'], axis=1).values
X_folklore_test_scaled = scaler.transform(X_folklore_test)

probabilities = model.predict_proba(X_folklore_test_scaled)[0]
prediction = np.argmax(probabilities)  # Tomar el índice con mayor probabilidad

predicted_genre = label_encoder.inverse_transform([prediction])[0]
confidence = probabilities[prediction]

print(f"\nCancion: {df_folklore_test['filename'].values[0]}")
print(f"Prediccion: {predicted_genre}")
print(f"Confianza: {confidence:.2%}")

print("\nProbabilidades por genero:")
for i, genre in enumerate(label_encoder.classes_):
    bar = "█" * int(probabilities[i] * 50)
    print(f"  {genre:12s}: {probabilities[i]:5.1%} {bar}")


# RESULTADOS
print("\nEXPERIMENTO B: Con folklore en entrenamiento (13 muestras)")
print(f"  Resultado: Clasificado como '{predicted_genre}'")
print(f"  Confianza: {confidence:.1%}")

# GRAFICOS
# Cargar resultados del experimento A para comparar
# (valores aproximados del experimento anterior)
exp_a_country = 0.71  # Ajusta con tus valores reales
exp_a_folklore = 0.0

exp_b_country = probabilities[list(label_encoder.classes_).index('country')]
exp_b_folklore = probabilities[list(label_encoder.classes_).index('folklore')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Comparacion de experimentos
categories = ['Sin folklore\nen training', 'Con folklore\nen training']
x = np.arange(len(categories))
width = 0.35

country_vals = [exp_a_country, exp_b_country]
folklore_vals = [exp_a_folklore, exp_b_folklore]

ax1.bar(x - width/2, country_vals, width, label='Country', color='orange', alpha=0.8)
ax1.bar(x + width/2, folklore_vals, width, label='Folklore', color='green', alpha=0.8)

ax1.set_ylabel('Probabilidad de prediccion', fontsize=12)
ax1.set_title('Comparacion: Prediccion de la misma cancion', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1])

# Distribucion con folklore
genres = label_encoder.classes_
probs = probabilities
colors = ['green' if g == 'folklore' else 'steelblue' for g in genres]

ax2.barh(genres, probs, color=colors, alpha=0.8)
ax2.set_xlabel('Probabilidad', fontsize=12)
ax2.set_title('Prediccion con folklore en training', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('results/folklore_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nGrafico guardado: results/folklore_comparison.png")


# GUARDAR MODELO
pickle.dump(model, open('models/svm_with_folklore.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler_with_folklore.pkl', 'wb'))
pickle.dump(label_encoder, open('models/label_encoder_with_folklore.pkl', 'wb'))

print("\nModelos guardados:")
print("  - models/svm_with_folklore.pkl")
print("  - models/scaler_with_folklore.pkl")
print("  - models/label_encoder_with_folklore.pkl")


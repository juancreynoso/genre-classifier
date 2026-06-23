import librosa
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm


# EXTRACCION DE FEATURES CON LIBROSA
DATA_PATH = "Data/genres_original/"
OUTPUT_FILE = "my_features.csv"

print("="*60)
print("EXTRACCION DE FEATURES CON LIBROSA")
print("="*60)

# Verificar generos
genres = [g for g in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, g))]
print(f"\nGeneros encontrados: {genres}")
print(f"Total: {len(genres)} generos\n")

def extract_features(file_path):
    try:
        # Cargar audio (primeros 30 segundos)
        y, sr = librosa.load(file_path, duration=30, sr=22050)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()
        else:
            tempo = float(tempo)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Concatenar todas las features
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


# PROCESAR TODO EL DATASET
print("Extrayendo features...")

all_features = []

for genre in genres:
    print(f"Procesando genero: {genre}")
    files = sorted(glob(os.path.join(DATA_PATH, genre, "*.wav")))
    
    for file in tqdm(files, desc=f"  {genre}", ncols=80):
        features = extract_features(file)
        if features is not None:
            features['filename'] = os.path.basename(file)
            features['genre'] = genre
            all_features.append(features)

# Convertir a DataFrame
df = pd.DataFrame(all_features)

# Reorganizar columnas (genre y filename al final)
cols = [c for c in df.columns if c not in ['filename', 'genre']]
cols = cols + ['filename', 'genre']
df = df[cols]

print("EXTRACCION COMPLETADA")
print(f"Total de muestras: {len(df)}")
print(f"Total de features: {len(df.columns) - 2}")

# GUARDAR RESULTADOS
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nArchivo guardado: {OUTPUT_FILE}")

# ANALISIS EXPLORATORIO

print("\nDistribucion de generos:")
print(df['genre'].value_counts().sort_index())

print("\nEstadisticas de Tempo por genero:")
tempo_stats = df.groupby('genre')['tempo'].agg(['mean', 'std', 'min', 'max'])
print(tempo_stats.round(2))
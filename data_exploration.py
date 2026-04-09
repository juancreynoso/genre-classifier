import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# EXPLORACIÓN DEL DATASET
DATA_PATH = "Data/genres_original/"

# Verificar géneros disponibles
genres = os.listdir(DATA_PATH)
genres = [g for g in genres if os.path.isdir(os.path.join(DATA_PATH, g))]
print(f"Géneros encontrados: {genres}")
print(f"Total de géneros: {len(genres)}\n")

# Contar archivos por género
for genre in genres:
    files = glob(os.path.join(DATA_PATH, genre, "*.wav"))
    print(f"{genre}: {len(files)} archivos")


# VISUALIZACIÓN DE EJEMPLOS
def plot_audio_examples(genre, n_examples=3):
    """Visualiza espectrogramas de ejemplos de un género"""
    files = glob(os.path.join(DATA_PATH, genre, "*.wav"))[:n_examples]
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(15, 4*n_examples))
    
    for i, file in enumerate(files):
        # Cargar audio
        y, sr = librosa.load(file, duration=30)
        
        # Waveform
        axes[i, 0].plot(np.linspace(0, 30, len(y)), y)
        axes[i, 0].set_title(f'{genre} - Waveform - {os.path.basename(file)}')
        axes[i, 0].set_xlabel('Tiempo (s)')
        axes[i, 0].set_ylabel('Amplitud')
        
        # Espectrograma
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[i, 1])
        axes[i, 1].set_title(f'{genre} - Espectrograma')
        fig.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()

# Visualizar ejemplos de 2 géneros diferentes
print("\n=== Visualizando ejemplos de Blues ===")
plot_audio_examples('blues', n_examples=2)

print("\n=== Visualizando ejemplos de Rock ===")
plot_audio_examples('rock', n_examples=2)

# EXTRACCIÓN DE FEATURES (Un ejemplo)
def extract_features(file_path):
    """Extrae features de audio usando librosa"""
    try:
        # Cargar audio (30 segundos)
        y, sr = librosa.load(file_path, duration=30)
        
        # MFCCs (Mel-frequency cepstral coefficients) - Los más importantes
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma (relacionado con las notas musicales)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Tempo
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        
        # Concatenar todas las features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            [spectral_centroid, spectral_rolloff, zcr, tempo]
        ])
        
        return features
    
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return None

# Prueba con un archivo
test_file = glob(os.path.join(DATA_PATH, genres[0], "*.wav"))[0]
test_features = extract_features(test_file)
print(f"\n=== Features extraídas de un archivo ===")
print(f"Archivo: {test_file}")
print(f"Dimensión del vector de features: {len(test_features)}")
print(f"Primeros 10 valores: {test_features[:10]}")

"""Funciones compartidas de extracción de features de audio.

Centraliza la extracción de las 45 features (antes copiada en los scripts 1, 3,
4 y 4b) y el manejo de cache del folklore, para no duplicar código ni re-extraer
el audio en cada corrida.

Las 45 features por canción son:
  - 13 MFCC (media)        - 13 MFCC (desviación estándar)
  - 12 Chroma (media)      - Spectral centroid (media y std)
  - Spectral rolloff       - Spectral bandwidth
  - Zero crossing rate     - Tempo (BPM)        - RMS energy
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import librosa

N_MFCC = 13
DURATION = 30
SAMPLE_RATE = 22050


def extract_features(file_path):
    """Extrae las 45 features de un archivo de audio.

    Devuelve un dict {nombre_feature: valor} o None si el archivo falla.
    El orden de las claves es el mismo para GTZAN y folklore, así los CSV
    resultantes son compatibles entre sí.
    """
    try:
        y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo.item() if isinstance(tempo, np.ndarray) else float(tempo)

        rms = librosa.feature.rms(y=y)

        return {
            **{f'mfcc_mean_{i}': mfcc_mean[i] for i in range(N_MFCC)},
            **{f'mfcc_std_{i}': mfcc_std[i] for i in range(N_MFCC)},
            **{f'chroma_{i}': chroma_mean[i] for i in range(12)},
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'zcr_mean': np.mean(zcr),
            'tempo': tempo,
            'rms_mean': np.mean(rms),
        }
    except Exception as e:
        print(f"Error en {file_path}: {e}")
        return None


def extract_folder(folder, genre=None, verbose=True):
    """Extrae features de todos los .wav/.mp3 de una carpeta.

    Devuelve un DataFrame con las columnas de features + 'filename'
    (y 'genre' si se pasa el argumento).
    """
    files = sorted(glob(os.path.join(folder, "*.wav")))
    files += sorted(glob(os.path.join(folder, "*.mp3")))

    rows = []
    for i, file in enumerate(files, 1):
        if verbose:
            print(f"  [{i}/{len(files)}] {os.path.basename(file)}")
        feats = extract_features(file)
        if feats is not None:
            feats['filename'] = os.path.basename(file)
            if genre is not None:
                feats['genre'] = genre
            rows.append(feats)
    return pd.DataFrame(rows)


def load_folklore_features(folklore_path="Data/folklore_test/",
                           cache_path="folklore_features.csv"):
    """Devuelve las features del folklore como DataFrame, con cache.

    Usa el cache (CSV) si existe y la cantidad de canciones coincide; si no,
    extrae desde el audio y guarda el cache. Así el folklore se procesa una
    sola vez y los demás scripts no re-extraen en cada corrida.
    """
    files = sorted(glob(os.path.join(folklore_path, "*.wav")))
    files += sorted(glob(os.path.join(folklore_path, "*.mp3")))
    n_files = len(files)

    if n_files == 0:
        raise FileNotFoundError(f"No se encontraron audios en {folklore_path}")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) == n_files:
            print(f"Folklore desde cache: {len(df)} canciones ({cache_path})")
            return df
        print("Cambió la cantidad de canciones de folklore -> re-extrayendo...")

    print(f"Extrayendo features de {n_files} canciones de folklore...")
    df = extract_folder(folklore_path, genre='folklore')
    df.to_csv(cache_path, index=False)
    print(f"Cache guardado: {cache_path}")
    return df


def feature_columns(df):
    """Columnas de features de un DataFrame (excluye 'filename' y 'genre')."""
    return [c for c in df.columns if c not in ('filename', 'genre')]

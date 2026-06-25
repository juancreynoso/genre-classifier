import pandas as pd
import os
from glob import glob
from tqdm import tqdm

from features import extract_features


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
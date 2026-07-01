import pandas as pd
import os
from glob import glob
from tqdm import tqdm

from features import extract_features, load_folklore_features

def gtzan_processing():
    """Analiza el dataset GTZAN utilizando librosa"""

    DATA_PATH_GTZAN = "Data/genres_original/"
    OUTPUT_FILE = "my_features.csv"

    print("="*60)
    print("EXTRACCION DE FEATURES CON LIBROSA - GTZAN")
    print("="*60)

    # Verificar generos
    genres = [g for g in os.listdir(DATA_PATH_GTZAN) if os.path.isdir(os.path.join(DATA_PATH_GTZAN, g))]
    print(f"\nGeneros encontrados: {genres}")
    print(f"Total: {len(genres)} generos\n")

    # PROCESAR TODO EL DATASET
    print("Extrayendo features...")

    all_features = []

    for genre in genres:
        print(f"Procesando genero: {genre}")
        files = sorted(glob(os.path.join(DATA_PATH_GTZAN, genre, "*.wav")))
        
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

def combination_gtzan_folklore():
    """Combina el dataset original GTZAN con las features obtenidas del DATASET de folklore"""
    print("="*60)
    print("COMBINACION DE FEATURES CON LIBROSA - GTZAN + FOLKLORE")
    print("="*60)

    OUTPUT_FILE = "my_features_combined.csv"

    # LOAD GTZAN
    df_gtzan = pd.read_csv('my_features.csv')
    print(f"\nGTZAN cargado: {len(df_gtzan)} muestras")

    # PROCESAR FOLKLORE (features desde cache; se extraen una sola vez)
    df_folklore = load_folklore_features()
    print(f"Folklore procesado: {len(df_folklore)} muestras")

    # COMBINAR GTZAN + FOLKLORE
    df_combined = pd.concat([df_gtzan, df_folklore], ignore_index=True)
    print(f"\nDataset combinado: {len(df_combined)} muestras")
    print(f"Generos: {df_combined['genre'].nunique()}")
    print("\nDistribucion:")
    print(df_combined['genre'].value_counts().sort_index())

    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nArchivo guardado: {OUTPUT_FILE}")

if __name__ == '__main__':
    gtzan_processing()
    combination_gtzan_folklore()
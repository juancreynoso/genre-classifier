import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

print("="*60)
print("EXPERIMENTO B: PREDICCION DE FOLKLORE CON ENTRENAMIENTO")
print("="*60)

# LOAD GTZAN + FOLKLORE
df_combined = pd.read_csv('my_features_combined.csv')
print(f"\nGTZAN + FOLKLORE cargado: {len(df_combined)} muestras")

# SPLIT DE FOLKLORE
df_folklore = df_combined[df_combined['genre'] == 'folklore'].sample(frac=1, random_state=None).reset_index(drop=True)

if len(df_folklore) < 2:
    print("\nError: Se necesitan al menos 2 canciones de folklore")
    exit()

# Copia una cancion de test del dataset folklore
df_folklore_test = df_folklore.iloc[:1].copy()

print(f"\nSplit de folklore (RANDOM):")
print(f"  Train: {len(df_folklore) - 1} canciones")
print(f"  Test:  1 cancion")
print(f"  Cancion de test: {df_folklore_test['filename'].values[0]}")

# DATASET DE ENTRENAMIENTO: todo menos la cancion de test
df_train = df_combined[df_combined['filename'] != df_folklore_test['filename'].values[0]]

print(f"\nDataset de entrenamiento: {len(df_train)} muestras")
print(f"Generos: {df_train['genre'].nunique()}")
print("\nDistribucion:")
print(df_train['genre'].value_counts().sort_index())

# ENTRENAR CON 11 GENEROS
print("\nENTRENAMIENTO CON FOLKLORE INCLUIDO")

X = df_train.drop(['filename', 'genre'], axis=1).values
y = df_train['genre'].values

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
print(f"Test: {len(X_test_gtzan)} muestras")

print("\nEntrenando SVM con 11 generos (GTZAN + Folklore)...")
model = SVC(kernel='rbf', random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_gtzan_scaled)
accuracy = accuracy_score(y_test_gtzan, y_pred)

print(f"Accuracy en el entrenamiento: {accuracy:.2%}")

# PRUEBA CON CANCION
print("\nPREDICCION CON CANCION DE FOLKLORE NO ENTRENADA")

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
    print(f"  {genre:12s}: {probabilities[i]:5.1%}")


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


# GUARDAR MODELO
pickle.dump(model, open('models/svm_with_folklore.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler_with_folklore.pkl', 'wb'))
pickle.dump(label_encoder, open('models/label_encoder_with_folklore.pkl', 'wb'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from features import load_folklore_features, feature_columns

print("="*60)
print("EXPERIMENTO A: PREDICCION DE FOLKLORE SIN ENTRENAMIENTO")
print("="*60)


# CARGAR MODELO
print("\nCargando modelo SVM entrenado...")

model = pickle.load(open('models/svm_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

print(f"Generos conocidos: {list(label_encoder.classes_)}")


# PROCESAR FOLKLORE
# Las features se cargan del cache (folklore_features.csv); si no existe se
# extraen una sola vez. Ya no se re-extrae el audio en cada corrida.
df_folklore = load_folklore_features()
folklore_names = list(df_folklore['filename'].values)
X_folklore = df_folklore[feature_columns(df_folklore)].values
X_folklore_scaled = scaler.transform(X_folklore)

# Predecir
predictions = model.predict(X_folklore_scaled)
probabilities = model.predict_proba(X_folklore_scaled)


# RESULTADOS
print("PREDICCIONES DEL MODELO")

results_df = pd.DataFrame({
    'Cancion': folklore_names,
    'Genero Predicho': label_encoder.inverse_transform(predictions),
    'Confianza': np.max(probabilities, axis=1)
})

print("\n", results_df.to_string(index=False))


# ESTADISTICAS
print("\nESTADISTICAS")

pred_counts = pd.Series(
    label_encoder.inverse_transform(predictions)
).value_counts()

print("\nEl modelo clasifico folklore argentino esta confianza:")
for genre, count in pred_counts.items():
    percentage = (count / len(predictions)) * 100
    print(f"  {genre:12s}: {count:2d} canciones ({percentage:5.1f}%)")


# CONFIANZA

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


# CONCLUSIONES

most_common = pred_counts.index[0]
most_common_pct = (pred_counts.iloc[0] / len(predictions)) * 100

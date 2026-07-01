import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from features import load_folklore_features, feature_columns

N_SPLITS = 5  # k-fold

os.makedirs('results', exist_ok=True)

print("="*60)
print("EXPERIMENTO B: FOLKLORE CON CROSS-VALIDATION")
print("="*60)


# ============================================
# CARGAR DATOS
# ============================================
df_gtzan = pd.read_csv('my_features.csv')
print(f"\nGTZAN cargado: {len(df_gtzan)} muestras")

df_folklore = load_folklore_features()  # desde cache; ver features.py
print(f"Folklore listo: {len(df_folklore)} muestras")

if len(df_folklore) < N_SPLITS:
    print(f"\nError: se necesitan al menos {N_SPLITS} canciones de folklore")
    exit()

feature_cols = feature_columns(df_gtzan)
X_folklore = df_folklore[feature_cols].values
folklore_names = df_folklore['filename'].values
X_gtzan = df_gtzan[feature_cols].values
y_gtzan = df_gtzan['genre'].values


# ============================================
# CROSS-VALIDATION (K-FOLD SOBRE EL FOLKLORE)
# ============================================
# En cada fold entrenamos con GTZAN + 4/5 del folklore y probamos con el 1/5
# restante. Asi cada cancion de folklore se evalua una vez sobre un modelo que
# no la vio, y el resultado no depende de un unico split (mas estable).
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_predictions = np.empty(len(df_folklore), dtype=object)  # prediccion por cancion
fold_accs = []
dummy_y = np.zeros(len(df_folklore))  # StratifiedKFold pide etiquetas; folklore es 1 sola clase

print("\n" + "="*60)
print(f"CROSS VALIDATION ({N_SPLITS}-FOLD)")
print("="*60)

for fold, (train_idx, test_idx) in enumerate(kf.split(X_folklore, dummy_y), 1):
    # Train = GTZAN + folklore de entrenamiento ; Test = folklore apartado
    X_train = np.vstack([X_gtzan, X_folklore[train_idx]])
    y_train = np.concatenate([y_gtzan, ['folklore'] * len(train_idx)])

    le = LabelEncoder()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)              # fit SOLO con train
    X_test_scaled = scaler.transform(X_folklore[test_idx])

    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_scaled, le.fit_transform(y_train))
    pred = le.inverse_transform(model.predict(X_test_scaled))

    oof_predictions[test_idx] = pred
    acc = np.mean(pred == 'folklore')
    fold_accs.append(acc)
    print(f"  Fold {fold}: {len(test_idx):2d} canciones -> reconocidas como folklore: {acc:.1%}")

fold_accs = np.array(fold_accs)


# ============================================
# RESULTADOS
# ============================================
print("\n" + "="*60)
print("RESULTADOS")
print("="*60)

recall = np.mean(oof_predictions == 'folklore')
print(f"\nFolklore reconocido (recall): {recall:.1%} (+/- {fold_accs.std():.1%})  "
      f"-> {np.sum(oof_predictions == 'folklore')}/{len(df_folklore)} canciones")

print("\nDistribucion de predicciones (a donde 'cae' cada cancion):")
pred_counts = pd.Series(oof_predictions).value_counts()
for genre, count in pred_counts.items():
    marca = "  <- correcto" if genre == 'folklore' else ""
    print(f"  {genre:12s}: {count:2d} ({count/len(df_folklore)*100:5.1f}%){marca}")

# Canciones que NO se reconocen como folklore (utiles para el analisis)
mal = [(folklore_names[i], oof_predictions[i])
       for i in range(len(df_folklore)) if oof_predictions[i] != 'folklore']
if mal:
    print(f"\nCanciones NO reconocidas como folklore ({len(mal)}):")
    for name, pred in mal:
        print(f"  {name:35s} -> {pred}")


# ============================================
# VISUALIZACION: a que genero se asigna el folklore
# ============================================
colors = ['green' if g == 'folklore' else 'coral' for g in pred_counts.index]
plt.figure(figsize=(8, 5))
plt.bar(pred_counts.index, pred_counts.values, color=colors, edgecolor='black', alpha=0.85)
plt.title(f'Folklore: recall {recall:.0%} (cross-validation {N_SPLITS}-fold)', fontweight='bold')
plt.xlabel('Genero predicho')
plt.ylabel('Numero de canciones')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/folklore_crossvalidation.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nGrafico guardado: results/folklore_crossvalidation.png")

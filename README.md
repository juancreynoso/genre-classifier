# Clasificador de Géneros Musicales

Proyecto de **Inteligencia Artificial** que clasifica canciones automáticamente usando machine learning y análisis acústico.

## Descripción

Este proyecto implementa un clasificador de géneros musicales que:
- Extrae características acústicas de archivos de audio usando `librosa`
- Entrena dos modelos (SVM y Red Neuronal) para clasificar en 10 géneros
- Experimenta con folklore argentino como género no visto
- Valida la capacidad de generalización del modelo

### Experimentos principales

**Experimento A:** ¿Qué pasa cuando el modelo ve un género que nunca entrenó?
- Resultado: clasifica la mayoría del folklore como country (~69%)
- Conclusión: detecta similitudes acústicas superficiales (guitarra acústica)

**Experimento B:** ¿Puede aprender un nuevo género con pocas muestras?
- Resultado: con 70 muestras logra **65.7% ± 9.5%** de recall (cross-validation 5-fold)
- Conclusión: las features funcionan; el problema del Exp. A era falta de ejemplos

**Comparación de modelos:** ¿SVM o Red Neuronal?
- En la misma tarea (11 clases, mismo split) quedan **parejos** (~66–68%)
- Conclusión: con un dataset chico, la representación importa más que el modelo

---

## Resultados

| Modelo | Dataset | Métrica |
|--------|---------|----------|
| SVM | GTZAN (10 géneros) | 68% accuracy |
| Red Neuronal | GTZAN + folklore (11 clases) | ~67% accuracy |
| SVM | GTZAN + folklore (11 clases) | 68% accuracy |
| SVM | Folklore sin entrenar (Exp A) | mayoría → country |
| SVM | Folklore con 70 muestras (Exp B, CV) | 65.7% ± 9.5% recall |

**Baseline aleatorio:** 10% (10 géneros) / ~9% (11 géneros)

---

## Instalación

### Prerrequisitos

- Python 3.8+
- pip

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Activar entorno
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar dataset GTZAN

Descarga el dataset desde [Kaggle - GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Estructura esperada:
```
Data/
└── genres_original/
    ├── blues/
    │   ├── blues.00000.wav
    │   └── ...
    ├── classical/
    ├── country/
    └── ... (10 géneros en total)
```

### 5. (Opcional) Agregar folklore argentino

Para el Experimento B, agrega canciones de folklore en:
```
Data/
└── folklore_test/
    ├── folklore_01.wav
    ├── folklore_02.wav
    └── ...
```

> **Nota sobre archivos generados:** los CSV de features (`my_features.csv`, etc.)
> y los modelos (`models/`) **no se versionan** (están en `.gitignore`) porque son
> regenerables y hacen pesado el repo. Tras clonar, generálos corriendo el
> pipeline desde el script 1 (necesita el audio de GTZAN ya descargado).

---

## Uso

### Opción 1: Pipeline completo con Makefile

```bash
# Ver comandos disponibles
make help

# Ejecutar pipeline completo (recomendado)
make run-all

# O paso a paso:
make extract      # 1. Extraer features de GTZAN
make train        # 2. Entrenar SVM
make analyze      # 3. Folklore sin entrenar (Exp A)
make experiment   # 4. Folklore como clase, 1 canción (Exp B)
make crossval     # 5. Folklore con cross-validation (Exp B)
make compare      # 6. SVM vs Red Neuronal (11 clases)
```

### Opción 2: Scripts individuales

```bash
# 1. Extraer features (GTZAN) y armar el dataset combinado
python 1_data_extraction.py

# 2. Entrenar modelo SVM sobre GTZAN
python 2_model_training.py

# 3. Experimento A: folklore sin entrenar
python 3_folklore_analysis.py

# 4. Experimento B: folklore como clase (test de 1 canción)
python 4_folklore_training_and_testing.py

# 5. Experimento B: cross-validation (recomendado)
python 5_folklore_crossvalidation.py

# 6. Comparación SVM vs Red Neuronal (11 clases)
python 6_model_comparison.py
```

### Limpieza

```bash
# Borrar modelos y resultados (mantiene CSV)
make clean

# Borrar TODO
make clean-all
```

---

## Estructura del Proyecto

```
music-genre-classifier/
├── Data/
│   ├── genres_original/      # Dataset GTZAN
│   └── folklore_test/        # Canciones de folklore
├── results/                  # Gráficos generados
│   ├── confusion_matrix.png
│   ├── folklore_without_training.png
│   ├── folklore_crossvalidation.png
│   └── model_comparison.png
├── models/                   # Modelos entrenados (SVM x2 + scaler + encoder)
├── docs/                     # Guía de estudio (LaTeX + PDF)
├── features.py                         # Funciones compartidas de extracción
├── 1_data_extraction.py                # Extracción de features (GTZAN + combinado)
├── 2_model_training.py                 # Entrenamiento SVM
├── 3_folklore_analysis.py              # Experimento A (folklore sin entrenar)
├── 4_folklore_training_and_testing.py  # Experimento B (test de 1 canción)
├── 5_folklore_crossvalidation.py       # Experimento B (cross-validation)
├── 6_model_comparison.py               # SVM vs Red Neuronal (11 clases)
├── my_features.csv                     # Features de GTZAN (generado)
├── my_features_combined.csv            # GTZAN + folklore (generado)
├── folklore_features.csv               # Features de folklore (cache)
├── Makefile                            # Automatización
├── requirements.txt                    # Dependencias
└── README.md
```

---

## Metodología

### 1. Extracción de Features

Cada canción (30 segundos) se convierte en un vector de 45 números:

| Feature | Cantidad | Descripción |
|---------|----------|-------------|
| **MFCCs** | 26 | Capturan el "timbre" del sonido (mean + std) |
| **Chroma** | 12 | Capturan armonía y notas musicales |
| **Spectral Centroid** | 2 | "Brillo" del sonido (mean + std) |
| **Spectral Rolloff** | 1 | Punto donde cae 85% de energía |
| **Spectral Bandwidth** | 1 | "Ancho" del espectro |
| **Zero Crossing Rate** | 1 | Cruces por cero (útil para percusión) |
| **Tempo** | 1 | BPM estimado |
| **RMS Energy** | 1 | Energía promedio de la señal |

**Total:** 45 features

### 2. Modelo

**SVM (Support Vector Machine)** con kernel RBF:
- Funciona bien con datos de alta dimensión (45 features)
- Kernel RBF permite separaciones no lineales
- Eficiente con datasets medianos (~1000 muestras)

### 3. Validación

- **Train/Test split:** 80/20 estratificado
- **Normalización:** StandardScaler (media=0, std=1)
- **Cross-validation:** 5-fold sobre el folklore (cada canción se evalúa una vez sobre un modelo que no la vio)

---

## Resultados Detallados

### Confusiones más comunes

| Real | Predicho | Frecuencia | ¿Por qué? |
|------|----------|------------|-----------|
| Hiphop | Disco | 30% | Beats electrónicos similares |
| Disco | Hiphop | 25% | BPM y estructura parecida |
| Rock | Country | 15% | Instrumentación acústica similar |

### Análisis por género (GTZAN)

| Género | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Blues | 0.619 | 0.650 | 0.634 |
| Classical | 0.947 | 0.900 | 0.923 |
| Country | 0.667 | 0.700 | 0.683 |
| Disco | 0.476 | 0.500 | 0.488 |
| Hiphop | 0.478 | 0.550 | 0.512 |
| Jazz | 0.731 | 0.950 | 0.826 |
| Metal | 0.875 | 0.700 | 0.778 |
| Pop | 0.762 | 0.800 | 0.780 |
| Reggae | 0.611 | 0.550 | 0.579 |
| Rock | 0.714 | 0.500 | 0.588 |

---

## Conclusiones

### Aprendizajes

1. **Representación importa:** 
   - Features acústicas capturan propiedades musicales relevantes
   - MFCCs son cruciales para distinguir timbres

2. **Los modelos son literales:**
   - Aprenden exactamente lo que ven en entrenamiento
   - Sin folklore → no lo reconocen
   - Con folklore → aprenden (incluso con pocas muestras)

3. **Similitudes superficiales:**
   - Folklore → country por instrumentación acústica
   - No captura contexto cultural

### Limitaciones

- Dataset pequeño (100 canciones/género)
- Solo features acústicas (no considera letras, metadata)
- Folklore muy heterogéneo (chacarera ≠ zamba ≠ chamamé)
- GTZAN tiene canciones mal etiquetadas

---

## Tecnologías Utilizadas

- **Python 3.8+**
- **Librosa:** Análisis de audio y extracción de features
- **Scikit-learn:** Modelos de ML (SVM, preprocessing, métricas)
- **Pandas/NumPy:** Manipulación de datos
- **Matplotlib/Seaborn:** Visualización
- **TensorFlow/Keras:** CNN (opcional)

---

## Referencias

### Dataset
- [GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### Papers relacionados
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
- Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use.

### Documentación
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---

## Autores

- **Juan Cruz Reynoso**
- **Marcelo Juarez**
- **Jhonatan Calle Galeano**

---

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## FAQ

### ¿Por qué solo 68% de accuracy?

El 68% es un buen resultado considerando:
- Clasificación aleatoria daría 10%
- GTZAN es un dataset desafiante (géneros similares)
- Algunos géneros se solapan naturalmente (disco/hiphop, rock/country)
- Es comparable con resultados académicos (70-85% típico)

### ¿Por qué folklore se clasifica como country?

Porque comparten características acústicas:
- Instrumentación: guitarras, violines, percusión
- Estructuras armónicas simples
- Rango de tempo similar (90-140 BPM)
- Las features capturan similitudes superficiales, no contexto cultural

### ¿Cuántas canciones de folklore necesito?

- **Mínimo:** 10-15 para probar el experimento
- **Recomendado:** 50-100 para resultados robustos
- **Ideal:** 100+ separadas en sub-géneros (chacarera, zamba, chamamé)

### ¿Puedo usar mis propias canciones?

Sí, coloca archivos `.wav` o `.mp3` en `Data/folklore_test/` y ejecuta:
```bash
python 3_folklore_analysis.py
```

### ¿Funciona con otros géneros?

Sí, el modelo puede clasificar cualquier canción en los 10 géneros de GTZAN:
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Contacto

¿Preguntas? Abre un [issue](https://github.com/tu-usuario/music-genre-classifier/issues) en GitHub.

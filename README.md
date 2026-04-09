# 🎵 Clasificador de Géneros Musicales

Proyecto de **Inteligencia Artificial** que clasifica canciones automáticamente usando machine learning y análisis acústico.

## 📋 Descripción

Este proyecto implementa un clasificador de géneros musicales que:
- Extrae características acústicas de archivos de audio usando `librosa`
- Entrena modelos de ML (SVM) para clasificar en 10 géneros
- Experimenta con folklore argentino como género no visto
- Valida la capacidad de generalización del modelo

### Experimentos principales

**Experimento A:** ¿Qué pasa cuando el modelo ve un género que nunca entrenó?
- Resultado: Clasifica folklore como country (71%)
- Conclusión: Detecta similitudes acústicas superficiales

**Experimento B:** ¿Puede aprender un nuevo género con pocas muestras?
- Resultado: Con 13 muestras logra 57% accuracy (cross-validation)
- Conclusión: Las features funcionan, pero necesita más datos

---

## 🎯 Resultados

| Modelo | Dataset | Accuracy |
|--------|---------|----------|
| SVM | GTZAN (10 géneros) | 68% |
| SVM | Folklore sin entrenar | 0% (clasificado como country) |
| SVM | Folklore con 13 muestras | 57% |

**Baseline aleatorio:** 10% (1 de 10 géneros)

---

## 🚀 Instalación

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

---

## 📊 Uso

### Opción 1: Pipeline completo con Makefile

```bash
# Ver comandos disponibles
make help

# Ejecutar pipeline completo (recomendado)
make run-all

# O paso a paso:
make extract      # 1. Extraer features
make train        # 2. Entrenar SVM
make analyze      # 3. Analizar folklore sin entrenar
make experiment-cv # 4. Cross-validation con folklore
```

### Opción 2: Scripts individuales

```bash
# 1. Extraer features del dataset GTZAN
python 1_data_extraction.py

# 2. Entrenar modelo SVM
python 2_model_training.py

# 3. Experimento A: Folklore sin entrenar
python 3_folklore_analysis.py

# 4. Experimento B: Cross-validation con folklore
python 4b_folklore_crossvalidation.py

# (Opcional) 5. Entrenar CNN
python 5_cnn_training.py
```

### Limpieza

```bash
# Borrar modelos y resultados (mantiene CSV)
make clean

# Borrar TODO
make clean-all
```

---

## 📁 Estructura del Proyecto

```
music-genre-classifier/
├── Data/
│   ├── genres_original/      # Dataset GTZAN
│   └── folklore_test/         # Canciones de folklore (opcional)
├── results/                   # Gráficos generados
│   ├── confusion_matrix.png
│   ├── folklore_without_training.png
│   └── folklore_crossvalidation.png
├── models/                    # Modelos entrenados
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── 1_data_extraction.py                # Extracción de features
├── 2_model_training.py                 # Entrenamiento SVM
├── 3_folklore_analysis.py              # Experimento A
├── 4_folklore_training_and_testing_.py # Experimento B
├── 5_cnn_training.py                   # CNN (opcional)
├── Makefile                            # Automatización
├── requirements.txt                    # Dependencias
└── README.md                
```

---

## 🔬 Metodología

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
- **Cross-validation:** Leave-One-Out para folklore

---

## 📈 Resultados Detallados

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

## 🎓 Conclusiones

### ✅ Aprendizajes

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

### ⚠️ Limitaciones

- Dataset pequeño (100 canciones/género)
- Solo features acústicas (no considera letras, metadata)
- Folklore muy heterogéneo (chacarera ≠ zamba ≠ chamamé)
- GTZAN tiene canciones mal etiquetadas

### 🔮 Trabajo Futuro

- [ ] Aumentar dataset de folklore (50-100 muestras)
- [ ] Separar folklore en sub-géneros
- [ ] Probar CNN con mel-spectrograms
- [ ] Agregar features temporales (estructura de la canción)
- [ ] Data augmentation para aumentar muestras

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Librosa:** Análisis de audio y extracción de features
- **Scikit-learn:** Modelos de ML (SVM, preprocessing, métricas)
- **Pandas/NumPy:** Manipulación de datos
- **Matplotlib/Seaborn:** Visualización
- **TensorFlow/Keras:** CNN (opcional)

---

## 📚 Referencias

### Dataset
- [GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### Papers relacionados
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
- Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use.

### Documentación
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---

## 👥 Autores

- **Tu Nombre** - [GitHub](https://github.com/tu-usuario)
- **Marcelo** - Colaborador

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## 🙋 FAQ

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

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📞 Contacto

¿Preguntas? Abre un [issue](https://github.com/tu-usuario/music-genre-classifier/issues) en GitHub.

---

**⭐ Si te gustó el proyecto, dale una estrella en GitHub!**
.PHONY: help clean clean-all clean-csv extract train analyze experiment crossval compare run-all show-files show-results

# Variables
PYTHON = python3

help:
	@echo "=========================================="
	@echo "  Clasificador de Generos Musicales"
	@echo "=========================================="
	@echo ""
	@echo "Pipeline:"
	@echo "  make extract      - 1. Extraer features de GTZAN"
	@echo "  make train        - 2. Entrenar SVM (10 generos)"
	@echo "  make analyze      - 3. Folklore sin entrenar (Exp A)"
	@echo "  make experiment   - 4. Folklore como clase, 1 cancion (Exp B)"
	@echo "  make crossval     - 5. Folklore con cross-validation (Exp B)"
	@echo "  make compare      - 6. SVM vs Red Neuronal (11 clases)"
	@echo "  make run-all      - Pipeline completo (extract..compare)"
	@echo ""
	@echo "Limpieza:"
	@echo "  make clean        - Borrar modelos y resultados"
	@echo "  make clean-csv    - Borrar CSV de features (GTZAN + folklore)"
	@echo "  make clean-all    - Borrar TODO (CSV + modelos + resultados)"
	@echo ""
	@echo "Informacion:"
	@echo "  make show-files   - Mostrar archivos generados"
	@echo "  make show-results - Mostrar graficos generados"
	@echo ""

# Ejecutar pipeline completo (usa la version con cross-validation y la comparacion)
run-all: extract train analyze crossval compare


# Paso 1: Extraer features
extract:
	@echo "Extrayendo features con librosa..."
	$(PYTHON) 1_data_extraction.py

# Paso 2: Entrenar modelo
train:
	@echo "Entrenando modelo SVM..."
	$(PYTHON) 2_model_training.py

# Paso 3: Analizar folklore (Experimento A)
analyze:
	@echo "Analizando folklore sin entrenamiento..."
	$(PYTHON) 3_folklore_analysis.py

# Paso 4: Entrenar con folklore, 1 cancion de test (Experimento B)
experiment:
	@echo "Entrenando con folklore incluido (1 cancion de test)..."
	$(PYTHON) 4_folklore_training_and_testing.py

# Paso 5: Folklore con cross-validation (Experimento B, recomendado)
crossval:
	@echo "Evaluando folklore con cross-validation..."
	$(PYTHON) 5_folklore_crossvalidation.py

# Paso 6: Comparacion SVM vs Red Neuronal (11 clases)
compare:
	@echo "Comparando SVM vs Red Neuronal..."
	$(PYTHON) 6_model_comparison.py

# Limpiar archivos generados (mantiene CSV)
clean:
	@echo "Borrando modelos y resultados..."
	rm -rf models/
	rm -rf results/
	@echo "Limpieza completa"

# Limpiar solo CSV (GTZAN + cache de folklore)
clean-csv:
	@echo "Borrando CSV de features..."
	rm -f my_features.csv my_features_combined.csv folklore_features.csv
	@echo "CSV eliminados"

# Limpiar TODO
clean-all:
	@echo "Borrando TODOS los archivos generados..."
	rm -rf models/
	rm -rf results/
	rm -f my_features.csv folklore_features.csv
	rm -f *.pkl
	@echo "Limpieza total completa"

# Mostrar archivos generados
show-files:
	@echo "=========================================="
	@echo "  Archivos generados"
	@echo "=========================================="
	@echo ""
	@echo "CSV de features:"
	@ls -lh my_features.csv 2>/dev/null || echo "  (no existe)"
	@echo ""
	@echo "Modelos (models/):"
	@ls -lh models/ 2>/dev/null || echo "  (directorio vacio o no existe)"
	@echo ""
	@echo "Resultados (results/):"
	@ls -lh results/ 2>/dev/null || echo "  (directorio vacio o no existe)"
	@echo ""

# Mostrar resultados
show-results:
	@echo "Abriendo graficos generados..."
	@open results/confusion_matrix.png 2>/dev/null || echo "No encontrado: results/confusion_matrix.png"
	@open results/folklore_without_training.png 2>/dev/null || echo "No encontrado: results/folklore_without_training.png"
	@open results/folklore_comparison.png 2>/dev/null || echo "No encontrado: results/folklore_comparison.png"
.PHONY: help clean clean-all extract train analyze experiment run-all

# Variables
PYTHON = python3

help:
	@echo "=========================================="
	@echo "  Clasificador de Generos Musicales"
	@echo "=========================================="
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make extract      - Extraer features del dataset GTZAN"
	@echo "  make train        - Entrenar modelo SVM"
	@echo "  make analyze      - Analizar folklore sin entrenamiento (Exp A)"
	@echo "  make experiment   - Entrenar con folklore incluido (Exp B)"
	@echo "  make run-all      - Ejecutar pipeline completo"
	@echo ""
	@echo "Limpieza:"
	@echo "  make clean        - Borrar archivos generados (modelos y resultados)"
	@echo "  make clean-csv    - Borrar solo CSV de features"
	@echo "  make clean-all    - Borrar TODO (CSV + modelos + resultados)"
	@echo ""
	@echo "Informacion:"
	@echo "  make show-files   - Mostrar archivos generados"
	@echo "  make show-results - Mostrar graficos generados"
	@echo ""

# Ejecutar pipeline completo
run-all: extract train analyze experiment


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

# Paso 4: Entrenar con folklore (Experimento B)
experiment:
	@echo "Entrenando con folklore incluido..."
	$(PYTHON) 4_folklore_training_and_testing.py

# Limpiar archivos generados (mantiene CSV)
clean:
	@echo "Borrando modelos y resultados..."
	rm -rf models/
	rm -rf results/
	@echo "Limpieza completa"

# Limpiar solo CSV
clean-csv:
	@echo "Borrando CSV de features..."
	rm -f my_features.csv
	@echo "CSV eliminado"

# Limpiar TODO
clean-all:
	@echo "Borrando TODOS los archivos generados..."
	rm -rf models/
	rm -rf results/
	rm -f my_features.csv
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
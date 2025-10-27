# Sistema de Predicción de Inflación en España

Sistema de inteligencia artificial para predecir la evolución de la tasa de inflación en España utilizando datos históricos del Instituto Nacional de Estadística (INE). El sistema implementa múltiples modelos de machine learning (ARIMA, Random Forest, LSTM) para generar predicciones de inflación a 12 meses y produce informes técnicos completos con análisis económico.

## 🚀 Características Principales

- **Descarga Automática de Datos**: Extracción automatizada de series temporales del INE (IPC General, IPC por grupos, IPCA)
- **Procesamiento Inteligente**: Limpieza de datos, detección de outliers, y normalización automática
- **Modelos Múltiples**: Implementación de ARIMA, Random Forest y LSTM con selección automática del mejor modelo
- **Predicciones Robustas**: Generación de predicciones a 12 meses con intervalos de confianza
- **Informes Completos**: Generación automática de informes técnicos en PDF con visualizaciones y análisis económico
- **Monitoreo de Rendimiento**: Seguimiento en tiempo real del uso de recursos y optimización de memoria

## 📋 Requisitos del Sistema

### Requisitos Mínimos

- Python 3.8 o superior
- 4 GB de RAM disponible
- 2 GB de espacio en disco
- Conexión a internet para descarga de datos del INE

### Requisitos Recomendados

- Python 3.9+
- 8 GB de RAM
- 4 núcleos de CPU
- SSD para mejor rendimiento

## 🛠️ Instalación

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd prediccion-inflacion-espana
```

### 2. Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python src/main.py --help
```

## 🚀 Uso Rápido

### Ejecución Completa del Pipeline

```bash
python src/main.py
```

Este comando ejecutará todo el pipeline de predicción:

1. Descarga de datos del INE
2. Procesamiento y limpieza de datos
3. Ingeniería de características
4. Entrenamiento de modelos
5. Generación de predicciones
6. Creación de informes

### Ejecución por Módulos

#### Descargar Solo Datos

```python
from src.ine_extractor import INEExtractor

extractor = INEExtractor()
extractor.export_all_data("2020-01-01", "2024-12-31")
```

#### Procesar Datos Existentes

```python
from src.data_cleaner import DataProcessor

processor = DataProcessor()
data = processor.load_raw_data("data/raw/ipc_general.csv")
cleaned_data = processor.handle_missing_values(data)
```

#### Generar Solo Predicciones

```python
from src.predictor import Predictor

predictor = Predictor()
predictions = predictor.generate_predictions(12)  # 12 meses
```

## ⚙️ Configuración

El sistema utiliza el archivo `config/config.yaml` para toda la configuración. Las secciones principales son:

### Configuración de Datos

```yaml
data:
  start_date: "2002-01-01"
  end_date: "2024-12-31"
  retry:
    max_attempts: 3
    timeout: 30
```

### Configuración de Modelos

```yaml
models:
  arima:
    max_p: 5
    max_d: 2
    max_q: 5
  random_forest:
    n_estimators: 100
    max_depth: 10
  lstm:
    epochs: 100
    batch_size: 32
```

### Configuración de Predicciones

```yaml
prediction:
  horizon_months: 12
  confidence_level: 0.95
```

### Personalización de Rutas

```yaml
paths:
  data:
    raw: "data/raw/"
    processed: "data/processed/"
  models: "models/"
  reports: "reports/"
```

## 📊 Estructura de Salidas

El sistema genera los siguientes archivos de salida:

```
reports/
├── predictions.csv                    # Predicciones en formato CSV
├── predictions.json                   # Predicciones en formato JSON
├── technical_report.pdf               # Informe técnico completo
├── economic_analysis.json             # Análisis económico detallado
├── pipeline_execution_state.json     # Estado de ejecución del pipeline
└── visualizations/
    ├── inflation_trends.png
    ├── model_comparison.png
    └── predictions_chart.png
```

### Formato de Predicciones

#### CSV Format

```csv
fecha,predicted_inflation,confidence_lower,confidence_upper,model_used
2024-01-01,2.45,1.89,3.01,LSTM
2024-02-01,2.52,1.95,3.09,LSTM
```

#### JSON Format

```json
{
  "predictions": [
    {
      "fecha": "2024-01-01",
      "predicted_inflation": 2.45,
      "confidence_lower": 1.89,
      "confidence_upper": 3.01,
      "model_used": "LSTM"
    }
  ],
  "metadata": {
    "generation_date": "2024-01-15T10:30:00",
    "horizon_months": 12,
    "confidence_level": 0.95
  }
}
```

## 🧪 Ejecución de Tests

### Tests Unitarios

```bash
python -m pytest tests/test_data_processor.py -v
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/test_ine_extractor.py -v
```

### Tests de Integración

```bash
python -m pytest tests/test_integration_pipeline.py -v
```

### Ejecutar Todos los Tests

```bash
python tests/run_all_tests.py
```

## 📈 Monitoreo y Logs

### Logs del Sistema

Los logs se guardan en `logs/inflation_prediction.log` con información detallada sobre:

- Progreso de cada etapa del pipeline
- Uso de recursos del sistema
- Errores y advertencias
- Métricas de rendimiento

### Monitoreo en Tiempo Real

```python
from src.main import InflationPredictionPipeline

pipeline = InflationPredictionPipeline()
status = pipeline.get_pipeline_status()
print(f"Estado: {status['status']}")
print(f"Progreso: {status['completed_steps']}/{status['total_steps']}")
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

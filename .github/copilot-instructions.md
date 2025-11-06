# Instrucciones para agentes de IA en tensiones2

Este documento proporciona información esencial para trabajar con el código del Monitor automático de tensión.

## Arquitectura y Componentes Principales

La aplicación es una interfaz web Dash que monitorea y analiza datos de aceleración para calcular la tensión en tirantes. La estructura clave es:

- `app.py`: Punto de entrada principal que inicializa la aplicación Dash
- `tensiones_app/`: Módulo principal con los siguientes componentes:
  - `layout.py`: Define la interfaz de usuario completa con todos los controles
  - `analysis.py`: Contiene la lógica del procesamiento de señales y cálculos 
  - `callbacks.py`: Maneja la interactividad y actualizaciones en tiempo real
  - `storage.py`: Gestiona la persistencia de configuraciones

## Flujos de Datos Principales

1. Los archivos CSV se monitorean en un directorio configurable
2. Cada archivo debe contener un marcador `DATA_START` seguido por encabezados y datos
3. Los datos se procesan para:
   - Calcular el espectro de potencia (PSD)
   - Identificar la frecuencia fundamental y armónicos
   - Estimar la tensión del tirante usando parámetros físicos

## Convenciones del Proyecto

### Formato de Datos de Entrada
```
...contenido...
DATA_START
timestamp,sensor1,sensor2
123,0.1,0.2
124,0.15,0.25
...
```

### Configuración de Sensores
Se usa un formato JSON para mapear nombres de columnas a tirantes:
```json
{
    "canal_x": {"tirante": "Tirante Norte", "f0": 1.35, "ke": 0.42}
}
```

## Dependencias Críticas

- dash>=2.11: Framework web interactivo
- plotly>=5.18: Visualización de datos
- scipy>=1.9: Procesamiento de señales (FFT, detección de picos)

## Flujo de Desarrollo

1. Instalar dependencias: `pip install -r requirements.txt`
2. Iniciar servidor: `python app.py`
3. Acceder a: http://127.0.0.1:8050

## Patrones de Diseño

- Los parámetros de análisis se mantienen en el estado del cliente (dcc.Store)
- Las funciones de análisis en `analysis.py` son independientes de la UI
- Se usa `@dataclass` para estructuras de datos inmutables (ej: `AnalysisResults`)
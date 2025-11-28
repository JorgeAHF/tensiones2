# Monitor automático de tensión

Esta aplicación web, construida con Dash, monitoriza periódicamente un directorio en busca de nuevos archivos CSV con datos de aceleración. Cada vez que detecta un archivo, permite analizar un segmento configurable de la señal para estimar la frecuencia fundamental, identificar armónicos y calcular la tensión de un tirante a partir de sus propiedades físicas.

## Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

1. Coloca los archivos CSV en un directorio accesible (por defecto `./data`).
2. Inicia la aplicación:

```bash
python app.py
```

3. Abre el navegador en la URL indicada en la terminal (por defecto http://127.0.0.1:8050).
4. Configura los parámetros en la barra lateral:
   - Directorio a monitorear e intervalo de actualización.
   - Parámetros de análisis (frecuencia de muestreo, tamaño de ventana, etc.).
   - Rango porcentual del registro a estudiar.
   - Opcionalmente, una frecuencia fundamental sugerida.
   - Longitud y masa lineal del tirante para estimar la tensión.
5. La interfaz actualizará automáticamente la lista de archivos. Selecciona el deseado y revisa las gráficas (acelerograma completo, segmento, PSD y STFT) y la tabla de resultados.

Los archivos CSV ahora se generan por sensor con nombres como `sensor_10603_acceleration_20251114_162708_000-002min.csv` y contienen los encabezados:

```
timestamp_local,timestamp_utc,stay_id,sensor_id,fs_hz,ax_g,ay_g,az_g,is_valid
```

La aplicación toma el canal de aceleración disponible (priorizando `az_g`), filtra filas inválidas y, si defines un mapeo JSON en la barra lateral, renombra cada `sensor_id` al tirante configurado.

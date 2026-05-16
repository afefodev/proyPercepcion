# proyPercepcion

Proyecto base para clasificar frutas frescas y podridas con visión por computador.

## Estado actual

- Entorno virtual ya configurado.
- Dependencias instaladas en `requirements.txt`.
- Primer notebook preparado para exploración inicial del dataset.

## Estructura recomendada

- `data/`: dataset descargado desde Kaggle, no se sube al repositorio.
- `notebooks/`: análisis exploratorio y pruebas de modelos.
- `src/`: scripts reutilizables para entrenamiento e inferencia.
- `models/`: pesos del modelo entrenado.

## Siguiente paso

Abre `notebooks/01_exploracion_datos.ipynb`, descarga el dataset en `data/` y ejecuta las celdas en orden. Si la carpeta `data/train` no existe todavía, el notebook te lo indicará sin romperse.
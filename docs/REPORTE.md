# Informe del proyecto — Clasificación de frutas (fresh vs rotten)

Este documento resume el estado del proyecto, explica los tres notebooks existentes, presenta la interpretación de resultados y contiene puntos clave para defender el trabajo en una presentación.

---

## 1. Resumen ejecutivo

- Objetivo: construir un clasificador que identifique frutas frescas y podridas usando el dataset de Kaggle.
- Artefactos principales en el repositorio: `notebooks/01_exploracion_datos.ipynb`, `notebooks/02_baseline_HOG_SVM.ipynb`, `notebooks/03_modelo_final_transfer_learning.ipynb`.
- Métricas principales obtenidas:
  - Baseline (HOG + SVM): accuracy ≈ 0.80 en test.
  - Modelo final (EfficientNet‑B0 transfer learning): accuracy ≈ 0.9826 en test (reporte por clases incluido en el notebook).

---

## 2. Qué contiene cada notebook

- `01_exploracion_datos.ipynb`
  - Detecta `project_root` y valida la estructura `data/train` y `data/test`.
  - Muestra distribución por clase y ejemplos visuales.
  - Implementa funciones de preprocesamiento (`cargar_imagen`, `preprocesar_imagen`) y visualizaciones comparativas original vs preprocesado, diferencia absoluta e histogramas.

- `02_baseline_HOG_SVM.ipynb`
  - Implementa extracción de características HOG (Histogram of Oriented Gradients).
  - Entrena un SVM (clasificador) sobre los vectores HOG.
  - Evalúa en test y produce métricas (accuracy, classification_report, matriz de confusión).

- `03_modelo_final_transfer_learning.ipynb`
  - Carga `efficientnet_b0` preentrenada en ImageNet, reemplaza la cabeza por un `Dropout + Linear` para `num_classes`.
  - Define `train/val/test` loaders con augmentaciones en `train_transforms` y normalización ImageNet en `eval_transforms`.
  - Estrategia: freeze de `features` (entrenar la cabeza) y optimización con Adam + scheduler ReduceLROnPlateau; guarda el mejor checkpoint por `val_acc`.
  - Grafica curvas train/val de loss y accuracy y evalúa en test (muestra accuracy y classification_report).

---

## 3. Preprocesamiento y uso de los datos

- Estructura de datos: `data/train/<class>/*.jpg`, `data/test/<class>/*.jpg`.
- Carga en la CNN: `torchvision.datasets.ImageFolder` y split stratificado para `train`/`val`.
- Preprocesamiento principal:
  - Resize a 224×224, conversión a RGB.
  - Normalización con mean/std de ImageNet para la CNN.
  - Augmentaciones: RandomHorizontalFlip, RandomRotation(±15°), ColorJitter — ayudan a robustecer frente a variaciones.
- Para HOG: se usa tamaño reducido (ej. 64×64) y extracción de histogramas por celda.

---

## 4. Explicación técnica (conceptos esenciales)

- Época: una pasada completa por todo el conjunto de entrenamiento.
- Batch: subconjunto de ejemplos usados para una actualización de gradiente.
- Loss: función que mide el error (CrossEntropy en clasificación multiclasal).
- Optimización: Adam (parámetros adaptativos). Scheduler ReduceLROnPlateau baja el LR si la métrica de validación no mejora.
- Checkpoint: se guarda el estado del modelo con mejor `val_acc` para evaluación final.

---

## 5. HOG + SVM: cómo y por qué funciona (resumen)

- HOG: describe la distribución de orientaciones de gradientes en celdas locales; robusto a pequeñas variaciones pero información limitada (bordes y contornos).
- SVM: clasificador discriminativo que encuentra una frontera óptima en el espacio de características (puede usar kernel linear o RBF).
- Ventajas: explicable, rápido, no requiere GPU, funciona bien con conjuntos pequeños.
- Limitaciones: no captura texturas complejas ni patrones de color; representación fija (no aprende) → capacidad limitada.

---

## 6. Transfer learning (EfficientNet‑B0): bloque por bloque y por qué es superior

- Transfer learning: reutilizar una red preentrenada (ImageNet) como extractor de características; permite obtener representaciones ricas con pocos datos.
- EfficientNet‑B0 (resumen): arquitectura eficiente que usa bloques MBConv (inverted bottleneck + depthwise conv) y Squeeze‑and‑Excitation para recalibrar canales.
- Bloques principales:
  1. Stem (conv inicial, BN, activación).
  2. Series de MBConv blocks (expansión, depthwise conv, SE, proyección). Aumentan la abstracción espacial mientras reducen resolución.
  3. Head: Global Average Pooling → Dropout → Fully Connected (logits).
- Procedimiento en el notebook:
  - Cargar pesos preentrenados.
  - Congelar `features` y reemplazar `classifier` por una cabeza adaptada.
  - Entrenar cabeza (luego opcionalmente descongelar capas superiores para fine‑tuning).

Por qué mejor que HOG+SVM:
- Capacidad jerárquica: aprende low→mid→high level features.
- Preentrenamiento: filtros iniciales ya detectan bordes/texturas, lo que acelera y mejora el aprendizaje.
- Flexibilidad: la cabeza se adapta específicamente al problema.

---

## 7. Interpretación de resultados (por qué 0.80 vs 0.9826)

1. Representación:
   - HOG es un descriptor de bajo nivel (orientaciones). No modela color/ textura fina. Muchas características que distinguen fruta fresca vs podrida son sutiles en color o manchas.
   - EfficientNet ofrece representación jerárquica rica que captura texturas y patrones locales relevantes.

2. Capacidad del modelo:
   - SVM con HOG tiene menor capacidad (típicamente lineal en el espacio HOG si se usa kernel lineal) y no puede modelar fronteras complejas.
   - CNN tiene millones de parámetros que le permiten ajustar fronteras no lineales complejas.

3. Preprocesamiento y resolución:
   - HOG en imágenes pequeñas (p.ej. 64×64) pierde detalle; CNN en 224×224 conserva más información discriminativa.

4. Regularización y augmentaciones:
   - CNN se entrenó con augmentaciones y técnicas (dropout, scheduler), mejorando generalización.

5. Transferencia de conocimiento:
   - EfficientNet ya trae filtros útiles de ImageNet; con pocas iteraciones puede adaptarse al dominio.

En conjunto, estas razones explican la gran mejora de accuracy del modelo final.

---

## 8. Interpretación concreta del `classification_report` obtenido

- El reporte por clases muestra precision/recall/f1 altos (≥0.95 para casi todas las clases). Esto indica no solo alta accuracy global, sino también buen balance por clase.
- Revisar soporte (nº ejemplos por clase) en el test para confirmar suficiencia estadística (en este dataset los tamaños de test varían por clase). En el notebook figura el soporte por clase.
- Matriz de confusión: usarla para identificar confusiones específicas (p. ej. freshapples vs rottenapples). Si hay confusiones, revisar ejemplos mal clasificados con Grad‑CAM para interpretar.

---

## 9. Puntos clave para defender en una presentación

- Explica la diferencia conceptual: descriptor fijo (HOG) vs representaciones aprendidas (CNN).
- Señala decisiones de diseño: tamaño de entrada, augmentaciones, freeze de la base y guardar checkpoints por `val_acc`.
- Muestra evidencias: curvas de entrenamiento (loss/accuracy), matriz de confusión y classification_report (ya en notebook 03).
- Responde por qué transfer learning: rapidez de convergencia, mejor generalización con datos limitados.

---

## 10. Posibles preguntas y respuestas preparadas

- ¿Por qué transfer learning y no entrenar desde cero?
  - Entrenar desde cero requiere muchísimos datos y recursos. Transfer learning aprovecha conocimiento previo (filtros) y requiere menos datos y tiempo para converger bien.
- ¿Por qué EfficientNet y no ResNet o MobileNet?
  - EfficientNet ofrece un buen balance entre accuracy y eficiencia; ResNet/MobileNet son alternativas válidas que pueden probarse para comparar.
- ¿Hubo overfitting? ¿cómo se mitigó?
  - Se revisaron curvas train vs val; se usan augmentaciones, dropout y scheduler. Además, se guarda el mejor checkpoint por validación.
- ¿Qué mejoras rápidas propones?
  - Descongelar capas superiores y reentrenar (fine‑tuning), probar augmentations más fuertes o ensembles.

---

## 11. Acciones recomendadas antes de la defensa

1. Ejecutar `Restart Kernel and Run All` en los notebooks para dejar outputs reproducibles.
2. Confirmar que `models/` contiene el checkpoint final si lo quieres mostrar.
3. Preparar 2–3 ejemplos mal clasificados para discutir limitaciones.

---

Si quieres que guarde/actualice este archivo en otra ubicación o que lo reduzca a bullets para notas rápidas, dime dónde.

---

## 12. Explicación en profundidad — código, bloque por bloque

Esta sección desglosa las partes clave del código en los tres notebooks y explica qué hace cada bloque, por qué está ahí y qué efectos tiene en el entrenamiento/evaluación.

### Notebook 01 — Exploración y preprocesamiento

- Detección de `project_root` y rutas de datos:
  - Propósito: encontrar la raíz del proyecto de forma robusta (soporta ejecuciones desde distintos kernels/entornos).
  - Qué hace: comprueba variables de entorno y la presencia de carpetas esperadas (`data/train`, `data/test`); si no las encuentra lanza un error informativo.
  - Efecto: evita errores posteriores por rutas relativas y facilita reproducibilidad.

- `cargar_imagen(path)`:
  - Propósito: leer una imagen desde disco de forma segura y estandarizar el modo de color.
  - Qué hace: usa `PIL.Image.open(path).convert('RGB')` para asegurar 3 canales; captura excepciones I/O y devuelve `None` o lanza con mensaje claro si falla.
  - Efecto: previene errores por imágenes corruptas o en modo 'RGBA'/'L'.

- `preprocesar_imagen(image, size=(224,224))`:
  - Propósito: aplicar transformaciones determinísticas requeridas por modelos CNN.
  - Qué hace: `resize`, `center_crop` (si aplica), conversión a `numpy` o `torch.Tensor`, normalización por mean/std de ImageNet.
  - Consideraciones: el orden importa (resize antes de crop), y la normalización debe coincidir con los pesos preentrenados.

- Funciones de visualización (original vs preprocesada, diferencia, histogramas):
  - Propósito: comprobar visualmente efectos del preprocesado y detectar pérdidas de información.
  - Qué hace: calcula diferencia absoluta (`abs(orig - processed)`), histogramas por canal y muestra en una cuadrícula para inspección manual.
  - Efecto: ayuda a detectar si la normalización/clipping/resize introducen artefactos.

- Simulación de augmentaciones (función `simular_augmentaciones(image, n=6)`):
  - Propósito: mostrar ejemplos de cómo aumentos aleatorios afectan la apariencia (útil para justificar su uso).
  - Qué hace: aplica `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` con semilla reproducible y devuelve mosaico de imágenes.
  - Efecto: evidencia visual para la defensa sobre cómo las augmentaciones ayudan a generalizar.

### Notebook 02 — Baseline HOG + SVM

- Bloque: Carga y balanceo de datos
  - Qué hace: recorre `data/train` y `data/test`, lee imágenes, las redimensiona a un tamaño fijo (p.ej. 64×64) y construye arrays `X` y `y`.
  - Por qué: HOG funciona mejor con imágenes pequeñas y uniformes; su computación escala con resolución.

- Función `extraer_hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3))`:
  - Qué hace: convierte la imagen a escala de grises, calcula gradientes, binnea orientaciones y normaliza bloques.
  - Parámetros clave:
    - `orientations`: número de bins angulares; más bins → más precisión angular, pero mayor dimensionalidad.
    - `pixels_per_cell`: resolución espacial de la celda; más grande = menos detalle local.
    - `cells_per_block`: tamaño del bloque para normalización; afecta invarianza a iluminación.
  - Efecto: produce un vector de características que captura contornos y texturas locales.

- Escalado de características (`StandardScaler`) antes de SVM:
  - Qué hace: centra y escala cada dimensión a media 0 y varianza 1.
  - Por qué: SVM es sensible a la escala de las features, y la normalización mejora convergencia y rendimiento.

- Entrenamiento SVM (`sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')`):
  - Qué hace: ajusta un hiperplano en el espacio de características HOG; con RBF permite fronteras no lineales.
  - Consideraciones: `C` controla la penalización por errores (bias-variance tradeoff); `gamma` controla alcance del kernel.
  - Efecto: buen baseline rápido; útil para comparar representaciones aprendidas.

- Evaluación: `classification_report`, matriz de confusión y curva de aprendizaje rápida
  - Qué hace: calcula precision/recall/f1 por clase y muestra dónde el descriptor falla.
  - Efecto: identifica clases con alta confusión que requieren representaciones más ricas.

### Notebook 03 — Transfer Learning con EfficientNet‑B0

- Bloque: Definición de `train_transforms` y `eval_transforms`
  - `train_transforms`: combinan augmentaciones aleatorias (flip, rotation, color jitter), `RandomResizedCrop(224)`, conversión a tensor y `Normalize(mean,std)`.
  - `eval_transforms`: `Resize(256)` + `CenterCrop(224)` + `ToTensor` + `Normalize`.
  - Por qué: augmentaciones en `train` aumentan robustez; `eval` debe ser determinista para métricas reproducibles.

- Bloque: `ImageFolder` y `DataLoader` con `Subset` para split train/val
  - Qué hace: usa `torchvision.datasets.ImageFolder` para mapear carpetas a clases; construye índices para split estratificado (o `sklearn.model_selection.train_test_split` con `stratify=y`).
  - Parámetros: `batch_size`, `num_workers`, `shuffle=True` en entrenamiento.
  - Efecto: asegura batches balanceados y tasa de IO adecuada.

- Función `build_model(num_classes, pretrained=True)`:
  - Qué hace: carga `efficientnet_b0(weights=...)`, reemplaza la cabeza final por `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))`.
  - Congelado: si se congela, itera `for param in model.features.parameters(): param.requires_grad = False`.
  - Efecto: reduce parámetros entrenables, acelera convergencia y previene overfitting inicial.

- Función `run_epoch(model, loader, criterion, optimizer=None, device='cpu')`:
  - Modo entrenamiento vs evaluación: si `optimizer` es `None` está en `eval` (no backprop); si existe, activa `model.train()` y hace `loss.backward()` + `optimizer.step()`.
  - Qué hace dentro del batch loop:
    1. Mueve `inputs, targets` a `device`.
    2. `outputs = model(inputs)` → logits.
    3. `loss = criterion(outputs, targets)`.
    4. Si entrenando: `optimizer.zero_grad(); loss.backward(); optimizer.step()`.
    5. Calcula métricas (batch corrects, batch loss) y acumula para promedio.
  - Efecto: unificación de la lógica train/val, reduce duplicación y errores.

- Bloque: loop de entrenamiento principal
  - Estructura típica:
    - `best_val_acc = 0.0`
    - Para cada época:
      1. `train_loss, train_acc = run_epoch(..., optimizer=opt)`
      2. `val_loss, val_acc = run_epoch(..., optimizer=None)`
      3. `scheduler.step(val_loss)` o `scheduler.step(metric)` según implementación.
      4. Si `val_acc > best_val_acc`: guardar checkpoint con `torch.save({'model_state_dict': model.state_dict(), ...}, PATH)`.
  - Efecto: guarda el mejor modelo observado en validación y evita overfitting por early stopping manual.

- Guardado de checkpoints (`torch.save`) y carga (`torch.load`):
  - Guardado: serializa `model.state_dict()`, `optimizer.state_dict()`, `epoch`, `best_val_acc`.
  - Carga: `checkpoint = torch.load(PATH, map_location=device)` y `model.load_state_dict(checkpoint['model_state_dict'])`.
  - Efecto: reproducibilidad y capacidad de reanudar entrenamiento o evaluación exacta.

- Evaluación final y métricas:
  - Tras cargar el mejor checkpoint, el notebook ejecuta el modelo en el `test_loader`, recopila `y_true` y `y_pred`, y usa `sklearn.metrics.classification_report` y `confusion_matrix`.
  - Visualizaciones: matriz de confusión normalizada y `sns.heatmap(..., annot=True)` con `xticks/yticks` ajustados a `class_names`.

- Consideraciones prácticas y comprobaciones incluidas en el notebook:
  - Device detection: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` y mover modelo e inputs a `device`.
  - Seeds: fijar `random`, `numpy` y `torch.manual_seed` para reproducibilidad.
  - `num_workers` en `DataLoader`: si se usa Windows y `spawn` start method se tienen que ajustar `if __name__ == '__main__'` para pruebas fuera de notebooks.

---

Si quieres, puedo además:
- añadir fragmentos de código (snippets) concretos dentro del `docs/REPORTE.md` para cada función (por ejemplo la implementación de `run_epoch`),
- o generar una versión reducida para diapositivas con bullets y ejemplos visuales.


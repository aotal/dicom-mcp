# Análisis del Flujo de Cálculo de MTF: `main_mtf_processor.py` y `mtf_analyzer.py`

## 1. Propósito General y Flujo de Trabajo

Estos dos ficheros constituyen un sistema completo y avanzado para el cálculo de la **Función de Transferencia de Modulación (MTF)** a partir de imágenes DICOM. El proceso no se limita a una sola imagen, sino que está diseñado para **procesar un lote de imágenes, promediar los resultados para reducir el ruido y la variabilidad, y finalmente, generar un modelo matemático (un polinomio) que describe el rendimiento del sistema de imagen**.

El flujo de trabajo se orquesta desde `main_mtf_processor.py` y utiliza las herramientas matemáticas encapsuladas en `mtf_analyzer.py`. El proceso se puede resumir en los siguientes grandes pasos:

1.  **Configuración Inicial (`main_mtf_processor.py`)**: Se definen todos los parámetros del estudio: la lista de imágenes a analizar, la geometría de las ROIs, los parámetros para los algoritmos de cálculo y las rutas a los ficheros de calibración.

2.  **Procesamiento en Bucle (`main_mtf_processor.py`)**: El script itera sobre cada imagen DICOM.
    a.  **Extracción de ROIs**: Usa la clase `RoiExtractor` (analizada previamente) para obtener las dos regiones de interés (una para el borde vertical y otra para el horizontal) en formato *raw*.
    b.  **Análisis Individual**: Cada ROI es procesada por la clase `MtfAnalyzer` para obtener su curva MTF individual. Este paso incluye la **linealización** de la señal, un paso crítico para asegurar la precisión.

3.  **Agregación y Modelado (`main_mtf_processor.py` y `mtf_analyzer.py`)**:
    a.  **Promedio por Dirección**: Todas las MTFs verticales se promedian, y lo mismo se hace con las horizontales.
    b.  **Promedio Combinado**: Las dos curvas promedio (V y H) se combinan para obtener una única MTF global.
    c.  **Ajuste Polinómico**: Esta MTF global se ajusta a un polinomio, cuyos coeficientes se convierten en la salida final del análisis.

4.  **Visualización y Salida**: Se generan gráficos detallados en cada etapa y se muestran los coeficientes del polinomio final.

## 2. Análisis Detallado de Ficheros

### 2.1. `mtf_analyzer.py`: El Cerebro Matemático

Este fichero contiene toda la lógica para transformar una ROI en una curva MTF. Se divide en dos partes: funciones independientes y la clase `MtfAnalyzer`.

#### Funciones de Cálculo (Bloque 1):

Estas son funciones puramente matemáticas que operan sobre arrays de NumPy.

-   **`estimate_angle_from_midpoints_vectorized(...)`**: Una función altamente optimizada que determina el ángulo exacto del borde dentro de la ROI. Lo hace encontrando los puntos medios del gradiente en cada fila y realizando una regresión lineal sobre ellos. La versión vectorizada usa operaciones de NumPy para ser extremadamente rápida.
-   **`calculate_esf_from_roi(...)`**: Calcula la **Edge Spread Function (ESF)**. Proyecta todos los píxeles de la ROI en un eje perpendicular al borde (usando el ángulo calculado previamente) con un sobremuestreo (sub-pixelado), creando un perfil del borde de muy alta resolución. Aplica un filtro de Savitzky-Golay para suavizarlo.
-   **`calculate_lsf(...)`**: Calcula la **Line Spread Function (LSF)**. Lo hace derivando la ESF suavizada. También realiza una corrección de la línea base y aplica una ventana de Hanning para reducir el ruido espectral antes de la FFT.
-   **`calculate_mtf(...)`**: El paso final. Normaliza el área de la LSF y calcula su **Transformada Rápida de Fourier (FFT)**. El módulo de la FFT resultante es la MTF. La salida se filtra para mostrar solo las frecuencias de interés.

#### Clase `MtfAnalyzer` (Bloque 2):

Esta clase encapsula todo el proceso y gestiona los datos de calibración y los parámetros.

-   **`__init__(...)`**: El constructor es clave. Almacena los **datos de calibración** (un DataFrame de Pandas) y los **factores RQA** (un diccionario), que son necesarios para la linealización. También guarda todos los parámetros del análisis (tamaños de ventana, umbrales, etc.).
-   **`analyze_roi(...)`**: Es el método principal. Recibe una ROI *raw* y orquesta la secuencia de cálculo:
    1.  **Linealización**: Llama a `linearize_preprocessed_image_from_df` (del fichero `utils.py`) para corregir la señal de la ROI.
    2.  **Orientación**: Rota la ROI si es necesario para que el borde sea predominantemente vertical.
    3.  **Llamadas a las funciones de cálculo**: Invoca secuencialmente a `estimate_angle`, `calculate_esf`, `calculate_lsf` y `calculate_mtf`.
    4.  **Devuelve un diccionario** con todos los resultados, tanto finales (MTF) como intermedios (ESF, LSF, ángulo, etc.).
-   **`calculate_average_mtf(...)` y `calculate_grand_average_mtf(...)`**: Métodos estáticos para promediar múltiples curvas MTF. Interpolan todas las curvas a un eje de frecuencia común y luego calculan el promedio y la desviación estándar.
-   **`fit_average_mtf_polynomial(...)`**: Ajusta una curva MTF a un polinomio usando `np.polyfit`.
-   **Métodos de `plot_*`**: Un conjunto de funciones para generar gráficos detallados de cada paso del análisis, facilitando la depuración y la interpretación de los resultados.

### 2.2. `main_mtf_processor.py`: El Orquestador

Este script es el punto de entrada que utiliza las herramientas anteriores para realizar un análisis completo.

-   **Configuración**: La primera sección del `if __name__ == "__main__":` es un panel de control donde el usuario define:
    -   `dicom_files`: La lista de imágenes a procesar.
    -   `roi*_offset_mm`, `roi*_shape_yx`: La geometría de extracción de ROIs.
    -   `mtf_params`: Los parámetros para los algoritmos de `MtfAnalyzer`.
    -   `calibration_csv_path` y `RQA_FACTORS`: Rutas y datos para la calibración.
    -   Banderas para controlar qué gráficos se muestran (`plot_*`).
-   **Inicialización**: Carga los datos de calibración usando `obtener_datos_calibracion_vmp_k` de `utils.py` y crea una única instancia de `MtfAnalyzer` que se reutilizará para todas las imágenes.
-   **Bucle Principal**: Itera sobre la lista `dicom_files`.
    -   Llama a `RoiExtractor` para obtener las ROIs de la imagen actual.
    -   Determina el tipo de RQA (actualmente *hardcodeado* a 'RQA5', pero diseñado para ser dinámico).
    -   Llama a `analyzer.analyze_roi()` para cada una de las dos ROIs.
    -   Añade los resultados válidos a las listas `vertical_mtf_results` y `horizontal_mtf_results`.
-   **Cálculos Finales**: Una vez finalizado el bucle:
    -   Llama a `MtfAnalyzer.calculate_grand_average_mtf` para obtener las MTF promedio V y H.
    -   Calcula el promedio combinado de ambas.
    -   Llama a `analyzer.fit_average_mtf_polynomial` para obtener el modelo final.
-   **Salida**: Imprime los coeficientes del polinomio en la consola y muestra un gráfico final que resume todo el análisis.

## 3. Dependencias, Entradas y Salidas

### Dependencias Clave:

-   **Ficheros del Proyecto**: `roi_extractor.py`, `utils.py`.
-   **Librerías Externas**: `numpy`, `pandas`, `scipy`, `matplotlib`.

### Entradas Principales:

1.  **Imágenes DICOM**: Una lista de rutas a ficheros DICOM (`dicom_files` en `main_mtf_processor.py`).
2.  **Fichero de Calibración (`linearizacion.csv`)**: Un fichero CSV con columnas `VMP` y `K_uGy` que mapean los valores de píxel a dosis de radiación. Es **crítico** para la linealización.
3.  **Parámetros de Configuración**: Todos los diccionarios y variables definidos en la sección de configuración de `main_mtf_processor.py`.

### Salidas Principales:

1.  **Gráficos (`matplotlib`)**: El sistema genera múltiples gráficos para la depuración y un gráfico final que muestra la MTF promedio y su ajuste polinómico.
2.  **Salida a Consola**: Información detallada del proceso y, lo más importante, los **coeficientes del polinomio final** que modela la MTF.
3.  **Ficheros (Opcional)**: El código está preparado para guardar los coeficientes del polinomio en un fichero de texto o `.npy` si se descomentan las líneas correspondientes al final de `main_mtf_processor.py`.

## 4. Conclusión para Reutilización

El sistema está bien estructurado para su reutilización y expansión. La separación de la lógica matemática (`mtf_analyzer.py`) del flujo de control (`main_mtf_processor.py`) es una excelente práctica.

Para integrar este sistema en una herramienta más grande, los pasos serían:

1.  **Configurar las Entradas**: Proporcionar programáticamente la lista de ficheros DICOM y los datos de calibración.
2.  **Invocar el Proceso**: Se podría encapsular la lógica de `main_mtf_processor.py` en una función que acepte la configuración como argumento.
3.  **Capturar la Salida**: La función debería devolver el objeto `final_results`, que es un diccionario que contiene no solo los coeficientes del polinomio final, sino también todas las curvas MTF intermedias y promedios, permitiendo un análisis posterior o su almacenamiento en una base de datos.

La modularidad del código, especialmente la clase `MtfAnalyzer`, permite que el núcleo del cálculo de MTF sea un componente robusto y fiable dentro de un sistema de análisis de calidad de imagen más amplio.

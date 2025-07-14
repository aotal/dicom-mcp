# Plan de Implementación de la Funcionalidad MTF

## 1. Estado Inicial del Proyecto

Antes de comenzar, nos aseguraremos de que el proyecto se encuentra en el siguiente estado:

*   **`src/dicom_mcp/server.py`**: Contiene las herramientas básicas de DICOM (gestión de nodos, C-FIND, `get_dicomweb_pixel_data`) pero **sin ninguna función o importación relacionada con MTF**. La función `_fetch_dicom_dataset_from_dicomweb` no existe.
*   **`src/dicom_mcp/models.py`**: Contiene los modelos de datos existentes, pero **sin `MtfAnalysisResponse`**.
*   **`src/dicom_mcp/mtf_processor_wrapper.py`**: Este archivo **no existe**.
*   **`src/dicom_mcp/MTF/roi_extractor.py`**: Su constructor espera una `dicom_filepath`.
*   **`src/dicom_mcp/MTF/mtf_analyzer.py`**: La función `analyze_roi` realiza la linealización internamente.
*   **`src/dicom_mcp/MTF/utils.py`**: Contiene `linearize_preprocessed_image_from_df` y `obtener_datos_calibracion_vmp_k`.

## 2. Fases de Implementación

### Fase 1: Refactorizar la Extracción de DICOMweb en `server.py`

El objetivo es centralizar la lógica de descarga de datasets DICOM para su reutilización.

*   **Paso 1.1: Crear `_fetch_dicom_dataset_from_dicomweb`**
    *   **Acción**: Añadir una nueva función asíncrona `async def _fetch_dicom_dataset_from_dicomweb(...)` en `src/dicom_mcp/server.py`.
    *   **Responsabilidad**: Esta función se encargará de realizar la petición HTTP a DICOMweb, parsear la respuesta `multipart` y devolver un objeto `pydicom.Dataset`.
    *   **Ubicación**: Después de `_extract_pixel_array_info` y antes de `@mcp.prompt def simple_test_prompt`.

*   **Paso 1.2: Modificar `get_dicomweb_pixel_data`**
    *   **Acción**: Actualizar la función `async def get_dicomweb_pixel_data(...)` en `src/dicom_mcp/server.py` para que utilice `_fetch_dicom_dataset_from_dicomweb` para obtener el `pydicom.Dataset`.
    *   **Resultado**: `get_dicomweb_pixel_data` se vuelve más concisa y reutiliza la lógica de descarga.

### Fase 2: Crear una Función de Linealización Dedicada en `utils.py`

El objetivo es desacoplar la linealización de la lógica de `MtfAnalyzer`.

*   **Paso 2.1: Crear `linearize_pixel_array`**
    *   **Acción**: Añadir una nueva función `def linearize_pixel_array(...)` en `src/dicom_mcp/MTF/utils.py`.
    *   **Responsabilidad**: Esta función tomará un `np.ndarray` (el array de píxeles), el `calibration_df`, el `rqa_type` y `rqa_factors_dict`, y devolverá el array de píxeles linealizado. Contendrá la lógica de cálculo de la pendiente y la división.
    *   **Ubicación**: Antes de la función `linearize_preprocessed_image_from_df`.

*   **Paso 2.2: Modificar `linearize_preprocessed_image_from_df`**
    *   **Acción**: Modificar `def linearize_preprocessed_image_from_df(...)` en `src/dicom_mcp/MTF/utils.py` para que simplemente llame a la nueva función `linearize_pixel_array` con los argumentos adecuados.
    *   **Resultado**: `linearize_preprocessed_image_from_df` se convierte en un *wrapper* para la función más genérica.

### Fase 3: Adaptar `roi_extractor.py` y `mtf_analyzer.py`

El objetivo es que estas clases operen sobre datos ya procesados (arrays de píxeles y ROIs linealizadas).

*   **Paso 3.1: Modificar `RoiExtractor`**
    *   **Acción**: Modificar el constructor `__init__` de la clase `RoiExtractor` en `src/dicom_mcp/MTF/roi_extractor.py` para que acepte directamente un `pixel_array: np.ndarray` y `pixel_spacing_mm: float` en lugar de `dicom_filepath`.
    *   **Acción**: Eliminar la lógica de lectura de DICOM y extracción de `pixel_spacing` del constructor, ya que estos datos se pasarán directamente.
    *   **Resultado**: `RoiExtractor` se vuelve más modular y no necesita leer archivos DICOM.

*   **Paso 3.2: Modificar `MtfAnalyzer.analyze_roi`**
    *   **Acción**: Modificar la firma de `def analyze_roi(...)` en `src/dicom_mcp/MTF/mtf_analyzer.py` para que espere una `linearized_roi_array` (es decir, una ROI *ya linealizada*).
    *   **Acción**: Eliminar el paso de linealización (`linearize_preprocessed_image_from_df`) dentro de `analyze_roi`, ya que la ROI de entrada ya estará linealizada.
    *   **Resultado**: `MtfAnalyzer` se enfoca puramente en el cálculo de MTF, asumiendo que la linealización ya se ha realizado.

### Fase 4: Crear y Actualizar `mtf_processor_wrapper.py`

El objetivo es crear un orquestador central para el proceso MTF.

*   **Paso 4.1: Crear `src/dicom_mcp/mtf_processor_wrapper.py`**
    *   **Acción**: Crear el archivo `src/dicom_mcp/mtf_processor_wrapper.py`.
    *   **Contenido**: Implementar la función `def process_mtf_from_dicom_files(...)` en este archivo.
    *   **Responsabilidad de `process_mtf_from_dicom_files`**:
        *   Recibir una lista de `pydicom.Dataset` (no rutas de archivo).
        *   Cargar los datos de calibración (`obtener_datos_calibracion_vmp_k`).
        *   Iterar sobre cada `pydicom.Dataset`:
            *   Extraer `pixel_array` y `pixel_spacing`.
            *   Llamar a `linearize_pixel_array` (de `utils.py`) para linealizar el `pixel_array` completo.
            *   Instanciar `RoiExtractor` (modificada) con el `pixel_array` *linealizado* y `pixel_spacing`.
            *   Extraer las ROIs.
            *   Instanciar `MtfAnalyzer` y llamar a `analyze_roi` para cada ROI (que ya estarán linealizadas).
            *   Acumular los resultados de MTF individuales.
        *   Calcular los promedios de MTF (vertical, horizontal, combinado) y realizar el ajuste polinómico utilizando los métodos estáticos de `MtfAnalyzer`.
        *   Devolver un diccionario con los resultados finales (coeficientes del polinomio, R², RMSE, etc.).

### Fase 5: Actualizar `models.py`

El objetivo es definir la estructura de la respuesta para las herramientas MTF.

*   **Paso 5.1: Añadir `MtfAnalysisResponse`**
    *   **Acción**: Añadir la clase `MtfAnalysisResponse(BaseModel)` en `src/dicom_mcp/models.py`.
    *   **Estructura**: Incluir campos como `status`, `processed_files`, `valid_vertical_rois`, `valid_horizontal_rois`, `combined_poly_coeffs`, `fit_r_squared`, `fit_rmse`, `mtf_at_50_percent`, `error_details`.

### Fase 6: Integrar Herramientas MTF en `server.py`

El objetivo es exponer la funcionalidad MTF a través del servidor MCP.

*   **Paso 6.1: Añadir Importaciones**
    *   **Acción**: Añadir las importaciones necesarias en `src/dicom_mcp/server.py`:
        *   `MtfAnalysisResponse` de `models`.
        *   `process_mtf_from_dicom_files` de `mtf_processor_wrapper`.
        *   `tempfile`, `os` (si no están ya).

*   **Paso 6.2: Implementar `calculate_mtf_analysis`**
    *   **Acción**: Añadir la herramienta `@mcp.tool async def calculate_mtf_analysis(...)` en `src/dicom_mcp/server.py`.
    *   **Parámetros**: `study_instance_uid`, `series_instance_uid`, `sop_instance_uids: List[str]`, `rqa_type`.
    *   **Lógica**:
        *   Iterar sobre `sop_instance_uids`.
        *   Para cada UID, llamar a `_fetch_dicom_dataset_from_dicomweb` para obtener el `pydicom.Dataset`.
        *   Acumular los `pydicom.Dataset` en una lista.
        *   Llamar a `process_mtf_from_dicom_files` (del wrapper) con la lista de datasets y los parámetros MTF.
        *   Devolver `MtfAnalysisResponse` con los resultados.

*   **Paso 6.3: Implementar `calculate_mtf_analysis_from_series`**
    *   **Acción**: Añadir la herramienta `@mcp.tool async def calculate_mtf_analysis_from_series(...)` en `src/dicom_mcp/server.py`.
    *   **Parámetros**: `study_instance_uid`, `series_instance_uid`, `rqa_type`.
    *   **Lógica**:
        *   Utilizar `qido_web_query` para buscar instancias en la serie con `ImageComments = "MTF"` y obtener sus `SOPInstanceUID`.
        *   Reutilizar la lógica de descarga de datasets de `calculate_mtf_analysis` (o encapsularla en una función auxiliar si se repite mucho).
        *   Llamar a `process_mtf_from_dicom_files` (del wrapper) con la lista de datasets y los parámetros MTF.
        *   Devolver `MtfAnalysisResponse` con los resultados.

Este plan asegura una implementación modular, reutilizable y robusta de la funcionalidad MTF.

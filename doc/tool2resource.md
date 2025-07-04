# Evaluación de Herramientas y Recursos en `src/dicom_mcp/server.py`

Este documento evalúa las funciones decoradas con `@mcp.tool` en el fichero `src/dicom_mcp/server.py` para determinar cuáles deberían permanecer como `Tools` y cuáles podrían ser refactorizadas como `Resources` según la documentación de `fastmcp`.

## Criterios de Evaluación:

*   **Tools**: Funciones que realizan acciones, modifican el estado del servidor, o requieren lógica de procesamiento compleja con múltiples parámetros.
*   **Resources**: Fuentes de datos estáticas o templadas que el cliente puede leer.

## Plan de Refactorización:

### Convertir a Resource:

Las siguientes funciones serán refactorizadas como `Resources` debido a que su propósito principal es la recuperación de datos estáticos o la recuperación de datos específicos a través de una URI templada.

1.  **`list_dicom_nodes`**:
    *   **Razón**: Esta función devuelve una lista de nodos DICOM configurados, que es información estática del servidor. Puede ser expuesta como un recurso simple que el cliente puede leer para obtener la configuración actual de los nodos.
    *   **Propuesta de URI**: `/dicom_nodes`

2.  **`get_attribute_presets`**:
    *   **Razón**: Esta función proporciona los presets de atributos DICOM disponibles, que son datos de configuración estáticos. Es ideal para ser un recurso.
    *   **Propuesta de URI**: `/attribute_presets`

3.  **`get_dicomweb_pixel_data`**:
    *   **Razón**: Esta función recupera datos de píxeles para una instancia DICOM específica utilizando UIDs. Esto se alinea perfectamente con el concepto de un recurso templado, donde los UIDs pueden formar parte de la URI para acceder a los datos.
    *   **Propuesta de URI**: `/dicomweb/studies/{study_instance_uid}/series/{series_instance_uid}/instances/{sop_instance_uid}/pixel_data`

### Permanecer como Tool:

Las siguientes funciones permanecerán como `Tools` debido a que implican acciones, modificación de estado o consultas complejas que no se ajustan al modelo de `Resource`.

1.  **`switch_dicom_node`**:
    *   **Razón**: Esta función cambia el nodo DICOM activo, lo que implica una modificación del estado interno del servidor.

2.  **`verify_connection`**:
    *   **Razón**: Esta función realiza una operación activa (C-ECHO) para verificar la conectividad, lo cual es una acción.

3.  **`query_patients`**:
    *   **Razón**: Aunque es una operación de recuperación de datos, implica una consulta compleja con múltiples parámetros de filtrado y opciones de atributos, lo que la hace más adecuada como una `Tool`.

4.  **`query_studies`**:
    *   **Razón**: Similar a `query_patients`, esta función realiza una consulta compleja con múltiples parámetros para estudios.

5.  **`query_series`**:
    *   **Razón**: Similar a las consultas anteriores, esta función realiza una consulta compleja con múltiples parámetros para series.

6.  **`qido_web_query`**:
    *   **Razón**: Esta función realiza una consulta QIDO-RS que puede ser muy dinámica, con niveles de consulta variables y expansión de parámetros, lo que la hace más adecuada como una `Tool` para manejar la complejidad de la solicitud.

## Pasos Siguientes:

1.  Implementar los cambios en `src/dicom_mcp/server.py` para refactorizar las funciones identificadas como `Resources`.
2.  Asegurarse de que las nuevas `Resources` estén correctamente registradas en el servidor `FastMCP`.
3.  Actualizar cualquier código cliente que utilice estas funciones para que interactúe con las nuevas `Resources` o `Tools` según corresponda.
4.  Verificar la funcionalidad de todas las `Tools` y `Resources` después de la refactorización.

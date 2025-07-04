# Propuesta de Prompts para `src/dicom_mcp/server.py`

A continuación se presentan dos objetos `@mcp.prompt` diseñados para integrarse con la funcionalidad de consulta a PACS existente en `src/dicom_mcp/server.py`.

Estos prompts están pensados para ser añadidos dentro de la función `create_dicom_mcp_server`, después de las herramientas (`tools`) y antes de la sentencia `return mcp`.

## Propuestas de Prompts

### 1. `summarize_patient_studies`

Este prompt toma los datos de un paciente y una lista de sus estudios (obtenidos a través de las herramientas de consulta) y genera una solicitud para que un LLM cree un resumen clínico del historial de imágenes.

### 2. `explain_dicom_data`

Este prompt toma el resultado de una consulta DICOM (en formato JSON) y pide a un LLM que lo explique en términos sencillos, describiendo qué significa cada atributo.

## Código para `src/dicom_mcp/server.py`

```python
    @mcp.prompt
    def summarize_patient_studies(
        patient_name: str,
        patient_id: str,
        studies: List[Dict[str, Any]]
    ) -> str:
        """
        Genera un prompt para que un LLM resuma los estudios de imagen de un paciente.

        Este prompt toma información demográfica del paciente y una lista de sus estudios
        para crear un texto que solicita a un modelo de lenguaje que produzca un resumen clínico.

        Args:
            patient_name: El nombre del paciente.
            patient_id: El ID del paciente.
            studies: Una lista de diccionarios, donde cada diccionario representa un estudio
                     y contiene detalles como StudyDate, ModalitiesInStudy y StudyDescription.

        Returns:
            Una cadena de texto formateada lista para ser enviada a un modelo de lenguaje.
        """
        study_details = []
        for study in studies:
            details = (
                f"- Estudio en {study.get('StudyDate', 'fecha desconocida')} "
                f"({study.get('ModalitiesInStudy', 'modalidad desconocida')}): "
                f"{study.get('StudyDescription', 'sin descripción')}"
            )
            study_details.append(details)

        study_list = "\n".join(study_details)

        prompt = f"""
Eres un asistente clínico útil. Un radiólogo necesita un resumen del historial de imágenes reciente de un paciente.
Basado en los datos a continuación, proporciona un resumen breve y cronológico adecuado para una visión clínica general.

Nombre del Paciente: {patient_name}
ID del Paciente: {patient_id}

Estudios de Imagen:
{study_list}

Por favor, genera el resumen.
"""
        return prompt

    @mcp.prompt
    def explain_dicom_data(
        query_level: str,
        query_results: List[Dict[str, Any]]
    ) -> str:
        """
        Genera un prompt para explicar los resultados de una consulta DICOM.

        Este prompt toma los resultados de una consulta (paciente, estudio o serie) y
        los formatea en un prompt más grande que solicita a un modelo de lenguaje que explique
        los datos de una manera amigable para el usuario.

        Args:
            query_level: El nivel de la consulta (ej. 'Patient', 'Study', 'Series').
            query_results: La lista de resultados de la consulta DICOM.

        Returns:
            Una cadena de texto formateada lista para ser enviada a un modelo de lenguaje.
        """
        import json
        # Imprimir el JSON de forma bonita para una mejor legibilidad en el prompt
        results_str = json.dumps(query_results, indent=2)

        prompt = f"""
Eres un experto en DICOM. Un usuario ha realizado una consulta contra un PACS y ha recibido los siguientes datos.
Explica qué representan estos datos en términos sencillos. Describe los atributos clave y lo que significan.

Nivel de Consulta: {query_level}

Resultados de la Consulta (formato JSON):
```json
{results_str}
```

Por favor, proporciona una explicación clara y fácil de entender de estos resultados.
"""
        return prompt
```
hola mundo
```
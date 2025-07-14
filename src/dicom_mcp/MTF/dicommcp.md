# Documentación del Servidor DICOM MCP

## 1. Introducción

Este documento detalla el funcionamiento del servidor DICOM MCP, una aplicación construida en Python que actúa como una interfaz robusta y flexible para interactuar con sistemas DICOM (PACS, estaciones de trabajo, etc.). El servidor abstrae las complejidades de la comunicación de red DICOM, permitiendo a los usuarios realizar consultas y operaciones de una manera sencilla y estandarizada.

El proyecto se basa en dos librerías clave:

-   **FastMCP**: Un framework para construir servidores MCP (Model Context Protocol) de forma rápida y eficiente. Proporciona la estructura para definir "herramientas" (tools) y "recursos" (resources) que pueden ser invocados por un cliente.
-   **pynetdicom**: Una librería de Python pura para implementar el protocolo de red DICOM. Se utiliza para toda la comunicación DICOM subyacente, como C-FIND (búsqueda) y C-ECHO (verificación).
-   **httpx**: Una librería de cliente HTTP moderna para realizar peticiones a servidores DICOMweb (QIDO-RS y WADO-RS).

## 2. Arquitectura del Servidor

El servidor está orquestado principalmente por el fichero `src/dicom_mcp/server.py`, que utiliza otros módulos para organizar la lógica:

-   `server.py`: Define el punto de entrada principal. Utiliza `FastMCP` para crear el servidor y registrar todas las herramientas y recursos disponibles. Gestiona el ciclo de vida de la aplicación, incluyendo la carga de configuración y la inicialización del cliente DICOM.
-   `dicom_client.py`: Contiene la clase `DicomClient`, que encapsula la lógica para interactuar con un nodo DICOM utilizando `pynetdicom`. Es responsable de realizar operaciones como C-ECHO y C-FIND.
-   `config.py`: Gestiona la carga y validación de la configuración del servidor desde un fichero `configuration.yaml` utilizando `Pydantic`.
-   `models.py`: Define los modelos de datos `Pydantic` para las peticiones y respuestas de la API. Esto asegura que todos los datos intercambiados estén bien estructurados y validados.
-   `attributes.py`: Centraliza la gestión de los conjuntos de atributos DICOM (presets) que se pueden solicitar en las consultas (mínimo, estándar, extendido).

## 3. Funcionalidades Principales (Herramientas y Recursos)

El servidor expone su funcionalidad a través de un conjunto de herramientas (`@mcp.tool`) y recursos (`@mcp.resource`) definidos en `server.py`.

### 3.1. Gestión de Nodos DICOM

-   **`list_dicom_nodes`**: Devuelve una lista de todos los nodos DICOM configurados en `configuration.yaml` y muestra cuál es el nodo activo actualmente.
-   **`switch_dicom_node`**: Permite cambiar el nodo DICOM activo. Esto es útil para dirigir las operaciones a diferentes PACS o sistemas de archivo.
-   **`verify_connection`**: Realiza una operación C-ECHO (equivalente a un "ping" en DICOM) para verificar que la conexión con el nodo activo es correcta.

### 3.2. Consultas DICOM (C-FIND)

El servidor proporciona herramientas para buscar información en diferentes niveles de la jerarquía DICOM:

-   **`query_patients`**: Busca pacientes por nombre, ID o fecha de nacimiento.
-   **`query_studies`**: Busca estudios para un paciente, por fecha, modalidad, etc.
-   **`query_series`**: Busca series dentro de un estudio específico.

Todas estas funciones de consulta permiten personalizar los atributos devueltos mediante "presets" (`minimal`, `standard`, `extended`) y la inclusión/exclusión de atributos adicionales.

### 3.3. Integración con DICOMweb

El servidor también se integra con servicios DICOMweb para realizar consultas QIDO-RS y recuperar datos de píxeles WADO-RS.

-   **`qido_web_query`**: Realiza una consulta QIDO-RS (Query based on ID for DICOM Objects) a un servidor DICOMweb. Permite buscar estudios, series o instancias y es más flexible que el C-FIND tradicional.
-   **`get_dicomweb_pixel_data`**: Recupera los datos de píxeles de una instancia DICOM específica a través de WADO-RS (Web Access to DICOM Objects).

## 4. Análisis Profundo de `get_dicomweb_pixel_data`

Esta función es un recurso (`@mcp.resource`) diseñado para obtener los datos de imagen (píxeles) de una instancia DICOM desde un servidor DICOMweb. A diferencia de las operaciones C-GET o C-MOVE que son parte del protocolo DICOM estándar sobre TCP, esta función utiliza el estándar DICOMweb, que se basa en HTTP.

La lógica de esta función es fundamental para entender cómo el servidor puede extenderse para manejar datos de imagen y otros tipos de datos binarios a través de la web.

### 4.1. Funcionamiento Paso a Paso

1.  **Construcción de la URL**: La función primero construye la URL de WADO-RS. Esta URL tiene un formato estandarizado que identifica de forma única el estudio, la serie y la instancia (SOPInstanceUID) que se desea recuperar.
    ```python
    dicomweb_url = (
        f"{config.dicomweb_url}/studies/{study_instance_uid}/"
        f"series/{series_instance_uid}/instances/{sop_instance_uid}"
    )
    ```

2.  **Petición HTTP con `httpx`**: Se utiliza la librería `httpx` para realizar una petición `GET` asíncrona a la URL construida. Es crucial establecer la cabecera `Accept` correctamente para indicar al servidor que queremos recibir la respuesta en un formato específico. En este caso, se solicita `multipart/related; type="application/dicom"`, que es el formato estándar para encapsular un fichero DICOM dentro de una respuesta HTTP.
    ```python
    response = await client.get(
        dicomweb_url, 
        headers={"Accept": 'multipart/related; type="application/dicom"'},
        timeout=30.0
    )
    ```

3.  **Análisis de la Respuesta `multipart`**: La respuesta del servidor no es simplemente el fichero DICOM binario, sino una respuesta HTTP `multipart`. Esto significa que el cuerpo de la respuesta está dividido en varias partes, similar a un correo electrónico con archivos adjuntos. Una de estas partes contiene el fichero DICOM.
    -   Para analizar esta respuesta, el código reconstruye primero las cabeceras y el cuerpo para que puedan ser procesados por la clase `BytesParser` del módulo `email` de Python.
    -   Luego, itera sobre las partes de la respuesta (`parsed_msg.walk()`) hasta encontrar la que tiene el `Content-Type` igual a `application/dicom`.

4.  **Lectura del Fichero DICOM con `pydicom`**: Una vez que se extraen los bytes del fichero DICOM, se utiliza la librería `pydicom` para leerlos en memoria. `pydicom.filebase.DicomBytesIO` permite a `pydicom` tratar un buffer de bytes como si fuera un fichero en disco.
    ```python
    ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes), force=True)
    ```

5.  **Extracción de la Información de Píxeles**: Con el objeto DICOM (`ds`) cargado, la función auxiliar `_extract_pixel_array_info` extrae la información relevante:
    -   Accede al array de píxeles a través de `ds.pixel_array`. `pydicom` se encarga de decodificar los datos de píxeles en un array de NumPy.
    -   Obtiene las dimensiones de la imagen (filas y columnas) de las etiquetas DICOM correspondientes.
    -   Genera una pequeña vista previa (preview) de la matriz de píxeles para que el usuario pueda tener una idea de los datos sin necesidad de transferir la imagen completa.

6.  **Respuesta Estructurada**: Finalmente, la función devuelve un objeto `PixelDataResponse` (un modelo Pydantic) que contiene toda la información extraída de forma estructurada: dimensiones, tipo de datos del array de píxeles y la vista previa.

### 4.2. Relevancia para Futuras Implementaciones

La lógica de `get_dicomweb_pixel_data` es un excelente punto de partida para implementar nuevas funcionalidades que requieran:
-   **Recuperación de otros tipos de datos**: La misma técnica se puede usar para recuperar otros objetos DICOM, como informes de radiología estructurados (Structured Reports) o PDFs encapsulados.
-   **Procesamiento de imágenes**: Una vez que se tiene el `pixel_array`, se pueden aplicar algoritmos de procesamiento de imágenes, realizar mediciones o incluso alimentar los datos a modelos de inteligencia artificial.
-   **Interacción con otros servicios web**: El uso de `httpx` demuestra cómo el servidor puede actuar como un proxy o un cliente para otros servicios web, no solo para DICOMweb.

## 5. Configuración

El comportamiento del servidor se controla a través del fichero `configuration.yaml`. Este fichero define:

-   **`nodes`**: Un diccionario con la configuración de cada nodo DICOM al que el servidor puede conectarse (host, puerto, AE Title).
-   **`current_node`**: El nombre del nodo que se utilizará por defecto.
-   **`calling_aet`**: El AE Title que este servidor MCP utilizará al iniciar conexiones DICOM.
-   **`dicomweb_url`**: La URL base del servidor DICOMweb para las operaciones QIDO-RS y WADO-RS.

## 6. Modelos de Datos con Pydantic

El uso de `Pydantic` en `models.py` es crucial para la robustez del servidor. Cada "tool" y "resource" define un modelo de respuesta, lo que garantiza que la salida siempre sea consistente y esté validada.

Una característica importante es el validador universal en la clase base `DicomResponseBase`, que convierte automáticamente tipos de datos específicos de `pydicom` (como `PN` para nombres de pacientes o `IS` para cadenas de enteros) a tipos primitivos de Python (strings, ints), asegurando la compatibilidad y una serialización JSON predecible.

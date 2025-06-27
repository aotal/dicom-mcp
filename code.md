# Análisis del Proyecto dicom-mcp

## Resumen del Proyecto

El proyecto `dicom-mcp` es un servidor que implementa el **Model Context Protocol (MCP)**. Su objetivo principal es actuar como un puente o intermediario entre un asistente de inteligencia artificial y sistemas de imagen médica que utilizan el protocolo DICOM (como servidores PACS, VNAs, etc.).

Permite al asistente realizar operaciones estándar de DICOM, como consultas de pacientes/estudios y movimiento de datos, a través de un conjunto de herramientas (tools) definidas en el servidor.

## Arquitectura y Componentes Clave

1.  **Punto de Entrada (`__main__.py`):** Es el script que inicia el servidor. Parsea los argumentos de la línea de comandos, principalmente la ruta al fichero de configuración, y arranca el servidor MCP.

2.  **Configuración (`config.py`, `configuration.yaml`):** La configuración se gestiona a través de un fichero YAML. `config.py` utiliza la librería `pydantic` para cargar y validar esta configuración, que define los nodos DICOM a los que se puede conectar (host, puerto, AE Title) y el AE Title que usará el propio servidor (`calling_aet`).

3.  **Servidor MCP (`server.py`):** Utiliza la librería `mcp` (específicamente `FastMCP`) para crear el servidor. Define un ciclo de vida (`lifespan`) que inicializa el cliente DICOM al arrancar. Expone las funcionalidades como "tools" (`@mcp.tool()`), que son las funciones que el asistente de IA podrá invocar. Ejemplos: `query_patients`, `move_series`, `extract_pdf_text_from_dicom`, etc.

4.  **Cliente DICOM (`dicom_client.py`):** Es el corazón de la lógica de comunicación DICOM. Utiliza la librería `pynetdicom` para realizar las operaciones de red:
    *   `C-ECHO` para verificar la conexión.
    *   `C-FIND` para realizar búsquedas (pacientes, estudios, series, instancias).
    *   `C-MOVE` para enviar estudios o series a otros nodos DICOM.
    *   `C-GET` para recuperar instancias específicas (como los PDF encapsulados).
    *   También utiliza `PyPDF2` para extraer el texto de los informes en PDF que se obtienen de los ficheros DICOM.

## Funcionamiento

El flujo de trabajo es el siguiente:
1.  El servidor `dicom-mcp` se inicia y lee el `configuration.yaml` para saber a qué nodos DICOM puede conectarse.
2.  Un asistente de IA se conecta al servidor `dicom-mcp`.
3.  El asistente invoca una de las herramientas, por ejemplo, `query_patients(name_pattern="DOE*")`.
4.  El `server.py` recibe la llamada, utiliza la instancia de `DicomClient` y ejecuta el método correspondiente (`query_patient`).
5.  El `dicom_client.py` construye la consulta DICOM adecuada y la envía al nodo DICOM configurado actualmente usando `pynetdicom`.
6.  El resultado se devuelve al asistente a través del servidor MCP.

## Análisis de Librerías

### `pynetdicom`

El fichero `dicom_client.py` utiliza `pynetdicom` de una manera muy correcta y robusta, siguiendo las mejores prácticas:

*   **Entidad de Aplicación (AE):** Se crea una única instancia de `AE` en el constructor del `DicomClient` a la que se le añaden todos los "contextos de presentación" necesarios.
*   **Asociación:** Cada método sigue el ciclo de vida correcto de la asociación: `ae.associate()`, `assoc.is_established`, `assoc.send_*()`, y `assoc.release()`.
*   **Operaciones DICOM:**
    *   **C-FIND:** Las funciones `query_*` construyen un `Dataset` de `pydicom` y lo envían con `assoc.send_c_find()`.
    *   **C-MOVE:** Las funciones `move_*` llaman a `assoc.send_c_move()` especificando el AE de destino.
    *   **C-GET:** La función `extract_pdf_text_from_dicom` implementa correctamente el flujo C-GET, manejando el evento `EVT_C_STORE` para recibir los ficheros.

### `mcp`

Aunque no se encontró la documentación exacta para la librería `mcp` usada (`mcp[cli]>=1.3.0`), su funcionamiento se puede inferir a partir de `server.py`:

*   **`FastMCP(name, lifespan)`**: Es la clase principal del servidor. El `lifespan` es un gestor de contexto asíncrono que se usa para inicializar recursos (como el `DicomClient`).
*   **`@mcp.tool()`**: Es un decorador que expone una función como una herramienta para que un cliente MCP la pueda invocar. La signatura de la función y los docstrings definen la especificación de la herramienta.
*   **`mcp.run(transport)`**: Inicia el servidor y define el método de comunicación (ej. `stdio`).

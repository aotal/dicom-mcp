# DICOM MCP Server

A powerful, high-performance [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for interacting with DICOM PACS archives. This project provides a bridge between modern AI models and standard medical imaging systems, supporting both DICOMweb (QIDO-RS, WADO-RS) and traditional DIMSE (C-FIND, C-ECHO) protocols.

Built with [**FastMCP**](https://gofastmcp.com/), `pynetdicom`, and `httpx`.

> **Note**: This project was originally forked from [ChristianHinge/dicom-mcp](https://github.com/ChristianHinge/dicom-mcp).

## ✨ Key Features

* **Dual Protocol Support**: Interact with your PACS using modern, web-friendly **DICOMweb** services or traditional **DIMSE** operations.

* **Dynamic Node Management**: Switch between multiple configured PACS nodes (`list_dicom_nodes`, `switch_dicom_node`) and verify connectivity (`verify_connection`) on the fly.

* **Advanced Querying**:

  * A generic and powerful **QIDO-RS** query tool (`qido_web_query`) for flexible searches.

  * Helper tools for standard C-FIND queries at the Patient, Study, and Series levels.

* **Advanced Image Analysis**: Includes a high-level tool (`analyze_mtf_for_series`) to perform complete **Modulation Transfer Function (MTF)** analysis on an entire DICOM series with a single call.

* **Configuration-Driven**: Easily configure all your DICOM nodes and server settings in a simple `configuration.yaml` file.

* **Structured & Validated I/O**: Leverages Pydantic models for all tool inputs and outputs, ensuring data consistency and clear API contracts.

## ⚙️ Configuration

Before running the server, you must create a `configuration.yaml` file. The server will look for this file in the `src/dicom_mcp/` directory by default.

Here is an example `configuration.yaml`:

```yaml
# DICOM nodes this server can connect to
nodes:
  pacs_local:
    host: 127.0.0.1
    port: 11112
    ae_title: "DCM4CHEE"
    description: "Local dcm4chee-arc-light instance."
  orthanc_test:
    host: 192.168.1.100
    port: 4242
    ae_title: "ORTHANC"
    description: "Orthanc test server."

# The node to use on startup
current_node: "pacs_local"

# The AE Title this MCP server will use when initiating connections
calling_aet: "DICOM_MCP"

# Base URL for the DICOMweb services (QIDO-RS, WADO-RS)
dicomweb_url: "http://localhost:8080/dcm4chee-arc/aets/DCM4CHEE/rs"

# Timeout in seconds for DICOMweb HTTP requests
dicomweb_timeout: 60.0

# Directory for storing files (e.g., received via C-STORE SCP)
local_storage_dir: "./dicom_storage"
```

## 🚀 Installation and Running the Server

1.  **Install the package**:

    For regular use, install the package from the root of the repository:

    ```bash
    pip install .
    ```

    For development, install in editable mode:

    ```bash
    pip install -e .
    ```

2.  **Run the server**:

    You can run the server from your terminal. By default, it uses the `stdio` transport, but you can also specify `sse` for Server-Sent Events.

    ```bash
    # Run with stdio transport
    python -m src.dicom_mcp

    # Run with SSE transport
    python -m src.dicom_mcp --transport sse
    ```

## 🛠️ Available Tools (API Reference)

This server exposes a powerful set of tools through the MCP protocol. Here are some of the key tools available:

### Node Management

* **`list_dicom_nodes()`**: Lists all DICOM nodes configured in `configuration.yaml` and indicates which one is currently active.

* **`switch_dicom_node(node_name: str)`**: Switches the active DICOM connection to another configured node.

* **`verify_connection()`**: Performs a DICOM C-ECHO to verify connectivity with the currently active node.

### DICOMweb Querying

* **`qido_web_query(query_level: str, query_params: dict)`**: A generic and powerful tool to perform any QIDO-RS query.

  * `query_level`: The resource path (e.g., `studies`, `series`, `studies/{uid}/instances`).

  * `query_params`: A dictionary with filter criteria (e.g., `{"PatientName": "DOE*", "includefield": "PatientID"}`).

* **`find_mtf_instances_in_series(study_instance_uid: str, series_instance_uid: str)`**: A high-level tool that finds and returns **only** the instances within a series that have `ImageComments` set to "MTF".

  > **Note**: This tool internally performs a broad query and then filters the results. This ensures accuracy even if the PACS does not support filtering on the `ImageComments` attribute.

### MTF Analysis

* **`analyze_mtf_for_series(study_instance_uid: str, series_instance_uid: str)`**: The main workflow tool. It automatically finds all instances with `ImageComments` = "MTF" in the specified series, downloads their data in memory, and returns a complete, averaged MTF analysis.

* **`calculate_mtf_from_instances(study_instance_uid: str, series_instance_uid: str, sop_instance_uids: List[str])`**: A lower-level tool that calculates the averaged MTF for an explicit list of SOP Instance UIDs.

## 🏛️ Architecture Overview

* **MCP Framework**: [FastMCP](https://gofastmcp.com/) is used to create the server, tools, and resources.

* **DIMSE Protocol**: [pynetdicom](https://github.com/pydicom/pynetdicom) is used for traditional DICOM networking (C-FIND, C-ECHO).

* **DICOMweb Protocol**: [httpx](https://www.python-httpx.org/) is used for making modern RESTful requests to QIDO-RS and WADO-RS endpoints.

* **Configuration**: [PyYAML](https://pyyaml.org/) and [Pydantic](https://docs.pydantic.dev/) are used for loading and validating the `configuration.yaml` file.

* **DICOM File Handling**: [pydicom](https://github.com/pydicom/pydicom) is used for reading and parsing DICOM datasets.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
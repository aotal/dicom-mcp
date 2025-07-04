# DICOM Model Context Protocol (MCP) Server

[![PyPI - Version](https://img.shields.io/pypi/v/dicom-mcp)](https://pypi.org/project/dicom-mcp/)
[![Python Version](https://img.shields.io/pypi/pyversions/dicom-mcp)](https://pypi.org/project/dicom-mcp/)
[![License](https://img.shields.io/github/license/your-repo/dicom-mcp)](LICENSE)

## Overview

This project implements a DICOM Model Context Protocol (MCP) server, providing a robust and flexible interface for interacting with DICOM (Digital Imaging and Communications in Medicine) servers. It abstracts the complexities of DICOM networking, allowing for seamless integration with various medical imaging systems (PACS, workstations, etc.).

The server is built using Python and leverages `pynetdicom` for DICOM communication and `FastMCP` for its core server framework.

## Key Features

The DICOM MCP Server offers a comprehensive set of functionalities for managing and querying DICOM data:

*   **DICOM Node Management**:
    *   List all configured DICOM nodes.
    *   Switch between different DICOM nodes for operations.
    *   Verify connectivity to DICOM nodes using C-ECHO.

*   **Query & Retrieve (C-FIND)**:
    *   **Patient Queries**: Search for patient records based on various criteria (ID, name, birth date).
    *   **Study Queries**: Find studies by patient ID, date, modality, description, accession number, or Study Instance UID.
    *   **Series Queries**: Query series within a specific study by modality, series number, description, or Series Instance UID.
    *   **Instance Queries**: Retrieve specific instances within a series.
    *   **Attribute Presets**: Utilize predefined attribute sets (minimal, standard, extended) for query results, with options to include/exclude additional attributes.

*   **DICOMweb Integration (QIDO-RS, WADO-RS)**:
    *   **QIDO-RS Queries**: Perform QIDO-RS queries to retrieve studies, series, or instances directly from a configured DICOMweb server. Supports expansion of predefined attribute sets.
    *   **WADO-RS Pixel Data Retrieval**: Fetch pixel data for DICOM instances via WADO-RS, providing image dimensions, data type, and a preview of the pixel array.

*   **DICOM C-MOVE Operations**:
    *   Move entire studies or specific series to another DICOM node.

*   **Encapsulated PDF Handling**:
    *   Retrieve DICOM instances containing encapsulated PDF documents and extract their text content.

*   **LLM Integration Prompts**:
    *   Generate structured prompts for Large Language Models (LLMs) to explain DICOM query results or specific DICOM attributes in a user-friendly manner.

## Configuration

The server's behavior and DICOM node connections are managed through a `configuration.yaml` file. This file defines:

*   **`nodes`**: A dictionary of DICOM nodes, each with a `host`, `port`, `ae_title` (Application Entity Title), and an optional `description`.
*   **`current_node`**: The name of the currently active DICOM node for operations.
*   **`calling_aet`**: The AE title that this MCP server will use when initiating DICOM connections.
*   **`local_storage_dir`**: The directory where received DICOM files will be stored by the SCP (if enabled).
*   **`dicomweb_url`**: The base URL for the DICOMweb server to be used for QIDO-RS and WADO-RS operations.

An example `configuration.yaml` is provided in `src/dicom_mcp/configuration.yaml`.

## Data Models

The project uses `Pydantic` for data validation and serialization, defining clear and robust models for DICOM responses and requests. The `DicomResponseBase` model includes a universal validator to convert non-primitive Python data types (especially `pydicom` specific types) to strings for consistent output.

## Getting Started

To run the DICOM MCP server:

1.  **Install Dependencies**: Ensure you have all required Python packages installed. It's recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```
    (Note: A `requirements.txt` can be generated from `pyproject.toml` using `pip freeze > requirements.txt` or `uv pip freeze > requirements.txt` if using `uv`).

    The core dependencies are:
    *   `httpx`
    *   `fastmcp`
    *   `numpy`
    *   `pynetdicom`
    *   `pypdf2`
    *   `pyyaml`
    *   `pydicom`

2.  **Configure**: Edit the `src/dicom_mcp/configuration.yaml` file to define your DICOM nodes and DICOMweb URL.

3.  **Run the Server**: Execute the main script:
    ```bash
    python -m src.dicom_mcp
    ```
    You can specify a custom configuration path and transport protocol:
    ```bash
    python -m src.dicom_mcp /path/to/your/config.yaml --transport sse
    ```

## Usage Examples

Here are some examples of how to interact with the DICOM MCP server's functionalities (assuming you are interacting with the MCP server via its defined tools and resources):

### DICOM Node Management

*   **List DICOM Nodes**:
    ```python
    # Example interaction with the MCP server to list nodes
    # (This would typically be called by an LLM or client application)
    # mcp_server.resource("resource://dicom_nodes").get()
    ```

*   **Switch DICOM Node**:
    ```python
    # Example interaction with the MCP server to switch nodes
    # mcp_server.tool("switch_dicom_node").call(node_name="orthanc")
    ```

*   **Verify Connection**:
    ```python
    # Example interaction with the MCP server to verify connection
    # mcp_server.tool("verify_connection").call()
    ```

### Query & Retrieve

*   **Query Patients**:
    ```python
    # Example interaction with the MCP server to query patients
    # mcp_server.tool("query_patients").call(name_pattern="DOE*", patient_id="12345")
    ```

*   **Query Studies**:
    ```python
    # Example interaction with the MCP server to query studies
    # mcp_server.tool("query_studies").call(patient_id="12345", study_date="20230101-20230131")
    ```

*   **Query Series**:
    ```python
    # Example interaction with the MCP server to query series
    # mcp_server.tool("query_series").call(study_instance_uid="1.2.3.4.5.6", modality="CT")
    ```

*   **Get DICOMweb Pixel Data**:
    ```python
    # Example interaction with the MCP server to get pixel data
    # mcp_server.resource("dicomweb://studies/{study_uid}/series/{series_uid}/instances/{sop_uid}/pixeldata").get(
    #     study_instance_uid="1.2.3.4", series_instance_uid="1.2.3.4.5", sop_instance_uid="1.2.3.4.5.6"
    # )
    ```

### DICOM C-MOVE Operations

*   **Move Series**:
    ```python
    # Example interaction with the MCP server to move a series
    # mcp_server.tool("move_series").call(destination_ae="DEST_AE", series_instance_uid="1.2.3.4.5")
    ```

*   **Move Study**:
    ```python
    # Example interaction with the MCP server to move a study
    # mcp_server.tool("move_study").call(destination_ae="DEST_AE", study_instance_uid="1.2.3.4")
    ```

### Encapsulated PDF Handling

*   **Extract PDF Text from DICOM**:
    ```python
    # Example interaction with the MCP server to extract PDF text
    # mcp_server.tool("extract_pdf_text_from_dicom").call(
    #     study_instance_uid="1.2.3.4", series_instance_uid="1.2.3.4.5", sop_instance_uid="1.2.3.4.5.6"
    # )
    ```

### LLM Integration Prompts

*   **Explain DICOM Attribute**:
    ```python
    # Example interaction with the MCP server to generate an LLM prompt
    # mcp_server.prompt("explain_dicom_attribute").call(attribute_name="PatientName", attribute_value="DOE^JOHN")
    ```

## Project Structure

```
dicom-mcp/
├── .github/                 # GitHub Actions workflows
├── doc/                     # Additional documentation and notes
├── src/
│   └── dicom_mcp/           # Main application source code
│       ├── __init__.py      # Package initialization
│       ├── __main__.py      # Main entry point for the server
│       ├── attributes.py    # DICOM attribute presets
│       ├── config.py        # Configuration loading and models
│       ├── dicom_client.py  # DICOM client functionalities (C-FIND, C-MOVE, C-GET)
│       ├── dicom_scp.py     # DICOM Storage SCP implementation
│       ├── models.py        # Pydantic data models for requests/responses
│       └── server.py        # FastMCP server definition and tool/resource registration
├── tests/                   # Unit and integration tests
├── .gitignore
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # This file
├── LICENSE                  # Project license
└── ...
```

## Contribution

Contributions are welcome! Please refer to the project's `.github/workflows/release.yml` for CI/CD setup and `pyproject.toml` for project dependencies. Follow standard Python development practices and ensure tests pass before submitting pull requests.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
# DICOM Model Context Protocol (MCP) Server

## Overview

This project implements a DICOM Model Context Protocol (MCP) server, providing a robust and flexible interface for interacting with DICOM (Digital Imaging and Communications in Medicine) servers. It abstracts the complexities of DICOM networking, allowing for seamless integration with various medical imaging systems (PACS, workstations, etc.).

The server is built using Python and leverages `pynetdicom` for DICOM communication and `FastMCP` for its core server framework.

## Key Features

The DICOM MCP Server offers a comprehensive set of functionalities for managing and querying DICOM data:

*   **DICOM Node Management**: 
    *   List all configured DICOM nodes.
    *   Switch between different DICOM nodes for operations.
    *   Verify connectivity to DICOM nodes using C-ECHO.

*   **Query & Retrieve (C-FIND, C-GET)**:
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

*   **DICOM C-STORE SCP**: 
    *   Includes a DICOM Storage SCP (Service Class Provider) that can receive and store DICOM files from other DICOM entities.

*   **Encapsulated PDF Handling**: 
    *   Retrieve DICOM instances containing encapsulated PDF documents and extract their text content.

*   **LLM Integration Prompts**: 
    *   Generate structured prompts for Large Language Models (LLMs) to summarize patient studies or explain DICOM query results in a user-friendly manner.

## Configuration

The server's behavior and DICOM node connections are managed through a `configuration.yaml` file. This file defines:

*   **`nodes`**: A dictionary of DICOM nodes, each with a `host`, `port`, `ae_title` (Application Entity Title), and an optional `description`.
*   **`current_node`**: The name of the currently active DICOM node for operations.
*   **`calling_aet`**: The AE title that this MCP server will use when initiating DICOM connections.
*   **`local_storage_dir`**: The directory where received DICOM files will be stored by the SCP.
*   **`dicomweb_url`**: The base URL for the DICOMweb server to be used for QIDO-RS and WADO-RS operations.

An example `configuration.yaml` is provided in `src/dicom_mcp/configuration.yaml`.

## Data Models

The project uses `Pydantic` for data validation and serialization, defining clear and robust models for DICOM responses and requests. The `DicomResponseBase` model includes a universal validator to convert non-primitive Python data types (especially `pydicom` specific types) to strings for consistent output.

## Getting Started

To run the DICOM MCP server:

1.  **Install Dependencies**: Ensure you have all required Python packages installed (e.g., `pynetdicom`, `fastmcp`, `pydicom`, `PyPDF2`, `httpx`, `pyyaml`, `pydantic`, `numpy`). You can typically install them via `pip install -r requirements.txt` (assuming a `requirements.txt` is generated from `pyproject.toml`).

2.  **Configure**: Edit the `src/dicom_mcp/configuration.yaml` file to define your DICOM nodes and DICOMweb URL.

3.  **Run the Server**: Execute the main script:
    ```bash
    python -m src.dicom_mcp
    ```
    You can specify a custom configuration path and transport protocol:
    ```bash
    python -m src.dicom_mcp /path/to/your/config.yaml --transport sse
    ```

## Documentation

The `doc/` directory contains additional documentation and notes:

*   `llms-full.txt`: Full LLM related content.
*   `prompts.md`: Details on prompt generation for LLMs.
*   `tool2resource.md`: Information on tool to resource mapping.

## Contribution

Contributions are welcome! Please refer to the project's `.github/workflows/release.yml` for CI/CD setup and `pyproject.toml` for project dependencies. Follow standard Python development practices and ensure tests pass before submitting pull requests.
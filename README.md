Of course\! Here is the complete, corrected, and updated `README.md` file translated into English.

# DICOM MCP Server

A powerful, high-performance [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for interacting with DICOM PACS archives. This project provides a bridge between modern AI models and standard medical imaging systems, supporting both DICOMweb (QIDO-RS, WADO-RS) and traditional DIMSE (C-FIND, C-ECHO) protocols.

Built with [**FastMCP**](https://gofastmcp.com/), `pynetdicom`, and `httpx`.

> **Note**: This project was originally forked from [ChristianHinge/dicom-mcp](https://github.com/ChristianHinge/dicom-mcp).

## ✨ Key Features

  * **Dual Protocol Support**: Interact with your PACS using modern, web-friendly **DICOMweb** services or traditional **DIMSE** operations.
  * **Dynamic Node Management**: Switch between multiple configured PACS nodes (`switch_dicom_node`) and verify connectivity (`verify_connection`) on the fly.
  * **Advanced Querying**:
      * A generic and powerful **QIDO-RS** query tool (`qido_web_query`) for flexible searches.
      * Helper tools for standard C-FIND queries at the Patient, Study, and Series levels.
  * **Advanced Image Analysis**: Includes a high-level tool (`analyze_mtf_for_series`) to perform a complete **Modulation Transfer Function (MTF)** analysis on an entire DICOM series with a single call.
  * **Configuration-Driven**: Easily configure all your DICOM nodes and server settings in a simple `configuration.yaml` file.
  * **Structured & Validated I/O**: Leverages Pydantic models for all tool inputs and outputs, ensuring data consistency and clear API contracts.

-----

## ⚙️ Configuration

Before running the server, you must create a `configuration.yaml` file. By default, the server will look for this file in the project's root directory.

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

-----

## 🚀 Installation and Running the Server

The `fastmcp` ecosystem recommends using **uv** for dependency and environment management.

### 1\. Installation

To contribute to development, clone the repository and install the dependencies in editable mode:

```bash
# Clone the repository
git clone https://github.com/aotal/dicom-mcp.git
cd dicom-mcp

# Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate

# Install the package and its development dependencies
uv pip install -e ".[dev]"
```

### 2\. Running the Server

You can run the server in several ways, depending on your needs.

#### HTTP Mode (Recommended for Development and Client Testing)

Start the server to listen for HTTP requests. This is the best option for testing with clients or development tools.

```bash
uv run python -m src.dicom_mcp --transport http --port 8000
```

The server will be available at `http://127.0.0.1:8000/mcp/`.

#### Stdio Mode (For Local Clients like Claude Desktop)

This mode is for integration with clients that manage the server process directly.

```bash
uv run python -m src.dicom_mcp --transport stdio
```

### 3\. Connecting with a Test Client

With the server running in HTTP mode, you can use a test client to interact with it:

```python
# test_client.py
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp/") as client:
        print("✅ Connection established.")
        
        # Call a tool
        result = await client.call_tool("verify_connection")
        print(f"Result of verify_connection: {result.data.message}")
        
        # Read a resource
        resource = await client.read_resource("resource://dicom_nodes")
        print(f"Configured nodes: {resource[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
```

-----

## 🛠️ Available Tools (API Reference)

### Node Management

  * **`list_dicom_nodes()`**: (`Resource`) Lists configured DICOM nodes and indicates the active one.
  * **`switch_dicom_node(node_name: str)`**: (`Tool`) Switches the active DICOM node.
  * **`verify_connection()`**: (`Tool`) Tests the connection to the active node using C-ECHO.

### DICOMweb Querying

  * **`qido_web_query(query_level: str, query_params: dict)`**: (`Tool`) Performs a generic QIDO-RS query.
  * **`find_mtf_instances_in_series(study_instance_uid: str, series_instance_uid: str)`**: (`Tool`) Finds and returns only instances with `ImageComments='MTF'` in a series.

### MTF Analysis

  * **`analyze_mtf_for_series(study_instance_uid: str, series_instance_uid: str)`**: (`Tool`) Main workflow that finds MTF instances, downloads them, and returns a complete, averaged MTF analysis.
  * **`calculate_mtf_from_instances(study_instance_uid: str, series_instance_uid: str, sop_instance_uids: List[str])`**: (`Tool`) Low-level tool that calculates the averaged MTF for an explicit list of instances.

## 🏛️ Architecture

  * **MCP Framework**: [FastMCP](https://gofastmcp.com/)
  * **DIMSE Protocol**: [pynetdicom](https://github.com/pydicom/pynetdicom)
  * **DICOMweb Protocol**: [httpx](https://www.python-httpx.org/)
  * **Configuration**: [PyYAML](https://pyyaml.org/) and [Pydantic](https://docs.pydantic.dev/)
  * **DICOM File Handling**: [pydicom](https://github.com/pydicom/pydicom)

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue to discuss potential changes.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
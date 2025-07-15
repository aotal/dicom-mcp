"""
DICOM configuration using Pydantic.
"""

import yaml
from pathlib import Path
from typing import Dict
from pydantic import BaseModel


class DicomNodeConfig(BaseModel):
    """Configuration for a single DICOM node (e.g., a PACS).

    Attributes:
        host: The hostname or IP address of the DICOM node.
        port: The port number of the DICOM node.
        ae_title: The Application Entity Title (AET) of the DICOM node.
        description: An optional description for the node.
    """
    host: str
    port: int
    ae_title: str
    description: str = ""


class DicomConfiguration(BaseModel):
    """Complete DICOM configuration for the MCP server.

    Attributes:
        nodes: A dictionary of all configured DICOM nodes, keyed by a unique name.
        current_node: The name of the currently active DICOM node for operations.
        calling_aet: The Application Entity Title (AET) that this MCP server will use.
        local_storage_dir: The directory where received DICOM files will be stored.
        dicomweb_url: The base URL for the DICOMweb server.
        dicomweb_timeout: The timeout in seconds for DICOMweb requests.
    """
    nodes: Dict[str, DicomNodeConfig]
    current_node: str
    calling_aet: str
    local_storage_dir: str = "./dicom_received"
    dicomweb_url: str
    dicomweb_timeout: float = 30.0

def load_config(config_path: str) -> DicomConfiguration:
    """Load DICOM configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        A parsed DicomConfiguration object.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    try:
        return DicomConfiguration(**data)
    except Exception as e:
        raise ValueError(f"Invalid configuration in {path}: {str(e)}")
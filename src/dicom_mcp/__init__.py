"""
DICOM Model Context Protocol Server

A Model Context Protocol (MCP) implementation for interacting with DICOM servers.
This package provides a server that can be used to query and retrieve DICOM data
from PACS and other DICOM nodes.
"""

from .server import create_dicom_mcp_server

__version__ = "0.1.0"
__all__ = ["create_dicom_mcp_server"]
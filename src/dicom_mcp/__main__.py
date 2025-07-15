"""
Main entry point for the DICOM MCP Server.

This script parses command-line arguments to start the DICOM MCP server
with a specified configuration and transport protocol.
"""
import argparse
import os
from .server import create_dicom_mcp_server

def main():
    """Parses command-line arguments and runs the DICOM MCP server."""
    parser = argparse.ArgumentParser(description="DICOM MCP Server")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "configuration.yaml"),
        help="Path to the configuration.yaml file",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="The transport protocol to use (stdio or sse)",
    )
    args = parser.parse_args()

    server = create_dicom_mcp_server(args.config_path)
    server.run(transport=args.transport)

if __name__ == "__main__":
    main()
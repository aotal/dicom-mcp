"""
Main entry point for the DICOM MCP Server.
"""
import argparse
import os
from .server import create_dicom_mcp_server

def main():
    """
    Main function to run the DICOM MCP server.
    """
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
        help="The transport protocol to use",
    )
    args = parser.parse_args()

    # Create the server and run it
    server = create_dicom_mcp_server(args.config_path)
    server.run(transport=args.transport)

if __name__ == "__main__":
    main()

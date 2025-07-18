# src/dicom_mcp/__main__.py (Corregido)

import argparse
import os
from .server import create_dicom_mcp_server

def main():
    """Analiza los argumentos de la línea de comandos y ejecuta el servidor DICOM MCP."""
    parser = argparse.ArgumentParser(description="DICOM MCP Server")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "configuration.yaml"),
        help="Ruta al fichero configuration.yaml",
    )
    # --- CORRECCIÓN ---
    # Se añade 'http' a las opciones y se actualiza el help.
    parser.add_argument(
        "--transport",
        default="http",
        choices=["stdio", "sse", "http"], # 'http' añadido como opción válida
        help="El protocolo de transporte a usar (stdio, sse, http)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host para el servidor HTTP")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para el servidor HTTP")
    args = parser.parse_args()

    server = create_dicom_mcp_server(args.config_path)
    
    # Pasamos los argumentos de red al método run, que los delegará a FastMCP
    server.run(transport=args.transport, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
# test_client.py
import asyncio
from fastmcp import Client

# La URL a la que tu servidor está escuchando.
# El path /mcp/ es el por defecto para el transporte Streamable HTTP en fastmcp.
SERVER_URL = "http://127.0.0.1:8000/mcp/"

async def main():
    """
    Se conecta al servidor DICOM MCP y ejecuta algunas herramientas de prueba.
    """
    print(f"🔗 Conectando al servidor MCP en {SERVER_URL}...")
    
    try:
        # Crea una instancia del cliente apuntando a la URL de tu servidor
        client = Client(SERVER_URL)

        # Usa un gestor de contexto para manejar la conexión
        async with client:
            print("✅ Conexión establecida.")

            # 1. Prueba de verificación de conexión (C-ECHO)
            print("\n--- Verificando conexión DICOM (C-ECHO)... ---")
            verify_result = await client.call_tool("verify_connection")
            if not verify_result.is_error:
                print(f"   Resultado: {verify_result.data.message}")
            else:
                print(f"   Error en verify_connection: {verify_result.content}")

            # 2. Prueba para listar los nodos configurados
            print("\n--- Listando nodos DICOM... ---")
            nodes_result = await client.read_resource("resource://dicom_nodes")
            if nodes_result:
                print(f"   Recurso de nodos obtenido: {nodes_result[0].text}")

    except Exception as e:
        print(f"\n❌ Error al conectar o interactuar con el servidor: {e}")

if __name__ == "__main__":
    asyncio.run(main())
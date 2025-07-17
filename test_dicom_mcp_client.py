import pytest
import pydicom
import json  # <-- CORRECCIÓN: Importación añadida para parsear JSON
from unittest.mock import AsyncMock, patch

from fastmcp import Client
from src.dicom_mcp.server import create_dicom_mcp_server
from src.dicom_mcp.models import (
    DicomNodeListResponse,
    OperationStatusResponse,
    FilteredInstanceResultsWrapper,
    MtfSeriesAnalysisResponse,
)

# --- Fixture del Servidor (sin cambios) ---
@pytest.fixture(scope="module")
def mcp_server():
    config_path = "src/dicom_mcp/configuration.yaml"
    server = create_dicom_mcp_server(config_path)
    return server


# --- Tests Corregidos y Robustecidos ---

@pytest.mark.asyncio
async def test_list_dicom_nodes(mcp_server):
    """
    Verifica que el RECURSO 'resource://dicom_nodes' funciona correctamente.
    """
    async with Client(mcp_server) as client:
        result = await client.read_resource("resource://dicom_nodes")
        
        # CORRECCIÓN: Parsear el JSON del atributo .text
        assert result and hasattr(result[0], 'text'), "El recurso no devolvió contenido de texto."
        parsed_json = json.loads(result[0].text)
        response_data = DicomNodeListResponse(**parsed_json)
        
        print("\n✅ test_list_dicom_nodes PASSED")
        print(f"   Nodo Actual: {response_data.current_node}")
        print(f"   Nodos Configurados: {[node.name for node in response_data.nodes]}")
        
        assert response_data.current_node == "dcm4chee"
        assert len(response_data.nodes) > 0


@pytest.mark.asyncio
async def test_switch_dicom_node(mcp_server):
    """
    Prueba la capacidad de cambiar el nodo DICOM activo.
    """
    async with Client(mcp_server) as client:
        new_node = "main"
        result = await client.call_tool("switch_dicom_node", {"node_name": new_node})

        assert not result.is_error
        response_data = result.data
        
        print(f"\n✅ test_switch_dicom_node PASSED")
        print(f"   Mensaje de cambio: {response_data.message}")
        
        # CORRECCIÓN: Verificación de tipo y atributos más robusta
        assert response_data.__class__.__name__ == "OperationStatusResponse"
        assert response_data.success
        assert new_node in response_data.message

        list_result = await client.read_resource("resource://dicom_nodes")
        list_data_json = json.loads(list_result[0].text)
        list_data = DicomNodeListResponse(**list_data_json)
        assert list_data.current_node == new_node


@pytest.mark.asyncio
@patch("src.dicom_mcp.server._internal_qido_query", new_callable=AsyncMock)
async def test_find_mtf_instances(mock_qido_query, mcp_server):
    """
    Prueba 'find_mtf_instances_in_series', simulando una respuesta QIDO.
    """
    mock_qido_query.return_value = [
        {"SOPInstanceUID": "1.2.3.456.1", "InstanceNumber": "1", "ImageComments": "MTF"},
        {"SOPInstanceUID": "1.2.3.456.2", "InstanceNumber": "2", "ImageComments": "OTHER"},
        {"SOPInstanceUID": "1.2.3.456.3", "InstanceNumber": "3", "ImageComments": "MTF"},
    ]

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "find_mtf_instances_in_series",
            {"study_instance_uid": "study123", "series_instance_uid": "series456"}
        )

        assert not result.is_error
        response_data = result.data

        print("\n✅ test_find_mtf_instances PASSED")
        print(f"   Instancias MTF encontradas: {len(response_data.result)}")
        
        # CORRECCIÓN: Verificación de tipo más robusta
        assert response_data.__class__.__name__ == "FilteredInstanceResultsWrapper"
        assert len(response_data.result) == 2
        assert response_data.result[0].SOPInstanceUID == "1.2.3.456.1"
        assert response_data.result[1].ImageComments == "MTF"


@pytest.mark.asyncio
@patch("src.dicom_mcp.server._fetch_dicom_dataset_from_dicomweb", new_callable=AsyncMock)
@patch("src.dicom_mcp.server.process_mtf_from_datasets")
async def test_analyze_mtf_from_instances(mock_process_mtf, mock_fetch_dicom, mcp_server):
    """
    Prueba 'calculate_mtf_from_instances', simulando la descarga de datasets
    y el procesamiento MTF.
    """
    mock_fetch_dicom.return_value = pydicom.Dataset()
    mock_process_mtf.return_value = {
        "status": "OK", "processed_files_count": 2, "valid_vertical_rois": 2,
        "valid_horizontal_rois": 2, "combined_poly_coeffs": [-0.1, 0.2, -0.3, 0.4, 1.0],
        "fit_r_squared": 0.999, "fit_rmse": 0.001, "mtf_at_50_percent": 1.5,
        "mtf_at_10_percent": 2.8, "error_details": None,
    }

    async with Client(mcp_server) as client:
        sop_uids = ["1.2.3.1", "1.2.3.2"]
        result = await client.call_tool(
            "calculate_mtf_from_instances",
            {"study_instance_uid": "study123", "series_instance_uid": "series456", "sop_instance_uids": sop_uids}
        )
        
        assert not result.is_error
        response_data = result.data

        print("\n✅ test_analyze_mtf_from_instances PASSED")
        print(f"   Resultado MTF: R²={response_data.fit_r_squared}, MTF@50%={response_data.mtf_at_50_percent} lp/mm")
        
        # CORRECCIÓN: Verificación de tipo más robusta
        assert response_data.__class__.__name__ == "MtfSeriesAnalysisResponse"
        assert mock_fetch_dicom.call_count == len(sop_uids)
        mock_process_mtf.assert_called_once()
        assert response_data.processed_files_count == 2
        assert response_data.mtf_at_50_percent == 1.5
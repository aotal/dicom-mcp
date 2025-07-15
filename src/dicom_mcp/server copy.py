"""
DICOM MCP Server main implementation.
"""

import logging
import pydicom
import numpy as np
from pydicom.datadict import keyword_for_tag
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Any, AsyncIterator
from .models import PixelDataResponse, DicomNodeInfo, DicomNodeListResponse, OperationStatusResponse, ConnectionVerificationResponse, PatientQueryResult, StudyResponse, SeriesResponse, AttributePresetDetails, AttributePresetsResponse, QidoResponse, PatientQueryResultsWrapper, StudyQueryResultsWrapper, SeriesQueryResultsWrapper, QidoQueryResultsWrapper, ModalityLUTSequenceModel, ModalityLUTSequenceItem, MtfSingleInstanceResponse, MtfSeriesAnalysisResponse, FilteredInstanceResultsWrapper, FilteredInstanceResult
import httpx # Import httpx for DICOMweb operations
from email.message import Message
from email.parser import BytesParser

from fastmcp import FastMCP, Context

from .attributes import ATTRIBUTE_PRESETS
from .dicom_client import DicomClient
from .config import DicomConfiguration, load_config
from .MTF.utils import apply_dicom_linearity
from .mtf_processor_wrapper import process_mtf_from_datasets

@dataclass
class DicomContext:
    """Context for the DICOM MCP server."""
    config: DicomConfiguration
    client: DicomClient

FASTAPI_ATTRIBUTE_SETS = {
    "QC_Convencional": [
        "00180060",  # KVP
        "00181151",  # XRayTubeCurrent
        "00181153",  # ExposureInuAs
        "00204000",  # ImageComments
        "00283000"   # ModalityLUTSequence
    ],
    "InfoPacienteEstudio": [
        "00100010", # PatientName
        "00081030", # StudyDescription
        "00080090"  # ReferringPhysicianName
    ]
}

async def _fetch_dicom_dataset_from_dicomweb(
    study_instance_uid: str,
    series_instance_uid: str,
    sop_instance_uid: str,
    dicom_ctx: DicomContext
) -> pydicom.Dataset:
    """Fetches a single DICOM dataset from a DICOMweb server and returns it as a pydicom.Dataset object."""
    config = dicom_ctx.config
    if not config or not config.dicomweb_url:
        raise ValueError("DICOMweb URL is not configured.")

    dicomweb_url = (
        f"{config.dicomweb_url}/studies/{study_instance_uid}/"
        f"series/{series_instance_uid}/instances/{sop_instance_uid}"
    )
    
    logger.info(f"Fetching DICOM instance from DICOMweb: {dicomweb_url}")

    try:
        async with httpx.AsyncClient() as client:
            # --- MODIFICACIÓN CLAVE AQUÍ ---
            # Usamos el timeout desde la configuración en lugar de un valor fijo.
            response = await client.get(
                dicomweb_url, 
                headers={"Accept": 'multipart/related; type="application/dicom"'},
                timeout=config.dicomweb_timeout 
            )
            # --- FIN DE LA MODIFICACIÓN ---
            response.raise_for_status()

            # Reconstruct headers and body for the parser
            content_type_header = 'Content-Type: ' + response.headers['Content-Type']
            full_response_bytes = content_type_header.encode('utf-8') + b'\r\n\r\n' + response.content
            
            # Parse the multipart response to find the DICOM part
            parser = BytesParser()
            parsed_msg = parser.parsebytes(full_response_bytes)

            dicom_bytes = None
            for part in parsed_msg.walk():
                if part.get_content_type() == 'application/dicom':
                    dicom_bytes = part.get_payload(decode=True)
                    break
            
            if dicom_bytes is None:
                raise ValueError("No 'application/dicom' part found in multipart response.")

            # Read DICOM data from bytes and return the dataset
            return pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes), force=True)

    except httpx.HTTPStatusError as exc:
        logger.error(f"HTTP error fetching from DICOMweb: {exc.response.status_code} - {exc.response.text}")
        raise ValueError(f"HTTP error from DICOMweb: {exc.response.status_code} - {exc.response.text}")
    except httpx.RequestError as exc:
        logger.error(f"Network error fetching from DICOMweb: {exc}")
        raise ConnectionError(f"Network error connecting to DICOMweb: {exc}")
    except Exception as e:
        logger.error(f"Error processing DICOMweb response: {e}", exc_info=True)
        raise

async def _perform_qido_web_query(
    query_level: str,
    query_params: Dict[str, Any],
    config: DicomConfiguration
) -> List[Dict[str, Any]]:
    """
    Función auxiliar para realizar una consulta QIDO-RS.
    """
    if not config.dicomweb_url:
        raise ValueError("DICOMweb URL is not configured. Please set 'dicomweb_url' in configuration.yaml")

    base_url = config.dicomweb_url.rstrip('/') # Ensure no trailing slash

    if query_level.startswith("studies/") or query_level in ["studies", "series", "instances"]:
        qido_url = f"{base_url}/{query_level}"
    else:
        raise ValueError(f"Invalid query_level: {query_level}. Must be 'studies', 'series', 'instances' or follow 'studies/{{uid}}/series' pattern.")

    # Prepare query parameters, expanding includefield if present
    processed_params = query_params.copy() if query_params else {}
    if 'includefield' in processed_params:
        expanded_fields = []
        fields_to_check = processed_params['includefield'].split(',')
        for field in fields_to_check:
            field = field.strip()
            if field in FASTAPI_ATTRIBUTE_SETS:
                expanded_fields.extend(FASTAPI_ATTRIBUTE_SETS[field])
            else:
                expanded_fields.append(field)
        processed_params['includefield'] = ",".join(expanded_fields)

    logger.info(f"Proxying QIDO-RS query to: {qido_url} with params: {processed_params}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                qido_url,
                params=processed_params,
                headers={"Accept": "application/dicom+json"},
                timeout=30.0
            )
            response.raise_for_status()
            return _process_dicom_json_output(response.json())
        except httpx.HTTPStatusError as exc:
            logger.error(f"DICOMweb QIDO-RS HTTP error: {exc.response.status_code} - {exc.response.text}")
            raise ValueError(f"DICOMweb QIDO-RS HTTP error: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Network error during DICOMweb QIDO-RS query: {exc}")
            raise ConnectionError(f"Network error connecting to DICOMweb: {exc}")
        except Exception as e:
            logger.error(f"Error processing DICOMweb response: {e}", exc_info=True)
            raise

def _process_dicom_json_output(json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Procesa la salida JSON de una consulta QIDO-RS para simplificar su estructura.
    Convierte el formato {"tag": {"vr": "...", "Value": [...]}} a {"keyword": "value"}.
    """
    processed_list = []
    if not isinstance(json_data, list):
        # A veces, la respuesta puede no ser una lista si hay un error o no hay resultados
        return []

    for item in json_data:
        processed_item = {}
        for tag, data in item.items():
            keyword = keyword_for_tag(tag) # Convierte (0010,0010) a "PatientName"
            if keyword: # Solo procesa si el tag tiene una palabra clave conocida
                value = data.get("Value", [])
                # Si 'Value' es una lista, toma el primer elemento si es único, o la lista completa.
                # Esto maneja tanto atributos de valor único como múltiple.
                if len(value) == 1:
                    processed_item[keyword] = value[0]
                elif len(value) > 1:
                    processed_item[keyword] = value
                else:
                    processed_item[keyword] = None # O un valor por defecto que prefieras

        processed_list.append(processed_item)
    return processed_list        

# Configure logging
logger = logging.getLogger("dicom_mcp")

# En src/dicom_mcp/server.py, junto a las otras funciones auxiliares

def _create_dataset_from_dicomweb(metadata: List[Dict[str, Any]], pixel_data: bytes) -> pydicom.Dataset:
    """Crea un objeto pydicom.Dataset en memoria a partir de metadatos JSON y píxeles crudos."""
    ds = pydicom.Dataset()
    for tag_data in metadata:
        for tag, values in tag_data.items():
            # pydicom necesita el tag como un objeto Tag, no como string
            pydicom.dataset.Dataset.add_new(ds, tag, values.get('vr'), values.get('Value'))
    
    # Añadir los datos de píxeles
    ds.PixelData = pixel_data
    # Es crucial definir la metainformación para que pydicom pueda interpretar los píxeles
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian # O el que corresponda
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    return ds


async def _fetch_instance_data_from_dicomweb(
    study_instance_uid: str,
    series_instance_uid: str,
    sop_instance_uid: str,
    dicom_ctx: DicomContext
) -> pydicom.Dataset:
    """
    Obtiene los metadatos y los píxeles de una instancia vía DICOMweb y los combina
    en un objeto pydicom.Dataset en memoria.
    """
    config = dicom_ctx.config
    if not config or not config.dicomweb_url:
        raise ValueError("DICOMweb URL no está configurado.")

    base_url = f"{config.dicomweb_url}/studies/{study_instance_uid}/series/{series_instance_uid}/instances/{sop_instance_uid}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Obtener metadatos en formato JSON
        logger.info(f"Fetching metadata from: {base_url}/metadata")
        metadata_response = await client.get(f"{base_url}/metadata", headers={"Accept": "application/dicom+json"})
        metadata_response.raise_for_status()
        metadata_json = metadata_response.json()

        # 2. Obtener los píxeles crudos
        logger.info(f"Fetching pixel data from: {base_url}/rendered")
        # Pedimos los píxeles renderizados en un formato simple (PNG) y luego los leemos
        # o podríamos pedir `application/octet-stream` si el procesador lo soporta.
        # Por simplicidad, este ejemplo asume que tu procesador MTF puede manejar arrays de una imagen renderizada.
        # Si necesita el 'application/octet-stream' original, el cliente debe estar preparado para decodificarlo.
        pixel_response = await client.get(base_url, headers={"Accept": "application/octet-stream"})
        pixel_response.raise_for_status()
        pixel_data = pixel_response.content
        
        # 3. Crear el Dataset en memoria
        # Esta es una simplificación. Crear un dataset desde cero requiere más tags.
        # Nos enfocaremos en pasar los tags que necesita tu procesador MTF.
        ds = pydicom.Dataset()
        for tag_info in metadata_json:
            tag_id = next(iter(tag_info))
            tag_data = tag_info[tag_id]
            vr = tag_data['vr']
            value = tag_data.get('Value', [])
            
            # Convertir el tag string a un objeto pydicom.tag.Tag si es necesario
            tag = pydicom.tag.Tag(tag_id)
            
            # pydicom espera que el valor sea correcto para el VR
            if vr == 'US':
                value = [int(v) for v in value]
            elif vr == 'DS':
                value = [float(v) for v in value]

            ds.add_new(tag, vr, value)

        ds.PixelData = pixel_data
        # Es crucial añadir metainformación para que pydicom sepa cómo leer los píxeles
        ds.file_meta = pydicom.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian # Ajustar si es necesario
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Asegurarse de que los tags necesarios para el procesado MTF estén presentes
        if 'PixelSpacing' not in ds:
             # Añadir un valor por defecto o lanzar un error si es crítico
             logger.warning("PixelSpacing no encontrado en metadatos, usando valor por defecto 1.0")
             ds.PixelSpacing = [1.0, 1.0]
        if 'RescaleSlope' not in ds:
            ds.RescaleSlope = 1
        if 'RescaleIntercept' not in ds:
            ds.RescaleIntercept = 0

        return ds

async def _internal_qido_query(
    query_level: str,
    query_params: Dict[str, Any],
    dicom_ctx: 'DicomContext' # Usamos DicomContext directamente
) -> List[Dict[str, Any]]:
    """
    Función interna para realizar la consulta QIDO-RS. Es llamada por las herramientas.
    """
    config = dicom_ctx.config
    if not config.dicomweb_url:
        raise ValueError("La URL de DICOMweb no está configurada.")

    cleaned_query_level = query_level.strip()
    base_url = config.dicomweb_url.rstrip('/')
    qido_url = f"{base_url}/{cleaned_query_level}"

    processed_params = {}
    if query_params:
        for key, value in query_params.items():
            cleaned_key = key.strip()
            cleaned_value = value.strip() if isinstance(value, str) else value
            processed_params[cleaned_key] = cleaned_value

    if 'includefield' in processed_params:
        expanded_fields = []
        fields_to_check_str = str(processed_params.get('includefield', ''))
        for field in fields_to_check_str.split(','):
            field = field.strip()
            if field in FASTAPI_ATTRIBUTE_SETS:
                expanded_fields.extend(FASTAPI_ATTRIBUTE_SETS[field])
            else:
                expanded_fields.append(field)
        processed_params['includefield'] = ",".join(expanded_fields)

    logger.info(f"Realizando consulta QIDO-RS interna a: {qido_url} con params: {processed_params}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                qido_url,
                params=processed_params,
                headers={"Accept": "application/dicom+json"},
                timeout=config.dicomweb_timeout
            )
            response.raise_for_status()
            return _process_dicom_json_output(response.json())
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"Error HTTP del servidor DICOMweb: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            raise ConnectionError(f"Error de red al conectar con DICOMweb: {exc}")
        

def create_dicom_mcp_server(config_path: str, name: str = "DICOM MCP") -> FastMCP:
    """Create and configure a DICOM MCP server."""
    
    # Define a simple lifespan function
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[DicomContext]:
        # Load config
        config = load_config(config_path)
        
        # Get the current node and calling AE title
        current_node = config.nodes[config.current_node]
        
        # Create client
        client = DicomClient(
            host=current_node.host,
            port=current_node.port,
            calling_aet=config.calling_aet,
            called_aet=current_node.ae_title
        )
        
        logger.info(f"DICOM client initialized: {config.current_node} (calling AE: {config.calling_aet})")
        
        try:
            yield DicomContext(config=config, client=client)
        finally:
            pass
    
    # Create server
    mcp = FastMCP(name, lifespan=lifespan)
    
    # Register tools
    @mcp.resource(uri="resource://dicom_nodes")
    def list_dicom_nodes(ctx: Context = None) -> DicomNodeListResponse:
        """List all configured DICOM nodes and their connection information.
        
        This tool returns information about all configured DICOM nodes in the system
        and shows which node is currently selected for operations. It also provides
        information about available calling AE titles.
        
        Returns:
            Dictionary containing:
            - current_node: The currently selected DICOM node name
            - nodes: List of all configured node names
        
        Example:
            {
                "current_node": "pacs1",
                "nodes": ["pacs1", "pacs2", "orthanc"],
            }
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        
        current_node =  config.current_node
        nodes = [DicomNodeInfo(name=node_name, description=node.description) for node_name, node in config.nodes.items()]

        return DicomNodeListResponse(
            current_node=current_node,
            nodes=nodes,
        )
    

    @mcp.tool
    def switch_dicom_node(node_name: str, ctx: Context = None) -> OperationStatusResponse:
        """Switch the active DICOM node connection to a different configured node.
        
        This tool changes which DICOM node (PACS, workstation, etc.) subsequent operations
        will connect to. The node must be defined in the configuration file.
        
        Args:
            node_name: The name of the node to switch to, must match a name in the configuration
        
        Returns:
            Dictionary containing:
            - success: Boolean indicating if the switch was successful
            - message: Description of the operation result or error
        
        Example:
            {
                "success": true,
                "message": "Switched to DICOM node: orthanc"
            }
        
        Raises:
            ValueError: If the specified node name is not found in configuration
        """        
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        
        # Check if node exists
        if node_name not in config.nodes:
            raise ValueError(f"Node '{node_name}' not found in configuration")
        
        # Update configuration
        config.current_node = node_name
        
        # Create a new client with the updated configuration
        current_node = config.nodes[config.current_node]
        
        # Replace the client with a new instance
        dicom_ctx.client = DicomClient(
            host=current_node.host,
            port=current_node.port,
            calling_aet=config.calling_aet,
            called_aet=current_node.ae_title
        )
        
        return OperationStatusResponse(
            success=True,
            message=f"Switched to DICOM node: {node_name}"
        )

    @mcp.tool
    def verify_connection(ctx: Context = None) -> ConnectionVerificationResponse:
        """Verify connectivity to the current DICOM node using C-ECHO.
        
        This tool performs a DICOM C-ECHO operation (similar to a network ping) to check
        if the currently selected DICOM node is reachable and responds correctly. This is
        useful to troubleshoot connection issues before attempting other operations.
        
        Returns:
            A message describing the connection status, including host, port, and AE titles
        
        Example:
            "Connection successful to 192.168.1.100:104 (Called AE: ORTHANC, Calling AE: CLIENT)"
        """
        dicom_ctx = ctx.request_context.lifespan_context
        client = dicom_ctx.client
        
        success, message = client.verify_connection()
        return ConnectionVerificationResponse(message=message)

    @mcp.tool
    def query_patients(
        name_pattern: str = "", 
        patient_id: str = "", 
        birth_date: str = "", 
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = [],
        exclude_attributes: List[str] = [],
        additional_filters: Dict[str, str] = {},
        ctx: Context = None
    ) -> PatientQueryResultsWrapper:
        """Query patients matching the specified criteria from the DICOM node.
        
        This tool performs a DICOM C-FIND operation at the PATIENT level to find patients
        matching the provided search criteria. All search parameters are optional and can
        be combined for more specific queries.
        
        Args:
            name_pattern: Patient name pattern (can include wildcards * and ?), e.g., "SMITH*"
            patient_id: Patient ID to search for, e.g., "12345678"
            birth_date: Patient birth date in YYYYMMDD format, e.g., "19700101"
            attribute_preset: Controls which attributes to include in results:
                - "minimal": Only essential attributes
                - "standard": Common attributes (default)
                - "extended": All available attributes
            additional_attributes: List of specific DICOM attributes to include beyond the preset
            exclude_attributes: List of DICOM attributes to exclude from the results
            additional_filters: Dictionary of additional DICOM tags to use for filtering, e.g., {"PatientSex": "F"}
        
        Returns:
            List of dictionaries, each representing a matched patient with their attributes
        
        Raises:
            Exception: If there is an error communicating with the DICOM node
        """
        dicom_ctx = ctx.request_context.lifespan_context
        client = dicom_ctx.client
        
        try:
            results = client.query_patient(
                patient_id=patient_id,
                name_pattern=name_pattern,
                birth_date=birth_date,
                attribute_preset=attribute_preset,
                additional_attrs=additional_attributes,
                exclude_attrs=exclude_attributes,
                additional_filters=additional_filters
            )
            return PatientQueryResultsWrapper(result=[PatientQueryResult(**r) for r in results])
        except Exception as e:
            raise Exception(f"Error querying patients: {str(e)}")

    @mcp.tool
    def query_studies(
        patient_id: str = "", 
        study_date: str = "", 
        modality_in_study: str = "",
        study_description: str = "", 
        accession_number: str = "", 
        study_instance_uid: str = "",
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = [],
        exclude_attributes: List[str] = [],
        additional_filters: Dict[str, str] = {},
        ctx: Context = None
    ) -> StudyQueryResultsWrapper:
        """Query studies matching the specified criteria from the DICOM node.
        
        Args:
            patient_id: Patient ID to search for, e.g., "12345678"
            study_date: Study date or date range in DICOM format: "20230101" or "20230101-20230131"
            modality_in_study: Filter by modalities present in study, e.g., "CT" or "MR"
            study_description: Study description text (can include wildcards), e.g., "CHEST*"
            accession_number: Medical record accession number
            study_instance_uid: Unique identifier for a specific study
            attribute_preset: Controls which attributes to include in results
            additional_attributes: List of specific DICOM attributes to include beyond the preset
            exclude_attributes: List of DICOM attributes to exclude from the results
            additional_filters: Dictionary of additional DICOM tags to use for filtering
        
        Returns:
            List of dictionaries, each representing a matched study with its attributes
        
        Raises:
            Exception: If there is an error communicating with the DICOM node
        """
        dicom_ctx = ctx.request_context.lifespan_context
        client = dicom_ctx.client
        
        try:
            results = client.query_study(
                patient_id=patient_id,
                study_date=study_date,
                modality=modality_in_study,
                study_description=study_description,
                accession_number=accession_number,
                study_instance_uid=study_instance_uid,
                attribute_preset=attribute_preset,
                additional_attrs=additional_attributes,
                exclude_attrs=exclude_attributes,
                additional_filters=additional_filters
            )
            return StudyQueryResultsWrapper(result=[StudyResponse(**r) for r in results])
        except Exception as e:
            raise Exception(f"Error querying studies: {str(e)}")

    @mcp.tool
    def query_series(
        study_instance_uid: str, 
        modality: str = "", 
        series_number: str = "",
        series_description: str = "", 
        series_instance_uid: str = "",
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = [],
        exclude_attributes: List[str] = [],
        additional_filters: Dict[str, str] = {},
        ctx: Context = None
    ) -> SeriesQueryResultsWrapper:
        """Query series within a study from the DICOM node.
        
        Args:
            study_instance_uid: Unique identifier for the study (required)
            modality: Filter by imaging modality, e.g., "CT", "MR", "US", "CR"
            series_number: Filter by series number
            series_description: Series description text (can include wildcards), e.g., "AXIAL*"
            series_instance_uid: Unique identifier for a specific series
            attribute_preset: Controls which attributes to include in results
            additional_attributes: List of specific DICOM attributes to include beyond the preset
            exclude_attributes: List of DICOM attributes to exclude from the results
            additional_filters: Dictionary of additional DICOM tags to use for filtering
        
        Returns:
            List of dictionaries, each representing a matched series with its attributes
        
        Raises:
            Exception: If there is an error communicating with the DICOM node
        """
        dicom_ctx = ctx.request_context.lifespan_context
        client = dicom_ctx.client
        
        try:
            results = client.query_series(
                study_instance_uid=study_instance_uid,
                series_instance_uid=series_instance_uid,
                modality=modality,
                series_number=series_number,
                series_description=series_description,
                attribute_preset=attribute_preset,
                additional_attrs=additional_attributes,
                exclude_attrs=exclude_attributes,
                additional_filters=additional_filters
            )
            return SeriesQueryResultsWrapper(result=[SeriesResponse(**r) for r in results])
        except Exception as e:
            raise Exception(f"Error querying series: {str(e)}")

    

    @mcp.resource("dicomweb://studies/{study_instance_uid}/series/{series_instance_uid}/instances/{sop_instance_uid}/pixeldata")
    async def get_dicomweb_pixel_data(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
        ctx: Context
    ) -> PixelDataResponse:
        """
        Retrieves pixel data for a DICOM instance via DICOMweb (WADO-RS).
        """
        dicom_ctx: DicomContext = ctx.request_context.lifespan_context
        
        try:
            # Use the new helper function to fetch the dataset
            ds = await _fetch_dicom_dataset_from_dicomweb(
                study_instance_uid, series_instance_uid, sop_instance_uid, dicom_ctx
            )

            # Extract pixel information from the fetched dataset
            linearized_pixel_array = apply_dicom_linearity(ds)
            pixel_info = _extract_pixel_array_info(ds, linearized_pixel_array)

            # If all went well, return the response object
            return PixelDataResponse(
                sop_instance_uid=sop_instance_uid,
                **pixel_info
            )

        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
            # Re-raise specific, handled exceptions from the helper
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred in get_dicomweb_pixel_data: {e}", exc_info=True)
            raise

    @mcp.resource(uri="resource://attribute_presets")
    def get_attribute_presets() -> AttributePresetsResponse:
        """Get all available attribute presets for DICOM queries.
        
        This tool returns the defined attribute presets that can be used with the
        query_* functions. It shows which DICOM attributes are included in each
        preset (minimal, standard, extended) for each query level.
        
        Returns:
            Dictionary organized by query level (patient, study, series, instance),
            with each level containing the attribute presets and their associated
            DICOM attributes.
        """
        
        presets_by_level = {}
        for level in ["patient", "study", "series", "instance"]:
            presets_by_level[level] = {
                preset: ATTRIBUTE_PRESETS[preset][level]
                for preset in ["minimal", "standard", "extended"]
            }

        return AttributePresetsResponse(
            patient=AttributePresetDetails(**presets_by_level["patient"]),
            study=AttributePresetDetails(**presets_by_level["study"]),
            series=AttributePresetDetails(**presets_by_level["series"]),
            instance=AttributePresetDetails(**presets_by_level["instance"])
        )


# Reemplaza tu herramienta qido_web_query en src/dicom_mcp/server.py con esta:

    @mcp.tool
    async def qido_web_query(
        query_level: str,
        query_params: Dict[str, Any] = None,
        ctx: Context = None
    ) -> QidoQueryResultsWrapper:
        """
        Realiza una consulta QIDO-RS genérica al servidor DICOMweb configurado.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        results = await _internal_qido_query(query_level, query_params, dicom_ctx)
        return QidoQueryResultsWrapper(result=[QidoResponse(**r) for r in results])
    
    @mcp.tool
    async def find_mtf_instances_in_series(
        study_instance_uid: str,
        series_instance_uid: str,
        ctx: Context = None
    ) -> FilteredInstanceResultsWrapper:
        """
        Busca y devuelve SOLO las instancias de una serie que tengan ImageComments='MTF'.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        
        await ctx.info(f"Buscando instancias MTF en la serie {series_instance_uid.strip()}...")

        # 1. Construir la consulta QIDO (sabemos que puede devolver resultados extra)
        query_level = f"studies/{study_instance_uid.strip()}/series/{series_instance_uid.strip()}/instances"
        query_params = {
            "ImageComments": "MTF",
            "includefield": "SOPInstanceUID,InstanceNumber,ImageComments,PatientName,StudyDescription"
        }
        
        # 2. Llamar a la lógica de consulta interna
        all_instances = await _internal_qido_query(query_level, query_params, dicom_ctx)

        # 3. Filtrar los resultados localmente
        mtf_instances = [
            instance for instance in all_instances
            if instance.get("ImageComments") == "MTF"
        ]
        
        if not mtf_instances:
            await ctx.info("No se encontraron instancias que cumplan el criterio después de filtrar.")
        else:
            await ctx.info(f"Filtrado completado. Se encontraron {len(mtf_instances)} instancias MTF.")

        # 4. Devolver la respuesta con la estructura correcta
        return FilteredInstanceResultsWrapper(result=[FilteredInstanceResult(**r) for r in mtf_instances])

    async def _calculate_mtf_for_instances(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uids: List[str],
        ctx: Context
    ) -> MtfSeriesAnalysisResponse:
        """
        Función auxiliar interna que descarga y procesa un lote de instancias.
        Esta función contiene la lógica principal y reporta el progreso para evitar timeouts.
        """
        dicom_ctx: DicomContext = ctx.request_context.lifespan_context
        total_uids = len(sop_instance_uids)

        if total_uids == 0:
            raise ValueError("La lista de SOPInstanceUIDs para analizar no puede estar vacía.")

        # Notificamos al cliente que empezamos
        await ctx.info(f"Iniciando análisis de {total_uids} instancias...")
        await ctx.report_progress(progress=5, total=100, message="Iniciando descarga de grupo...")

        datasets_to_process = []
        for i, sop_uid in enumerate(sop_instance_uids):
            # 1. LIMPIAR ENTRADA (Buena práctica para evitar errores de URL)
            cleaned_sop_uid = sop_uid.strip()
            if not cleaned_sop_uid:
                await ctx.warning(f"Se encontró un SOPInstanceUID vacío en la lista en la posición {i}, será ignorado.")
                continue

            # 2. REPORTAR PROGRESO (¡La clave para evitar el timeout!)
            # Se envía una notificación al cliente en CADA iteración del bucle.
            progress_percent = 10 + (i / total_uids) * 85
            await ctx.report_progress(
                progress=progress_percent,
                total=100,
                message=f"Descargando instancia {i + 1}/{total_uids}"
            )

            try:
                # 3. Descargar la instancia
                ds = await _fetch_dicom_dataset_from_dicomweb(
                    study_instance_uid.strip(), series_instance_uid.strip(), cleaned_sop_uid, dicom_ctx
                )
                datasets_to_process.append(ds)
            except Exception as e:
                await ctx.warning(f"No se pudo descargar o procesar la instancia {cleaned_sop_uid}: {e}")
                # Continuamos con las demás instancias en lugar de detener todo el proceso
                continue
        
        if not datasets_to_process:
            raise ValueError("No se pudo descargar ninguna de las instancias solicitadas.")

        # Notificamos que la fase larga ha terminado
        await ctx.info(f"Descarga completa de {len(datasets_to_process)} instancias. Procesando...")
        await ctx.report_progress(progress=95, total=100, message="Procesando imágenes...")

        # Procesar el lote completo de datasets descargados
        results = process_mtf_from_datasets(datasets_to_process)

        await ctx.report_progress(progress=100, total=100, message="Análisis completado.")
        return MtfSeriesAnalysisResponse(**results)

    @mcp.tool
    async def calculate_mtf_from_instances(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uids: List[str],
        ctx: Context
    ) -> MtfSeriesAnalysisResponse:
        """
        Calcula la MTF promediada para una lista explícita de instancias DICOM.
        """
        # Simplemente llama a la lógica interna y devuelve el resultado.
        return await _calculate_mtf_for_instances(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            sop_instance_uids=sop_instance_uids,
            ctx=ctx
        )
            
    return mcp

    

    

    
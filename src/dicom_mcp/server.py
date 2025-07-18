"""
DICOM MCP Server main implementation.
"""

import logging
import pydicom
import anyio
import numpy as np
from pydicom.datadict import keyword_for_tag
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Any, AsyncIterator
from .models import PixelDataResponse, DicomNodeInfo, DicomNodeListResponse, OperationStatusResponse, ConnectionVerificationResponse, PatientQueryResult, StudyResponse, SeriesResponse, AttributePresetDetails, AttributePresetsResponse, QidoResponse, PatientQueryResultsWrapper, StudyQueryResultsWrapper, SeriesQueryResultsWrapper, QidoQueryResultsWrapper, ModalityLUTSequenceModel, ModalityLUTSequenceItem, MtfSingleInstanceResponse, MtfSeriesAnalysisResponse, FilteredInstanceResultsWrapper, FilteredInstanceResult, NnpsSeriesAnalysisResponse, NnpsGroupResult, NnpsAnalysisResponse
from .models import SeriesClassificationResponse, KermaGroup, ClassifiedInstance
import httpx # Import httpx for DICOMweb operations
from email.message import Message
from email.parser import BytesParser

from fastmcp import FastMCP, Context

from .attributes import ATTRIBUTE_PRESETS
from .dicom_client import DicomClient
from .config import DicomConfiguration, load_config
from .MTF.utils import apply_dicom_linearity
from .mtf_processor_wrapper import process_mtf_from_datasets
from .nnps_processor_wrapper import process_series_for_nnps, process_nnps_for_group
from .classification_wrapper import classify_instances




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
    """Fetches a single DICOM dataset from a DICOMweb server using WADO-RS.

    This function constructs the WADO-RS URL, sends a GET request to retrieve
    the DICOM instance as a multipart response, parses it, and returns a
    pydicom.Dataset object.

    Args:
        study_instance_uid: The Study Instance UID of the desired instance.
        series_instance_uid: The Series Instance UID of the desired instance.
        sop_instance_uid: The SOP Instance UID of the desired instance.
        dicom_ctx: The DICOM context containing configuration.

    Returns:
        A pydicom.Dataset object for the requested DICOM instance.

    Raises:
        ValueError: If the DICOMweb URL is not configured or the response is invalid.
        ConnectionError: If a network error occurs.
    """
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
            response.raise_for_status()

            content_type_header = 'Content-Type: ' + response.headers['Content-Type']
            full_response_bytes = content_type_header.encode('utf-8') + b'\r\n\r\n' + response.content
            
            parser = BytesParser()
            parsed_msg = parser.parsebytes(full_response_bytes)

            dicom_bytes = None
            for part in parsed_msg.walk():
                if part.get_content_type() == 'application/dicom':
                    dicom_bytes = part.get_payload(decode=True)
                    break
            
            if dicom_bytes is None:
                raise ValueError("No 'application/dicom' part found in multipart response.")

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
    """Helper function to perform a QIDO-RS query.

    Args:
        query_level: The DICOM level to query ('studies', 'series', etc.).
        query_params: A dictionary of query parameters.
        config: The DICOM configuration object.

    Returns:
        A list of dictionaries representing the query results.

    Raises:
        ValueError: If the DICOMweb URL is not configured or the query level is invalid.
        ConnectionError: If a network error occurs.
    """
    if not config.dicomweb_url:
        raise ValueError("DICOMweb URL is not configured. Please set 'dicomweb_url' in configuration.yaml")

    base_url = config.dicomweb_url.rstrip('/')

    if query_level.startswith("studies/") or query_level in ["studies", "series", "instances"]:
        qido_url = f"{base_url}/{query_level}"
    else:
        raise ValueError(f"Invalid query_level: {query_level}. Must be 'studies', 'series', 'instances' or follow 'studies/{{uid}}/series' pattern.")

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
    """Processes the JSON output of a QIDO-RS query to simplify its structure.

    Converts the format {"tag": {"vr": "...", "Value": [...]}} to {"keyword": "value"}.

    Args:
        json_data: The raw JSON data from the QIDO-RS response.

    Returns:
        A list of processed dictionaries with keywords as keys.
    """
    processed_list = []
    if not isinstance(json_data, list):
        return []

    for item in json_data:
        processed_item = {}
        for tag, data in item.items():
            keyword = keyword_for_tag(tag)
            if keyword:
                value = data.get("Value", [])
                if len(value) == 1:
                    processed_item[keyword] = value[0]
                elif len(value) > 1:
                    processed_item[keyword] = value
                else:
                    processed_item[keyword] = None

        processed_list.append(processed_item)
    return processed_list        

# Configure logging
logger = logging.getLogger("dicom_mcp")

def _create_dataset_from_dicomweb(metadata: List[Dict[str, Any]], pixel_data: bytes) -> pydicom.Dataset:
    """Creates a pydicom.Dataset object in memory from JSON metadata and raw pixel data.

    Args:
        metadata: A list of dictionaries containing DICOM tag metadata.
        pixel_data: The raw pixel data as bytes.

    Returns:
        A pydicom.Dataset object.
    """
    ds = pydicom.Dataset()
    for tag_data in metadata:
        for tag, values in tag_data.items():
            pydicom.dataset.Dataset.add_new(ds, tag, values.get('vr'), values.get('Value'))
    
    ds.PixelData = pixel_data
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    return ds


async def _fetch_instance_data_from_dicomweb(
    study_instance_uid: str,
    series_instance_uid: str,
    sop_instance_uid: str,
    dicom_ctx: DicomContext
) -> pydicom.Dataset:
    """Fetches metadata and pixel data for an instance via DICOMweb and combines them into a pydicom.Dataset.

    Args:
        study_instance_uid: The Study Instance UID.
        series_instance_uid: The Series Instance UID.
        sop_instance_uid: The SOP Instance UID.
        dicom_ctx: The DICOM context containing the configuration.

    Returns:
        A pydicom.Dataset object created from the fetched data.

    Raises:
        ValueError: If the DICOMweb URL is not configured.
    """
    config = dicom_ctx.config
    if not config or not config.dicomweb_url:
        raise ValueError("DICOMweb URL is not configured.")

    base_url = f"{config.dicomweb_url}/studies/{study_instance_uid}/series/{series_instance_uid}/instances/{sop_instance_uid}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        logger.info(f"Fetching metadata from: {base_url}/metadata")
        metadata_response = await client.get(f"{base_url}/metadata", headers={"Accept": "application/dicom+json"})
        metadata_response.raise_for_status()
        metadata_json = metadata_response.json()

        logger.info(f"Fetching pixel data from: {base_url}/rendered")
        pixel_response = await client.get(base_url, headers={"Accept": "application/octet-stream"})
        pixel_response.raise_for_status()
        pixel_data = pixel_response.content
        
        ds = pydicom.Dataset()
        for tag_info in metadata_json:
            tag_id = next(iter(tag_info))
            tag_data = tag_info[tag_id]
            vr = tag_data['vr']
            value = tag_data.get('Value', [])
            
            tag = pydicom.tag.Tag(tag_id)
            
            if vr == 'US':
                value = [int(v) for v in value]
            elif vr == 'DS':
                value = [float(v) for v in value]

            ds.add_new(tag, vr, value)

        ds.PixelData = pixel_data
        ds.file_meta = pydicom.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        if 'PixelSpacing' not in ds:
             logger.warning("PixelSpacing not found in metadata, using default value 1.0")
             ds.PixelSpacing = [1.0, 1.0]
        if 'RescaleSlope' not in ds:
            ds.RescaleSlope = 1
        if 'RescaleIntercept' not in ds:
            ds.RescaleIntercept = 0

        return ds

async def _internal_qido_query(
    query_level: str,
    query_params: Dict[str, Any],
    dicom_ctx: 'DicomContext'
) -> List[Dict[str, Any]]:
    """Internal function to perform a QIDO-RS query, called by tools.

    Args:
        query_level: The DICOM level to query.
        query_params: A dictionary of query parameters.
        dicom_ctx: The DICOM context.

    Returns:
        A list of dictionaries representing the query results.

    Raises:
        ValueError: If the DICOMweb URL is not configured.
        ConnectionError: If a network error occurs.
    """
    config = dicom_ctx.config
    if not config.dicomweb_url:
        raise ValueError("DICOMweb URL is not configured.")

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

    logger.info(f"Performing internal QIDO-RS query to: {qido_url} with params: {processed_params}")

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
            raise ValueError(f"HTTP error from DICOMweb server: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            raise ConnectionError(f"Network error connecting to DICOMweb: {exc}")
        

def create_dicom_mcp_server(config_path: str, name: str = "DICOM MCP") -> FastMCP:
    """Create and configure a DICOM MCP server.

    Args:
        config_path: Path to the configuration YAML file.
        name: The name of the MCP server.

    Returns:
        An instance of the FastMCP server.
    """
    
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[DicomContext]:
        """Manages the lifecycle of the DICOM server context.

        Args:
            server: The FastMCP server instance.

        Yields:
            A DicomContext object.
        """
        config = load_config(config_path)
        current_node = config.nodes[config.current_node]
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
    
    mcp = FastMCP(name, lifespan=lifespan)
    
    @mcp.resource(uri="resource://dicom_nodes")
    def list_dicom_nodes(ctx: Context = None) -> DicomNodeListResponse:
        """List all configured DICOM nodes and their connection information.
        
        This tool returns information about all configured DICOM nodes in the system
        and shows which node is currently selected for operations. It also provides
        information about available calling AE titles.
        
        Returns:
            A DicomNodeListResponse object containing the current node and a list of all nodes.
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
            node_name: The name of the node to switch to, must match a name in the configuration.
            ctx: The context object provided by FastMCP.
        
        Returns:
            An OperationStatusResponse indicating the result of the switch.
        
        Raises:
            ValueError: If the specified node name is not found in configuration.
        """        
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        
        if node_name not in config.nodes:
            raise ValueError(f"Node '{node_name}' not found in configuration")
        
        config.current_node = node_name
        current_node = config.nodes[config.current_node]
        
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
        
        Args:
            ctx: The context object provided by FastMCP.

        Returns:
            A ConnectionVerificationResponse with a message describing the connection status.
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
            name_pattern: Patient name pattern (can include wildcards * and ?).
            patient_id: Patient ID to search for.
            birth_date: Patient birth date in YYYYMMDD format.
            attribute_preset: Controls which attributes to include in results ('minimal', 'standard', 'extended').
            additional_attributes: List of specific DICOM attributes to include beyond the preset.
            exclude_attributes: List of DICOM attributes to exclude from the results.
            additional_filters: Dictionary of additional DICOM tags to use for filtering.
            ctx: The context object provided by FastMCP.
        
        Returns:
            A PatientQueryResultsWrapper containing a list of matched patients.
        
        Raises:
            Exception: If there is an error communicating with the DICOM node.
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
            patient_id: Patient ID to search for.
            study_date: Study date or date range in DICOM format.
            modality_in_study: Filter by modalities present in the study.
            study_description: Study description text (can include wildcards).
            accession_number: Medical record accession number.
            study_instance_uid: Unique identifier for a specific study.
            attribute_preset: Controls which attributes to include in results.
            additional_attributes: List of specific DICOM attributes to include beyond the preset.
            exclude_attributes: List of DICOM attributes to exclude from the results.
            additional_filters: Dictionary of additional DICOM tags to use for filtering.
            ctx: The context object provided by FastMCP.
        
        Returns:
            A StudyQueryResultsWrapper containing a list of matched studies.
        
        Raises:
            Exception: If there is an error communicating with the DICOM node.
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
            study_instance_uid: Unique identifier for the study (required).
            modality: Filter by imaging modality (e.g., 'CT', 'MR').
            series_number: Filter by series number.
            series_description: Series description text (can include wildcards).
            series_instance_uid: Unique identifier for a specific series.
            attribute_preset: Controls which attributes to include in results.
            additional_attributes: List of specific DICOM attributes to include beyond the preset.
            exclude_attributes: List of DICOM attributes to exclude from the results.
            additional_filters: Dictionary of additional DICOM tags to use for filtering.
            ctx: The context object provided by FastMCP.
        
        Returns:
            A SeriesQueryResultsWrapper containing a list of matched series.
        
        Raises:
            Exception: If there is an error communicating with the DICOM node.
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
        """Retrieves pixel data for a DICOM instance via DICOMweb (WADO-RS).

        Args:
            study_instance_uid: The Study Instance UID.
            series_instance_uid: The Series Instance UID.
            sop_instance_uid: The SOP Instance UID.
            ctx: The context object provided by FastMCP.

        Returns:
            A PixelDataResponse object containing information about the pixel data.

        Raises:
            Exception: If an error occurs during the DICOMweb request.
        """
        dicom_ctx: DicomContext = ctx.request_context.lifespan_context
        
        try:
            ds = await _fetch_dicom_dataset_from_dicomweb(
                study_instance_uid, series_instance_uid, sop_instance_uid, dicom_ctx
            )

            linearized_pixel_array = apply_dicom_linearity(ds)
            pixel_info = _extract_pixel_array_info(ds, linearized_pixel_array)

            return PixelDataResponse(
                sop_instance_uid=sop_instance_uid,
                **pixel_info
            )

        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
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
            An AttributePresetsResponse object containing the presets for each query level.
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

    @mcp.tool
    async def qido_web_query(
        query_level: str,
        query_params: Dict[str, Any] = None,
        ctx: Context = None
    ) -> QidoQueryResultsWrapper:
        """Performs a generic QIDO-RS query to the configured DICOMweb server.

        Args:
            query_level: The DICOM level to query (e.g., 'studies', 'series').
            query_params: A dictionary of query parameters.
            ctx: The context object provided by FastMCP.

        Returns:
            A QidoQueryResultsWrapper containing the results of the query.
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
        """Finds and returns only the instances in a series with ImageComments='MTF'.

        Args:
            study_instance_uid: The Study Instance UID.
            series_instance_uid: The Series Instance UID.
            ctx: The context object provided by FastMCP.

        Returns:
            A FilteredInstanceResultsWrapper containing the MTF instances.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        
        await ctx.info(f"Searching for MTF instances in series {series_instance_uid.strip()}...")

        query_level = f"studies/{study_instance_uid.strip()}/series/{series_instance_uid.strip()}/instances"
        query_params = {
            "ImageComments": "MTF",
            "includefield": "SOPInstanceUID,InstanceNumber,ImageComments,PatientName,StudyDescription"
        }
        
        all_instances = await _internal_qido_query(query_level, query_params, dicom_ctx)

        mtf_instances = [
            instance for instance in all_instances
            if instance.get("ImageComments") == "MTF"
        ]
        
        if not mtf_instances:
            await ctx.info("No instances found matching the criteria after filtering.")
        else:
            await ctx.info(f"Filtering complete. Found {len(mtf_instances)} MTF instances.")

        return FilteredInstanceResultsWrapper(result=[FilteredInstanceResult(**r) for r in mtf_instances])

    async def _calculate_mtf_for_instances(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uids: List[str],
        ctx: Context
    ) -> MtfSeriesAnalysisResponse:
        """Internal helper function that downloads and processes a batch of instances for MTF analysis.

        This function contains the main logic and reports progress to avoid timeouts.

        Args:
            study_instance_uid: The Study Instance UID.
            series_instance_uid: The Series Instance UID.
            sop_instance_uids: A list of SOP Instance UIDs to process.
            ctx: The context object provided by FastMCP.

        Returns:
            An MtfSeriesAnalysisResponse with the results of the analysis.

        Raises:
            ValueError: If the list of SOPInstanceUIDs is empty or no instances could be downloaded.
        """
        dicom_ctx: DicomContext = ctx.request_context.lifespan_context
        total_uids = len(sop_instance_uids)

        if total_uids == 0:
            raise ValueError("The list of SOPInstanceUIDs to analyze cannot be empty.")

        await ctx.info(f"Starting analysis of {total_uids} instances...")
        await ctx.report_progress(progress=5, total=100, message="Starting batch download...")

        datasets_to_process = []
        for i, sop_uid in enumerate(sop_instance_uids):
            cleaned_sop_uid = sop_uid.strip()
            if not cleaned_sop_uid:
                await ctx.warning(f"Found an empty SOPInstanceUID, it will be ignored.")
                continue

            progress_percent = 10 + (i / total_uids) * 85
            await ctx.report_progress(
                progress=progress_percent,
                total=100,
                message=f"Downloading instance {i + 1}/{total_uids}"
            )
            
            try:
                ds = await _fetch_dicom_dataset_from_dicomweb(
                    study_instance_uid.strip(), series_instance_uid.strip(), cleaned_sop_uid, dicom_ctx
                )
                datasets_to_process.append(ds)
            except Exception as e:
                await ctx.warning(f"Could not download or process instance {cleaned_sop_uid}: {e}")
                continue
        
        if not datasets_to_process:
            raise ValueError("Could not download any of the requested instances.")

        await ctx.info(f"Download complete for {len(datasets_to_process)} instances. Processing...")
        await ctx.report_progress(progress=95, total=100, message="Processing images...")

        results = process_mtf_from_datasets(datasets_to_process)

        await ctx.report_progress(progress=100, total=100, message="Analysis complete.")
        return MtfSeriesAnalysisResponse(**results)

    @mcp.tool
    async def calculate_mtf_from_instances(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uids: List[str],
        ctx: Context
    ) -> MtfSeriesAnalysisResponse:
        """Calculates the averaged MTF for an explicit list of DICOM instances.

        Args:
            study_instance_uid: The Study Instance UID.
            series_instance_uid: The Series Instance UID.
            sop_instance_uids: A list of SOP Instance UIDs to analyze.
            ctx: The context object provided by FastMCP.

        Returns:
            An MtfSeriesAnalysisResponse with the results of the analysis.
        """
        return await _calculate_mtf_for_instances(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            sop_instance_uids=sop_instance_uids,
            ctx=ctx
        )

    @mcp.tool
    async def analyze_mtf_for_series(
        study_instance_uid: str,
        series_instance_uid: str,
        ctx: Context
    ) -> MtfSeriesAnalysisResponse:
        """High-level tool that finds MTF instances in a series and calculates their averaged MTF in one step.

        Args:
            study_instance_uid: The Study Instance UID.
            series_instance_uid: The Series Instance UID.
            ctx: The context object provided by FastMCP.

        Returns:
            An MtfSeriesAnalysisResponse with the results of the analysis or an error status.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        clean_study_uid = study_instance_uid.strip()
        clean_series_uid = series_instance_uid.strip()

        try:
            await ctx.info(f"Searching for MTF instances in series {clean_series_uid}...")
            await ctx.report_progress(progress=5, total=100, message="Searching for instances...")

            qido_level = f"studies/{clean_study_uid}/series/{clean_series_uid}/instances"
            query_params = {"ImageComments": "MTF", "includefield": "SOPInstanceUID"}
            
            all_instances = await _internal_qido_query(qido_level, query_params, dicom_ctx)
            
            mtf_instances_uids = [
                instance.get('SOPInstanceUID') for instance in all_instances
                if instance.get("ImageComments") == "MTF" and instance.get('SOPInstanceUID')
            ]
            
            if not mtf_instances_uids:
                raise ValueError("No instances with ImageComments='MTF' were found in the specified series.")

            await ctx.info(f"Found {len(mtf_instances_uids)} instances. Delegating MTF analysis...")
            
            return await _calculate_mtf_for_instances(
                study_instance_uid=clean_study_uid,
                series_instance_uid=clean_series_uid,
                sop_instance_uids=mtf_instances_uids,
                ctx=ctx
            )

        except Exception as e:
            error_message = f"Failed in MTF analysis workflow for series {clean_series_uid}: {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error(f"Failed to analyze series: {e}")
            
            return MtfSeriesAnalysisResponse(
                status="Error",
                processed_files_count=0,
                valid_vertical_rois=0,
                valid_horizontal_rois=0,
                error_details=f"Error analyzing series: {e}",
                combined_poly_coeffs=None,
                fit_r_squared=None,
                fit_rmse=None,
                mtf_at_50_percent=None
            ) 
    @mcp.tool
    async def calculate_nnps_from_instances(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uids: List[str],
        ctx: Context
    ) -> NnpsAnalysisResponse:
        """
        Calculates the NNPS for an explicit list of instances, treating them as a single group.
        Requires at least 2 instances.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        try:
            if len(sop_instance_uids) < 2:
                raise ValueError("At least 2 instances are required for NNPS group analysis.")

            await ctx.info(f"Starting NNPS analysis for {len(sop_instance_uids)} instances...")
            
            datasets_to_process = []
            for i, sop_uid in enumerate(sop_instance_uids):
                progress = 10 + (i / len(sop_instance_uids)) * 90
                await ctx.report_progress(progress=progress, total=100, message=f"Downloading instance {i+1}/{len(sop_instance_uids)}")
                
                ds = await _fetch_dicom_dataset_from_dicomweb(
                    study_instance_uid.strip(), series_instance_uid.strip(), sop_uid.strip(), dicom_ctx
                )
                datasets_to_process.append(ds)

            await ctx.info("Processing data for NNPS...")
            
            results = await anyio.to_thread.run_sync(process_nnps_for_group, datasets_to_process)
            
            return NnpsAnalysisResponse(**results)

        except Exception as e:
            logger.error(f"Failed in NNPS analysis: {e}", exc_info=True)
            await ctx.error(f"Failed in NNPS analysis: {e}")
            return NnpsAnalysisResponse(status="Error", error_details=str(e))        
 

    @mcp.tool
    async def classify_instances_in_series(
        study_instance_uid: str,
        series_instance_uid: str,
        ctx: Context
    ) -> SeriesClassificationResponse:
        """
        Finds all instances in a series, downloads them, and classifies them by
        ImageComments. FDT images are further grouped by their calculated Kerma value.
        This tool is useful for discovering which instances are available for MTF or NNPS analysis.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        clean_study_uid = study_instance_uid.strip()
        clean_series_uid = series_instance_uid.strip()

        try:
            # 1. Find all relevant instances in the series
            await ctx.info(f"Searching for all instances in series {clean_series_uid}...")
            await ctx.report_progress(progress=5, total=100, message="Searching for instances...")

            qido_level = f"studies/{clean_study_uid}/series/{clean_series_uid}/instances"
            query_params = {"includefield": "SOPInstanceUID,InstanceNumber,ImageComments"}
            
            all_instances_meta = await _internal_qido_query(qido_level, query_params, dicom_ctx)
            
            if not all_instances_meta:
                raise ValueError("No instances found in the specified series.")

            sop_uids_to_download = [inst.get('SOPInstanceUID') for inst in all_instances_meta if inst.get('SOPInstanceUID')]

            # 2. Download all instances, reporting progress
            await ctx.info(f"Found {len(sop_uids_to_download)} total instances. Downloading...")
            
            datasets_to_process = []
            for i, sop_uid in enumerate(sop_uids_to_download):
                progress = 10 + (i / len(sop_uids_to_download)) * 85
                await ctx.report_progress(progress=progress, total=100, message=f"Downloading instance {i+1}/{len(sop_uids_to_download)}")
                
                ds = await _fetch_dicom_dataset_from_dicomweb(
                    clean_study_uid, clean_series_uid, sop_uid.strip(), dicom_ctx
                )
                datasets_to_process.append(ds)

            # 3. Call the wrapper to classify, running it in a separate thread
            await ctx.info("Download complete. Classifying instances...")
            await ctx.report_progress(progress=95, total=100, message="Classifying...")

            classification_results = await anyio.to_thread.run_sync(classify_instances, datasets_to_process)
            
            return SeriesClassificationResponse(
                status="OK",
                mtf_instances=[ClassifiedInstance(**i) for i in classification_results.get("mtf_instances", [])],
                tor_instances=[ClassifiedInstance(**i) for i in classification_results.get("tor_instances", [])],
                fdt_kerma_groups=[KermaGroup(**g) for g in classification_results.get("fdt_kerma_groups", [])],
                other_instances=[ClassifiedInstance(**i) for i in classification_results.get("other_instances", [])]
            )

        except Exception as e:
            error_message = f"Failed during instance classification for series {clean_series_uid}: {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error(error_message)
            return SeriesClassificationResponse(status="Error", error_details=str(e))
                     
    return mcp
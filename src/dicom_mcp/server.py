"""
DICOM MCP Server main implementation.
"""

import logging
import pydicom
from pydicom.datadict import keyword_for_tag
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Any, AsyncIterator
from .models import PixelDataResponse, DicomNodeInfo, DicomNodeListResponse, OperationStatusResponse, ConnectionVerificationResponse, PatientQueryResult, StudyResponse, SeriesResponse, AttributePresetDetails, AttributePresetsResponse, QidoResponse, PatientQueryResultsWrapper, StudyQueryResultsWrapper, SeriesQueryResultsWrapper, QidoQueryResultsWrapper, ModalityLUTSequenceModel, ModalityLUTSequenceItem
import httpx # Import httpx for DICOMweb operations
from email.message import Message
from email.parser import BytesParser

from fastmcp import FastMCP, Context

from .attributes import ATTRIBUTE_PRESETS
from .dicom_client import DicomClient
from .config import DicomConfiguration, load_config

# Configure logging
logger = logging.getLogger("dicom_mcp")


@dataclass
class DicomContext:
    """Context for the DICOM MCP server."""
    config: DicomConfiguration
    client: DicomClient

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
            pixel_info = _extract_pixel_array_info(ds)

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

    @mcp.tool
    async def qido_web_query(
        query_level: str, # e.g., "studies", "series", "instances"
        query_params: Dict[str, Any] = None,
        ctx: Context = None
    ) -> QidoQueryResultsWrapper:
        """Performs a QIDO-RS query to the configured DICOMweb server.

        This tool allows querying studies, series, or instances on the DICOMweb server
        using QIDO-RS. It supports expanding attribute sets (includefield)
        using predefined attribute sets.

        Args:
            query_level: The query level (e.g., "studies", "series", "instances").
                         Can also include UIDs like "studies/{study_uid}/series".
            query_params: A dictionary of query parameters for the QIDO-RS request.
                          Supports expanding 'includefield' using predefined attribute sets.

        Returns:
            A list of dictionaries, each representing a matching DICOM resource.

        Raises:
            ValueError: If the DICOMweb URL is not configured or query_level is invalid.
            httpx.HTTPStatusError: If the DICOMweb server returns an HTTP error.
            httpx.RequestError: If there is a network error connecting to the DICOMweb server.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config

        if not config.dicomweb_url:
            raise ValueError("DICOMweb URL is not configured. Please set 'dicomweb_url' in configuration.yaml")

        base_url = config.dicomweb_url.rstrip('/') # Ensure no trailing slash

        # Construct the full QIDO-RS URL
        # Handle cases like "studies" or "studies/{study_uid}/series"
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
                return QidoQueryResultsWrapper(result=[QidoResponse(**r) for r in _process_dicom_json_output(response.json())])
            except httpx.HTTPStatusError as exc:
                logger.error(f"DICOMweb QIDO-RS HTTP error: {exc.response.status_code} - {exc.response.text}")
                raise ValueError(f"DICOMweb QIDO-RS HTTP error: {exc.response.status_code} - {exc.response.text}")
            except httpx.RequestError as exc:
                logger.error(f"Network error during DICOMweb QIDO-RS query: {exc}")
                raise ConnectionError(f"DICOMweb QIDO-RS network error: {exc}")

    def _process_dicom_json_output(dicom_json_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of DICOM JSON objects (QIDO-RS output) to make it more readable.

        Transforms hexadecimal DICOM tags into readable attribute names and extracts
        'Value' list elements when appropriate.

        Args:
            dicom_json_list: A list of dictionaries, where each dictionary represents
                             a DICOM object (study, series, instance) with its attributes
                             in JSON format (e.g., {"00100010": {"vr": "PN", "Value": ["DOE^JOHN"]}}).

        Returns:
            A list of dictionaries, where each dictionary represents the DICOM object
            with attributes renamed to their DICOM keywords and simplified values.
        """
        processed_results = []
        for dicom_object in dicom_json_list:
            processed_object = {}
            for tag_hex, details in dicom_object.items():
                try:
                    # Convert the hexadecimal tag to a pydicom Tag object
                    tag = pydicom.tag.Tag(tag_hex)
                    # Get the keyword (name) of the attribute
                    keyword = keyword_for_tag(tag)
                    
                    # If the keyword exists, use it; otherwise, use the hexadecimal tag
                    display_key = keyword if keyword else tag_hex

                    # Handle sequences (VR = SQ)
                    if details.get("vr") == "SQ":
                        sequence_items = []
                        if "Value" in details and isinstance(details["Value"], list):
                            for item_dict in details["Value"]:
                                # Special handling for ModalityLUTSequence (0028,3000)
                                if tag_hex == "00283000":
                                    # Process each item in the sequence as ModalityLUTSequenceItem
                                    processed_item = {}
                                    for sub_tag_hex, sub_details in item_dict.items():
                                        sub_tag = pydicom.tag.Tag(sub_tag_hex)
                                        sub_keyword = keyword_for_tag(sub_tag)
                                        sub_display_key = sub_keyword if sub_keyword else sub_tag_hex
                                        sub_value = sub_details.get("Value")
                                        if isinstance(sub_value, list) and len(sub_value) == 1:
                                            processed_item[sub_display_key] = sub_value[0]
                                        else:
                                            processed_item[sub_display_key] = sub_value
                                    sequence_items.append(ModalityLUTSequenceItem(**processed_item))
                                else:
                                    # Recursive call to process each item in the sequence
                                    sequence_items.append(_process_dicom_json_output([item_dict])[0])
                        processed_object[display_key] = sequence_items
                    else:
                        # Extract the value. If 'Value' is a list with a single element, take that element.
                        # If it's a list with multiple elements, keep the list.
                        # If there is no 'Value' or it is null, use None.
                        value = details.get("Value")
                        if isinstance(value, list):
                            if len(value) == 1:
                                processed_object[display_key] = value[0]
                            else:
                                processed_object[display_key] = value
                        else:
                            processed_object[display_key] = value
                        
                except Exception as e:
                    # In case of error (e.g., invalid tag_hex), keep the original tag and raw value
                    processed_object[tag_hex] = details
                    logger.warning(f"Could not process DICOM tag {tag_hex}: {e}")
            if "NumberOfFrames" in processed_object:
                del processed_object["NumberOfFrames"]
            processed_results.append(processed_object)
        return processed_results

    def _extract_pixel_array_info(ds: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extracts the pixel array and relevant information from the DICOM dataset.
        """
        if not hasattr(ds, 'PixelData') or ds.PixelData is None:
            raise ValueError("The DICOM instance does not contain pixel data.")
        
        pixel_array = ds.pixel_array
        
        preview = None
        if pixel_array.ndim >= 2 and pixel_array.size > 0:
            if pixel_array.ndim == 2:
                rows_preview = min(pixel_array.shape[0], 5)
                cols_preview = min(pixel_array.shape[1], 5)
                preview = pixel_array[:rows_preview, :cols_preview].tolist()
            elif pixel_array.ndim == 3:
                if ds.get("SamplesPerPixel", 1) == 1:
                     rows_preview = min(pixel_array.shape[1], 5)
                     cols_preview = min(pixel_array.shape[2], 5)
                     preview = pixel_array[0, :rows_preview, :cols_preview].tolist()
                elif ds.get("SamplesPerPixel", 1) > 1 and pixel_array.shape[-1] == ds.SamplesPerPixel:
                     rows_preview = min(pixel_array.shape[0], 5)
                     cols_preview = min(pixel_array.shape[1], 5)
                     preview = pixel_array[:rows_preview, :cols_preview, 0].tolist()

        return {
            "rows": ds.Rows,
            "columns": ds.Columns,
            "pixel_array_shape": list(pixel_array.shape),
            "pixel_array_dtype": str(pixel_array.dtype),
            "pixel_array_preview": preview,
            "message": "Pixel data extracted. Preview shown."
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
                response = await client.get(
                    dicomweb_url, 
                    headers={"Accept": 'multipart/related; type="application/dicom"'},
                    timeout=30.0
                )
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

    @mcp.prompt
    def simple_test_prompt(message: str) -> str:
        """A simple test prompt that echoes the input message."""
        return f"Prompt received: {message}"

    @mcp.prompt
    def explain_dicom_attribute(attribute_name: str, attribute_value: Any) -> str:
        """
        Generates a prompt for an LLM to explain a specific DICOM attribute and its value.

        Args:
            attribute_name: The name of the DICOM attribute (e.g., "PatientName", "StudyDate").
            attribute_value: The value of the DICOM attribute.

        Returns:
            A formatted string ready to be sent to a language model.
        """
        prompt = f"""
You are a DICOM expert. Explain the following DICOM attribute and its value in simple terms.
Describe what this attribute means in the context of a medical image or study.

DICOM Attribute Name: {attribute_name}
Attribute Value: {attribute_value}

Please provide a clear and easy-to-understand explanation.
"""
        return prompt

    return mcp
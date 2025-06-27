"""
DICOM MCP Server main implementation.
"""

import logging
import threading
import pydicom
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Any, AsyncIterator
import httpx # Import httpx for DICOMweb operations

from mcp.server.fastmcp import FastMCP, Context

from .attributes import ATTRIBUTE_PRESETS
from .dicom_client import DicomClient
from .config import DicomConfiguration, load_config
from .dicom_scp import start_scp_server

# Configure logging
logger = logging.getLogger("dicom_mcp")


@dataclass
class DicomContext:
    """Context for the DICOM MCP server."""
    config: DicomConfiguration
    client: DicomClient

scp_thread: threading.Thread = None
scp_server_instance = None

def create_dicom_mcp_server(config_path: str, name: str = "DICOM MCP") -> FastMCP:
    """Create and configure a DICOM MCP server."""
    
    # Define a simple lifespan function
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[DicomContext]:
        global scp_thread, scp_server_instance
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
        
        # Start SCP server in a separate thread
        def server_callback(server_instance):            global scp_server_instance
            scp_server_instance = server_instance

        scp_thread = threading.Thread(
            target=start_scp_server,
            args=(config, server_callback,),
            daemon=True
        )
        scp_thread.start()
        
        try:
            yield DicomContext(config=config, client=client)
        finally:
            logger.info("Shutting down DICOM MCP server...")
            if scp_server_instance:
                logger.info("Requesting SCP server shutdown...")
                scp_server_instance.shutdown()
            
            if scp_thread and scp_thread.is_alive():
                logger.info("Waiting for SCP thread to finish...")
                scp_thread.join(timeout=5.0)
                if scp_thread.is_alive():
                    logger.warning("Warning: SCP server thread did not terminate cleanly.")
            logger.info("Shutdown complete.")
    
    # Create server
    mcp = FastMCP(name, lifespan=lifespan)
    
    # Register tools
    @mcp.tool()
    def list_dicom_nodes(ctx: Context = None) -> Dict[str, Any]:
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
        nodes = [{node_name: node.description} for node_name, node in config.nodes.items()]

        return {
            "current_node": current_node,
            "nodes": nodes,
        }
    

    @mcp.tool()
    def switch_dicom_node(node_name: str, ctx: Context = None) -> Dict[str, Any]:
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
        
        return {
            "success": True,
            "message": f"Switched to DICOM node: {node_name}"
        }

    @mcp.tool()
    def verify_connection(ctx: Context = None) -> str:
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
        return message

    @mcp.tool()
    def query_patients(
        name_pattern: str = "", 
        patient_id: str = "", 
        birth_date: str = "", 
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = None,
        exclude_attributes: List[str] = None,
        additional_filters: Dict[str, str] = None,
        ctx: Context = None
    ) -> List[Dict[str, Any]]:
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
        
        Example:
            [
                {
                    "PatientID": "12345",
                    "PatientName": "SMITH^JOHN",
                    "PatientBirthDate": "19700101",
                    "PatientSex": "M"
                }
            ]
        
        Raises:
            Exception: If there is an error communicating with the DICOM node
        """
        dicom_ctx = ctx.request_context.lifespan_context
        client = dicom_ctx.client
        
        try:
            return client.query_patient(
                patient_id=patient_id,
                name_pattern=name_pattern,
                birth_date=birth_date,
                attribute_preset=attribute_preset,
                additional_attrs=additional_attributes,
                exclude_attrs=exclude_attributes,
                additional_filters=additional_filters
            )
        except Exception as e:
            raise Exception(f"Error querying patients: {str(e)}")

    @mcp.tool()
    def query_studies(
        patient_id: str = "", 
        study_date: str = "", 
        modality_in_study: str = "",
        study_description: str = "", 
        accession_number: str = "", 
        study_instance_uid: str = "",
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = None,
        exclude_attributes: List[str] = None,
        additional_filters: Dict[str, str] = None,
        ctx: Context = None
    ) -> List[Dict[str, Any]]:
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
            return client.query_study(
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
        except Exception as e:
            raise Exception(f"Error querying studies: {str(e)}")

    @mcp.tool()
    def query_series(
        study_instance_uid: str, 
        modality: str = "", 
        series_number: str = "",
        series_description: str = "", 
        series_instance_uid: str = "",
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = None,
        exclude_attributes: List[str] = None,
        additional_filters: Dict[str, str] = None,
        ctx: Context = None
    ) -> List[Dict[str, Any]]:
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
            return client.query_series(
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
        except Exception as e:
            raise Exception(f"Error querying series: {str(e)}")

    @mcp.tool()
    def query_instances(
        series_instance_uid: str, 
        instance_number: str = "", 
        sop_instance_uid: str = "",
        attribute_preset: str = "standard", 
        additional_attributes: List[str] = None,
        exclude_attributes: List[str] = None,
        additional_filters: Dict[str, str] = None,
        ctx: Context = None 
    ) -> List[Dict[str, Any]]:
        """Query individual DICOM instances (images) within a series.
        
        Args:
            series_instance_uid: Unique identifier for the series (required)
            instance_number: Filter by specific instance number within the series
            sop_instance_uid: Unique identifier for a specific instance
            attribute_preset: Controls which attributes to include in results
            additional_attributes: List of specific DICOM attributes to include beyond the preset
            exclude_attributes: List of DICOM attributes to exclude from the results
            additional_filters: Dictionary of additional DICOM tags to use for filtering
        
        Returns:
            List of dictionaries, each representing a matched instance with its attributes
        
        Raises:
            Exception: If there is an error communicating with the DICOM node
        """
        dicom_ctx = ctx.request_context.lifespan_context
        client = dicom_ctx.client
        
        try:
            return client.query_instance(
                series_instance_uid=series_instance_uid,
                sop_instance_uid=sop_instance_uid,
                instance_number=instance_number,
                attribute_preset=attribute_preset,
                additional_attrs=additional_attributes,
                exclude_attrs=exclude_attributes,
                additional_filters=additional_filters
            )
        except Exception as e:
            raise Exception(f"Error querying instances: {str(e)}")
        
    @mcp.tool()
    def move_series(
        destination_node: str,
        series_instance_uid: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Move a DICOM series to another DICOM node.
        
        Args:
            destination_node: Name of the destination node as defined in the configuration
            series_instance_uid: The unique identifier for the series to be moved
        
        Returns:
            Dictionary containing:            - success: Boolean indicating if the operation was successful
            - message: Description of the operation result or error
            - completed: Number of successfully transferred instances
            - failed: Number of failed transfers
            - warning: Number of transfers with warnings
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        client = dicom_ctx.client
        
        if destination_node not in config.nodes:
            raise ValueError(f"Destination node '{destination_node}' not found in configuration")
        
        destination_ae = config.nodes[destination_node].ae_title
        
        result = client.move_series(
            destination_ae=destination_ae,
            series_instance_uid=series_instance_uid
        )
        
        return result

    @mcp.tool()
    def move_study(
        destination_node: str,
        study_instance_uid: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Move a DICOM study to another DICOM node.
        
        Args:
            destination_node: Name of the destination node as defined in the configuration
            study_instance_uid: The unique identifier for the study to be moved
        
        Returns:
            Dictionary containing:
            - success: Boolean indicating if the operation was successful
            - message: Description of the operation result or error
            - completed: Number of successfully transferred instances
            - failed: Number of failed transfers
            - warning: Number of transfers with warnings
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        client = dicom_ctx.client
        
        if destination_node not in config.nodes:
            raise ValueError(f"Destination node '{destination_node}' not found in configuration")
        
        destination_ae = config.nodes[destination_node].ae_title
        
        result = client.move_study(
            destination_ae=destination_ae,
            study_instance_uid=study_instance_uid
        )
        
        return result

    @mcp.tool()
    def retrieve_series_to_local(
        series_instance_uid: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Retrieve a DICOM series to the local server storage.
        
        This tool initiates a C-MOVE operation to retrieve a specific series
        from the currently active DICOM node to the local storage directory
        configured for this MCP server.
        
        Args:
            series_instance_uid: The unique identifier for the series to be retrieved.
        
        Returns:
            Dictionary containing the status of the C-MOVE operation.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        client = dicom_ctx.client
        
        local_scp_node = config.nodes.get('local_scp')
        if not local_scp_node:
            raise ValueError("'local_scp' node not found or properly configured in configuration file.")
        
        result = client.move_series(
            destination_ae=local_scp_node.ae_title,
            series_instance_uid=series_instance_uid
        )
        
        return result

    @mcp.tool()
    def retrieve_study_to_local(
        study_instance_uid: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Retrieve a DICOM study to the local server storage.
        
        This tool initiates a C-MOVE operation to retrieve an entire study
        from the currently active DICOM node to the local storage directory
        configured for this MCP server.
        
        Args:
            study_instance_uid: The unique identifier for the study to be retrieved.
        
        Returns:
            Dictionary containing the status of the C-MOVE operation.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        client = dicom_ctx.client
        
        local_scp_node = config.nodes.get('local_scp')
        if not local_scp_node:
            raise ValueError("'local_scp' node not found or properly configured in configuration file.")
        
        result = client.move_study(
            destination_ae=local_scp_node.ae_title,
            study_instance_uid=study_instance_uid
        )
        
        return result

    @mcp.tool()
    def get_local_instance_pixel_data(
        sop_instance_uid: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Gets pixel data from a locally stored DICOM instance.
        
        This tool retrieves pixel data metadata from a DICOM file that has been
        previously retrieved and stored locally via a C-MOVE operation.
        
        Args:
            sop_instance_uid: The SOP Instance UID of the instance to retrieve.
            
        Returns:
            A dictionary containing pixel data metadata.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config
        
        filepath = Path(config.local_storage_dir) / (sop_instance_uid + ".dcm")
        
        if not filepath.is_file():
            raise FileNotFoundError(f"DICOM file not found locally at {filepath}")
        
        try:
            ds = pydicom.dcmread(str(filepath), force=True)
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
                "sop_instance_uid": sop_instance_uid,
                "rows": ds.Rows,
                "columns": ds.Columns,
                "pixel_array_shape": list(pixel_array.shape),
                "pixel_array_dtype": str(pixel_array.dtype),
                "pixel_array_preview": preview,
                "message": "Pixel data accessed from locally stored file."
            }
        except Exception as e:
            logger.error(f"Error processing local DICOM file {filepath}: {e}", exc_info=True)
            raise

    @mcp.tool()
    def get_dicomweb_pixel_data(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Retrieves pixel data from a DICOM instance via DICOMweb (WADO-RS).
        
        This tool constructs a WADO-RS URL and fetches the pixel data for a specific
        DICOM instance directly from the configured DICOMweb server.
        
        Args:
            study_instance_uid: The Study Instance UID of the instance.
            series_instance_uid: The Series Instance UID of the instance.
            sop_instance_uid: The SOP Instance UID of the instance.
            
        Returns:
            A dictionary containing pixel data metadata and a preview.
        """
        dicom_ctx = ctx.request_context.lifespan_context
        config = dicom_ctx.config

        if not config.dicomweb_url:
            raise ValueError("DICOMweb URL is not configured. Please set 'dicomweb_url' in configuration.yaml")

        # Construct WADO-RS URL for pixel data
        # Example: /studies/{study}/series/{series}/instances/{instance}/frames/{frame}
        # For single-frame images, frame is usually 1
        dicomweb_url = (
            f"{config.dicomweb_url}/studies/{study_instance_uid}/"
            f"series/{series_instance_uid}/instances/{sop_instance_uid}/pixeldata"
        )
        
        logger.info(f"Fetching pixel data from DICOMweb: {dicomweb_url}")

        try:
            # Use httpx to make the request
            response = httpx.get(dicomweb_url, headers={"Accept": "application/dicom"})
            response.raise_for_status() # Raise an exception for HTTP errors

            # The response content is the DICOM file itself
            dicom_bytes = response.content
            
            # Use pydicom to read the DICOM data from bytes
            ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes), force=True)

            if not hasattr(ds, 'PixelData') or ds.PixelData is None:
                raise ValueError("The DICOM instance retrieved via DICOMweb does not contain pixel data.")
            
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
                "sop_instance_uid": sop_instance_uid,
                "rows": ds.Rows,
                "columns": ds.Columns,
                "pixel_array_shape": list(pixel_array.shape),
                "pixel_array_dtype": str(pixel_array.dtype),
                "pixel_array_preview": preview,
                "message": "Pixel data accessed via DICOMweb."
            }
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error fetching DICOMweb pixel data: {exc.response.status_code} - {exc.response.text}")
            raise ValueError(f"DICOMweb HTTP error: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Network error fetching DICOMweb pixel data: {exc}")
            raise ConnectionError(f"DICOMweb network error: {exc}")
        except Exception as e:
            logger.error(f"Error processing DICOMweb pixel data: {e}", exc_info=True)
            raise

    @mcp.tool()
    def get_attribute_presets() -> Dict[str, Dict[str, List[str]]]:
        """Get all available attribute presets for DICOM queries.
        
        This tool returns the defined attribute presets that can be used with the
        query_* functions. It shows which DICOM attributes are included in each
        preset (minimal, standard, extended) for each query level.
        
        Returns:
            Dictionary organized by query level (patient, study, series, instance),
            with each level containing the attribute presets and their associated
            DICOM attributes.
        """
        return ATTRIBUTE_PRESETS
    
    return mcp

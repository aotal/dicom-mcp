# dicom_scp.py
import os
import logging
from pathlib import Path
from pynetdicom import AE, evt, AllStoragePresentationContexts, ALL_TRANSFER_SYNTAXES
from pynetdicom.sop_class import Verification
import pydicom
from pydicom.dataset import FileMetaDataset

from .config import DicomConfiguration, DicomNodeConfig

logger = logging.getLogger(__name__)

def handle_store(event, storage_dir: Path):
    """Handler for the evt.EVT_C_STORE event in the receiving SCP."""
    try:
        ds = event.dataset
        
        if not hasattr(ds, 'SOPInstanceUID'):
            logger.error("Received C-STORE dataset has no SOPInstanceUID.")
            return 0xA801 # Processing failure

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        meta.ImplementationVersionName = "PYNETDICOM_MCP_SCP"
        meta.TransferSyntaxUID = event.context.transfer_syntax
        
        ds.file_meta = meta
        ds.is_little_endian = meta.TransferSyntaxUID.is_little_endian
        ds.is_implicit_VR = meta.TransferSyntaxUID.is_implicit_VR

        filename = ds.SOPInstanceUID + ".dcm"
        filepath = storage_dir / filename
        
        ds.save_as(filepath, enforce_file_format=True)
        
        logger.info(f"DICOM file received and saved: {filepath}")
        return 0x0000 # Success
    except Exception as e:
        sop_uid_for_log = getattr(event.dataset, 'SOPInstanceUID', 'UNKNOWN_UID')
        logger.error(f"Error handling C-STORE for SOPInstanceUID '{sop_uid_for_log}': {e}", exc_info=True)
        return 0xC001 # Error: Cannot process

def handle_echo(event):
    """Handler for the evt.EVT_C_ECHO event."""
    calling_ae = getattr(event.assoc.ae, 'calling_ae_title', 'Unknown')
    logger.info(f"Received C-ECHO from {calling_ae}")
    return 0x0000 

def start_scp_server(config: DicomConfiguration, callback=None):
    """
    Starts the C-STORE SCP server. This function is blocking.
    If a callback is provided, it is called with the AE server instance.
    """
    
    scp_node_config = config.nodes.get('local_scp')
    if not scp_node_config:
        logger.error("No 'local_scp' configuration found in the configuration file.")
        return

    storage_dir = Path(config.local_storage_dir)
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create DICOM reception directory at '{storage_dir}': {e}")
        return

    # Create a partial function for the C-STORE handler that includes the storage directory
    from functools import partial
    bound_handle_store = partial(handle_store, storage_dir=storage_dir)

    handlers = [
        (evt.EVT_C_STORE, bound_handle_store),
        (evt.EVT_C_ECHO, handle_echo)
    ]

    ae_scp = AE(ae_title=scp_node_config.ae_title)

    for context in AllStoragePresentationContexts:
        ae_scp.add_supported_context(context.abstract_syntax, ALL_TRANSFER_SYNTAXES)
    ae_scp.add_supported_context(Verification, ALL_TRANSFER_SYNTAXES)

    host = scp_node_config.host
    port = scp_node_config.port
    
    logger.info(f"Starting C-STORE SCP server on {host}:{port} with AET: {ae_scp.ae_title}")
    
    if callback:
        callback(ae_scp)
    
    try:
        ae_scp.start_server((host, port), block=True, evt_handlers=handlers)
    except Exception as e:
        logger.error(f"Fatal error starting or during SCP server execution: {e}", exc_info=True)
    finally:
        logger.info("SCP server stopped.")

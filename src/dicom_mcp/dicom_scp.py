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
    """Manejador para el evento evt.EVT_C_STORE en el SCP receptor."""
    try:
        ds = event.dataset
        
        if not hasattr(ds, 'SOPInstanceUID'):
            logger.error("Dataset C-STORE recibido no tiene SOPInstanceUID.")
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
        
        logger.info(f"Archivo DICOM recibido y guardado: {filepath}")
        return 0x0000 # Éxito
    except Exception as e:
        sop_uid_for_log = getattr(event.dataset, 'SOPInstanceUID', 'UID_DESCONOCIDO')
        logger.error(f"Error al manejar C-STORE para SOPInstanceUID '{sop_uid_for_log}': {e}", exc_info=True)
        return 0xC001 # Error: No se puede procesar

def handle_echo(event):
    """Manejador para el evento evt.EVT_C_ECHO."""
    calling_ae = getattr(event.assoc.ae, 'calling_ae_title', 'Desconocido')
    logger.info(f"Recibido C-ECHO de {calling_ae}")
    return 0x0000 

def start_scp_server(config: DicomConfiguration, callback=None):
    """
    Inicia el servidor C-STORE SCP. Esta función es bloqueante.
    Si se proporciona un callback, se llama con la instancia del servidor AE.
    """
    
    scp_node_config = config.nodes.get('local_scp')
    if not scp_node_config:
        logger.error("No se encontró la configuración para 'local_scp' en el fichero de configuración.")
        return

    storage_dir = Path(config.local_storage_dir)
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"No se pudo crear el directorio de recepción DICOM en '{storage_dir}': {e}")
        return

    # Crear una función parcial para el manejador de C-STORE que incluya el directorio de almacenamiento
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
    
    logger.info(f"Iniciando servidor C-STORE SCP en {host}:{port} con AET: {ae_scp.ae_title}")
    
    if callback:
        callback(ae_scp)
    
    try:
        ae_scp.start_server((host, port), block=True, evt_handlers=handlers)
    except Exception as e:
        logger.error(f"Error fatal al iniciar o durante la ejecución del servidor SCP: {e}", exc_info=True)
    finally:
        logger.info("Servidor SCP detenido.")

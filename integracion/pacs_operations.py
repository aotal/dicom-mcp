# pacs_operations.py
import asyncio
import logging
import os # [cite: 220]
from pydicom.dataset import Dataset # [cite: 220, 237]
from pathlib import Path
import functools
from typing import Dict, List, Any, Optional, Dict, Tuple # [cite: 31]

logger = logging.getLogger(__name__)

import pydicom # [cite: 32]
from pydicom.dataset import Dataset as DicomDataset
from pynetdicom import AE, debug_logger, evt, build_context # [cite: 220]
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind, StudyRootQueryRetrieveInformationModelMove

from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelFind,
    # SOP Classes de Almacenamiento Comunes añadidas para _create_ae_with_contexts
    CTImageStorage,
    MRImageStorage,
    ComputedRadiographyImageStorage,
    SecondaryCaptureImageStorage,
    DigitalXRayImageStorageForPresentation,
    Verification # MODIFIED: Changed from VerificationSOPClass
)
import config

try:
    # Intentamos con el nombre que Python sugirió que podría existir
    from pynetdicom.sop_class import StoragePresentationContexts
    logger.debug("StoragePresentationContexts importado exitosamente.")
except ImportError:
    StoragePresentationContexts = None
    logger.warning("No se pudo importar StoragePresentationContexts. Los contextos de almacenamiento se añadirán explícitamente.")

from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid

debug_logger() # Descomentar para logs detallados

# --- INICIO DE LA SECCIÓN CORREGIDA ---

# Define la función helper que se ejecutará en el hilo para C-FIND
def _execute_c_find_and_convert_to_list(current_assoc, id_dataset, model_uid_str):
    """
    Ejecuta C-FIND y convierte el generador resultante a una lista.
    Toda esta función se ejecuta en un hilo separado.
    """
    print(f"[_execute_c_find_and_convert_to_list] Ejecutando assoc.send_c_find con model_uid: {model_uid_str}") # DEBUG
    responses_generator = current_assoc.send_c_find(id_dataset, model_uid_str)
    result_list = list(responses_generator)
    print(f"[_execute_c_find_and_convert_to_list] C-FIND completado, {len(result_list)} respuestas recibidas en total (status, identifier pairs).") # DEBUG
    return result_list

async def perform_c_find_async(identifier: Dataset, pacs_config: dict, query_model_uid: str) -> list: # [cite: 34]
    print("[perform_c_find_async] Iniciando...") # DEBUG
    ae = AE(ae_title=pacs_config["AE_TITLE"]) # [cite: 23]
    actual_query_model_sop_class_uid = ""

    if query_model_uid.upper() == 'S':
        ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind) # [cite: 24]
        actual_query_model_sop_class_uid = StudyRootQueryRetrieveInformationModelFind # MODIFICADO [cite: 239]
    elif query_model_uid.upper() == 'P':
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind) # [cite: 24]
        actual_query_model_sop_class_uid = PatientRootQueryRetrieveInformationModelFind # MODIFICADO [cite: 239]
    else:
        print(f"[perform_c_find_async] Error: Query model UID '{query_model_uid}' no reconocido.")
        return []

    print(f"[perform_c_find_async] AE Title local: {pacs_config['AE_TITLE']}") # DEBUG
    print(f"[perform_c_find_async] Conectando a PACS: IP={pacs_config['PACS_IP']}, Puerto={pacs_config['PACS_PORT']}, AET={pacs_config['PACS_AET']}") # DEBUG

    assoc = await asyncio.to_thread(
        ae.associate,
        pacs_config["PACS_IP"], # [cite: 33]
        pacs_config["PACS_PORT"], # [cite: 33]
        ae_title=pacs_config["PACS_AET"] # [cite: 33]
    )

    results = []
    if assoc.is_established:
        print("[perform_c_find_async] Asociación establecida.") # DEBUG
        try:
            print(f"[perform_c_find_async] Dataset Identificador para C-FIND:\n{identifier}") # DEBUG
            print(f"[perform_c_find_async] SOP Class UID del modelo de consulta: {actual_query_model_sop_class_uid}") # DEBUG

            if not actual_query_model_sop_class_uid:
                print("[perform_c_find_async] Error: No se pudo determinar la SOP Class UID del modelo de consulta (está vacía).") # DEBUG
            else:
                # LLAMADA CORREGIDA a asyncio.to_thread usando la función helper
                responses = await asyncio.to_thread(
                    _execute_c_find_and_convert_to_list, # Función helper
                    assoc,                               # Primer argumento para la helper
                    identifier,                          # Segundo argumento para la helper
                    actual_query_model_sop_class_uid     # Tercer argumento para la helper
                )
                print(f"[perform_c_find_async] 'responses' (lista de tuplas status, identifier) recibido de la operación C-FIND (longitud): {len(responses) if responses is not None else 'None'}") # DEBUG

                for (status, result_identifier_ds) in responses: # [cite: 239]
                    print(f"[perform_c_find_async] Procesando respuesta C-FIND con estado: {status}") # DEBUG
                    if status and status.Status in (0xFF00, 0xFF01): # Pending [cite: 240]
                        if result_identifier_ds:
                            print(f"[perform_c_find_async] Identificador de resultado pendiente: {result_identifier_ds.get('PatientID', 'N/A')}, {result_identifier_ds.get('StudyInstanceUID', 'N/A')}") # DEBUG
                            results.append(result_identifier_ds)
                    elif status and status.Status == 0x0000: # Success [cite: 242]
                        print("[perform_c_find_async] Respuesta C-FIND final: Éxito (normalmente sin datos adicionales aquí).") # DEBUG
                    else: # Other statuses like Failure, Cancel, etc.
                        status_val = status.Status if status else 'N/A'
                        print(f"[perform_c_find_async] Respuesta C-FIND con estado no manejado o de error: {status_val}") # DEBUG

        except Exception as e:
            print(f"[perform_c_find_async] Excepción durante C-FIND o procesamiento de respuesta: {e}") # DEBUG
            logger.error(f"Excepción en perform_c_find_async: {e}", exc_info=True) # Logueo formal
            raise # Re-lanzar para que FastAPI devuelva un 500 y veas el error
        finally:
            print("[perform_c_find_async] Liberando asociación.") # DEBUG
            await asyncio.to_thread(assoc.release) # [cite: 232]
    else:
        print("[perform_c_find_async] Asociación NO establecida.") # DEBUG
        # Considera lanzar una excepción aquí para errores de conexión
        # raise ConnectionError("No se pudo establecer la asociación con el PACS.")

    print(f"[perform_c_find_async] Devolviendo {len(results)} resultados.") # DEBUG
    return results

# --- FIN DE LA SECCIÓN CORREGIDA ---

def _create_ae_with_contexts(client_aet_title: str, dicom_dataset: Optional[pydicom.Dataset] = None) -> AE:
    ae = AE(ae_title=client_aet_title)
    transfer_syntaxes_to_propose = [ExplicitVRLittleEndian, ImplicitVRLittleEndian]

    # Añadir explícitamente las SOP Classes de almacenamiento que se quieren soportar
    common_storage_sops = [
        CTImageStorage, MRImageStorage, ComputedRadiographyImageStorage,
        SecondaryCaptureImageStorage, DigitalXRayImageStorageForPresentation
    ]
    for sop_class in common_storage_sops:
        if sop_class: # sop_class puede ser None si alguna no se importó
            ae.add_requested_context(sop_class, transfer_syntaxes_to_propose)
    logger.debug("Contextos de almacenamiento comunes añadidos explícitamente.")

    if dicom_dataset and hasattr(dicom_dataset, 'SOPClassUID'):
        try:
            ae.add_requested_context(dicom_dataset.SOPClassUID, transfer_syntaxes_to_propose)
            logger.debug(f"Contexto específico añadido para SOP Class: {dicom_dataset.SOPClassUID.name if hasattr(dicom_dataset.SOPClassUID, 'name') else dicom_dataset.SOPClassUID}")
        except Exception as e_ctx:
            logger.warning(f"No se pudo añadir contexto específico para SOP Class {getattr(dicom_dataset, 'SOPClassUID', 'Desconocida')}: {e_ctx}")

    # Uso de VerificationSOPClass (nombre estándar)
    if VerificationSOPClass:
        ae.add_requested_context(VerificationSOPClass) # [cite: 220]
        logger.debug("Contexto de presentación para Verification (C-ECHO) añadido.")
    else:
        logger.error("No se pudo importar la SOP Class 'VerificationSOPClass'. El C-ECHO podría no funcionar.")
    return ae


async def perform_c_move_async(
    identifier: DicomDataset,
    pacs_config: Dict[str, Any],
    move_destination_aet: str,
    query_model_uid: str
) -> List[Tuple[DicomDataset, Optional[DicomDataset]]]:
    """
    Realiza una operación C-MOVE de forma asíncrona.
    Devuelve una lista de tuplas (status_dataset, identifier_dataset) de las respuestas.
    """
    loop = asyncio.get_running_loop()
    ae_title = pacs_config.get("AE_TITLE", "PYNETDICOM")
    pacs_ip = pacs_config.get("PACS_IP", "127.0.0.1")
    pacs_port = pacs_config.get("PACS_PORT", 11112)
    pacs_aet = pacs_config.get("PACS_AET", "DCM4CHEE")

    logger.info(f"Iniciando C-MOVE hacia {move_destination_aet}...")
    print(f"[perform_c_move_async] Iniciando C-MOVE...")
    print(f"[perform_c_move_async] AE Title local: {ae_title}")
    print(f"[perform_c_move_async] Conectando a PACS: IP={pacs_ip}, Puerto={pacs_port}, AET={pacs_aet}")
    print(f"[perform_c_move_async] Destino del MOVE: {move_destination_aet}")
    print(f"[perform_c_move_async] Dataset Identificador para C-MOVE:\n{identifier}")

    if query_model_uid == 'S':
        model_sop_class = StudyRootQueryRetrieveInformationModelMove
    else:
        raise ValueError(f"Modelo de consulta UID '{query_model_uid}' no soportado para C-MOVE.")
    
    print(f"[perform_c_move_async] SOP Class UID del modelo de consulta: {model_sop_class}")

    ae = AE(ae_title=ae_title)
    ae.add_requested_context(model_sop_class)
    
    # ... dentro de perform_c_move_async ...

    # Pre-configuramos la llamada a ae.associate con todos sus argumentos
    associate_callable = functools.partial(
    ae.associate, 
    pacs_ip, 
    pacs_port, 
    ae_title=pacs_aet
    )
    assoc = await loop.run_in_executor(None, associate_callable)

    results = []
    if assoc.is_established:
        print("[perform_c_move_async] Asociación establecida para C-MOVE.")
        
        responses_generator = await loop.run_in_executor(
            None,
            assoc.send_c_move,
            identifier,
            move_destination_aet,
            model_sop_class
        )
        
        for status_ds, returned_identifier_ds in responses_generator:
            results.append((status_ds, returned_identifier_ds))
            if status_ds:
                print(f"[perform_c_move_async] Respuesta C-MOVE Status: 0x{status_ds.Status:04X}")
                if 'NumberOfRemainingSuboperations' in status_ds:
                    print(f"  Restantes: {status_ds.NumberOfRemainingSuboperations}, "
                          f"Completadas: {status_ds.NumberOfCompletedSuboperations}, "
                          f"Fallidas: {status_ds.NumberOfFailedSuboperations}, "
                          f"Advertencias: {status_ds.NumberOfWarningSuboperations}")
        
        print("[perform_c_move_async] Liberando asociación C-MOVE.")
        await loop.run_in_executor(None, assoc.release)
    else:
        print("[perform_c_move_async] Fallo al establecer asociación C-MOVE.")
        raise ConnectionError("No se pudo establecer la asociación C-MOVE con el PACS.")
    
    print(f"[perform_c_move_async] Operación C-MOVE completada, devolviendo {len(results)} respuestas de estado.")
    return results

def _perform_pacs_send_sync(
    ae_instance: AE,
    filepath_str: str, # filepath_str es el que se pasa a send_c_store
    pacs_config: Dict[str, Any]
) -> bool:
    assoc = None
    file_basename = Path(filepath_str).name
    pacs_target_info = f"{pacs_config.get('PACS_AET', 'PACS_DESCONOCIDO')}@{pacs_config.get('PACS_IP', 'IP_DESCONOCIDA')}:{pacs_config.get('PACS_PORT', 'PUERTO_DESCONOCIDO')}"

    try:
        logger.info(f"Intentando asociar con PACS ({pacs_target_info}) para enviar: {file_basename}")
        assoc = ae_instance.associate(
            pacs_config["PACS_IP"],
            pacs_config["PACS_PORT"],
            ae_title=pacs_config["PACS_AET"]
        )
        if assoc.is_established:
            logger.info(f"Asociación establecida con PACS para {file_basename}. Contextos de presentación aceptados: {len(assoc.accepted_contexts)}")

            sop_class_to_send = None
            is_context_accepted_for_sop_class = False
            try:
                ds_to_send = pydicom.dcmread(filepath_str, force=True, stop_before_pixels=True, specific_tags=['SOPClassUID'])
                sop_class_to_send = ds_to_send.SOPClassUID
                is_context_accepted_for_sop_class = any(
                    ctx.abstract_syntax == sop_class_to_send for ctx in assoc.accepted_contexts
                )
                if not is_context_accepted_for_sop_class:
                    sop_name = sop_class_to_send.name if hasattr(sop_class_to_send, 'name') else str(sop_class_to_send)
                    logger.warning(f"Ningún contexto fue aceptado para la SOP Class específica ({sop_name}) del fichero {file_basename}.")
            except Exception as e_check:
                logger.warning(f"No se pudo leer SOPClassUID de {file_basename} para verificar contexto aceptado antes del envío: {e_check}.")

            if not assoc.accepted_contexts: # Chequeo general
                logger.warning(f"Asociación establecida para {file_basename} pero NO SE ACEPTARON CONTEXTOS DE PRESENTACIÓN. El C-STORE probablemente fallará.")
                return False

            if not is_context_accepted_for_sop_class and assoc.accepted_contexts:
                sop_name = sop_class_to_send.name if hasattr(sop_class_to_send, 'name') else str(sop_class_to_send)
                logger.warning(f"Aunque hay contextos aceptados, ninguno coincide con la SOP Class ({sop_name}) del fichero {file_basename}. El envío podría fallar.")

            status_store = assoc.send_c_store(filepath_str) # [cite: 231]

            if status_store:
                logger.debug(f"Respuesta C-STORE completa para {file_basename}: {status_store}")
                status_value = getattr(status_store, 'Status', None)
                if status_value == 0x0000:
                    logger.info(f"ÉXITO: {file_basename} enviado correctamente a PACS. Estado: 0x{status_value:04X}")
                    return True
                else:
                    error_comment = getattr(status_store, 'ErrorComment', 'Sin comentario de error específico.')
                    status_description = evt.STATUS_KEYWORDS.get(status_value, f'Estado desconocido (0x{status_value:04X})' if status_value is not None else 'Estado no recibido')
                    logger.error(f"FALLO C-STORE para {file_basename}: {status_description}. Comentario: {error_comment}")
                    return False
            else:
                logger.error(f"No se recibió dataset de estado de C-STORE para {file_basename}.")
                return False
        else:
            logger.error(f"No se pudo establecer asociación con PACS ({pacs_target_info}) para {file_basename}.")
            if assoc and hasattr(assoc, 'acceptor') and assoc.acceptor and hasattr(assoc.acceptor, 'primitive'):
                logger.error(f"   Detalles de la primitiva A-ASSOCIATE-RJ (si es un rechazo): {assoc.acceptor.primitive}")
            return False
    except ConnectionRefusedError:
        logger.error(f"Error de conexión: El PACS ({pacs_target_info}) rechazó la conexión para {file_basename}.")
        return False
    except ValueError as ve:
        if "Failed to encode" in str(ve):
            logger.error(f"ERROR DE CODIFICACIÓN DEL DATASET para {file_basename} antes del envío: {ve}", exc_info=True)
        else:
            logger.exception(f"ValueError no esperado durante la operación PACS para {file_basename}: {ve}")
        return False
    except Exception as e:
        logger.exception(f"Excepción no esperada durante la operación PACS para {file_basename}: {e}")
        return False
    finally:
        if assoc and assoc.is_established:
            assoc.release()
            logger.debug(f"Asociación con PACS liberada para {file_basename}.")


async def send_single_dicom_file_async(filepath_str: str, pacs_config: Dict[str, Any]) -> bool:
    filepath = Path(filepath_str)
    if not filepath.is_file():
        logger.error(f"Fichero DICOM para envío a PACS no existe o no es un fichero: {filepath_str}")
        return False
    dicom_dataset_for_context: Optional[pydicom.Dataset] = None
    try:
        dicom_dataset_for_context = pydicom.dcmread(
            filepath_str,
            force=True,
            stop_before_pixels=True,
            specific_tags=['SOPClassUID']
        )
    except Exception as e_read_meta:
        logger.warning(f"No se pudo leer SOPClassUID de {filepath.name} para contexto específico: {e_read_meta}. ")

    ae_instance = _create_ae_with_contexts(
        pacs_config.get("AE_TITLE", "MYPYTHONSCU"), # Usa el AE_TITLE de pacs_config (CLIENT_AET)
        dicom_dataset_for_context
    )
    try:
        return await asyncio.to_thread(
            _perform_pacs_send_sync,
            ae_instance,
            filepath_str,
            pacs_config
        )
    except Exception as e:
        logger.error(f"Error en asyncio.to_thread durante el envío PACS de {filepath.name}: {e}", exc_info=True)
        return False

async def send_dicom_folder_async(folder_path_str: str, pacs_config: Dict[str, Any]) -> bool:
    folder_path = Path(folder_path_str)
    if not folder_path.is_dir():
        logger.error(f"La ruta para envío a PACS no es un directorio válido: {folder_path_str}")
        return False
    dicom_files = [f for f in folder_path.glob("*.dcm") if f.is_file()]
    if not dicom_files:
        logger.info(f"No se encontraron archivos .dcm en {folder_path_str} para enviar al PACS.")
        return True
    pacs_target_info = f"{pacs_config.get('PACS_AET', 'PACS_DESCONOCIDO')}@{pacs_config.get('PACS_IP', 'IP_DESCONOCIDA')}:{pacs_config.get('PACS_PORT', 'PUERTO_DESCONOCIDO')}"
    logger.info(f"Iniciando envío de {len(dicom_files)} archivos desde {folder_path_str} al PACS ({pacs_target_info}).")
    tasks = [send_single_dicom_file_async(str(dicom_file), pacs_config) for dicom_file in dicom_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = 0
    failure_count = 0
    total_files = len(dicom_files)
    for i, res in enumerate(results):
        file_basename = dicom_files[i].name
        if isinstance(res, Exception):
            logger.error(f"Excepción no controlada enviando {file_basename} al PACS (asyncio.gather): {res}", exc_info=res)
            failure_count += 1
        elif res:
            success_count += 1
        else:
            failure_count +=1
    logger.info(f"Resultado del envío masivo a PACS desde {folder_path_str}: "
                f"{success_count} de {total_files} exitosos, {failure_count} de {total_files} fallidos.")
    return failure_count == 0

async def _test_pacs_operations():
    class MockConfigTesting:
        PACS_IP = "jupyter.arnau.scs.es" # O tu localhost si dcm4chee está local
        PACS_PORT = 11112
        PACS_AET = "DCM4CHEE" # AE Title de tu dcm4chee
        CLIENT_AET = "MYPYTHONSCU" # AE Title para este cliente de prueba
        OUTPUT_TEST_DIR_FOR_PACS = Path("output_pacs_test_send_v_anterior")

    MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS.mkdir(parents=True, exist_ok=True)
    test_dicom_path: Optional[Path] = None
    try:
        file_meta_test = pydicom.Dataset()
        file_meta_test.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta_test.MediaStorageSOPInstanceUID = generate_uid()
        file_meta_test.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta_test.TransferSyntaxUID = ExplicitVRLittleEndian

        ds_test = pydicom.dataset.FileDataset(None, {}, file_meta=file_meta_test, preamble=b"\0" * 128)
        ds_test.is_little_endian = True
        ds_test.is_implicit_VR = False

        ds_test.PatientName = "Test^PACS^Anterior"
        ds_test.PatientID = "TestPACSAnterior01"
        ds_test.StudyInstanceUID = generate_uid()
        ds_test.SeriesInstanceUID = generate_uid()
        ds_test.SOPInstanceUID = file_meta_test.MediaStorageSOPInstanceUID
        ds_test.SOPClassUID = file_meta_test.MediaStorageSOPClassUID # CTImageStorage
        ds_test.Modality = "CT"
        ds_test.InstanceNumber = "1"
        ds_test.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds_test.SpecificCharacterSet = "ISO_IR 100"
        ds_test.Rows = 2
        ds_test.Columns = 2
        ds_test.BitsAllocated = 16
        ds_test.BitsStored = 16
        ds_test.HighBit = 15
        ds_test.PixelRepresentation = 0
        ds_test.SamplesPerPixel = 1
        ds_test.PhotometricInterpretation = "MONOCHROME2"
        import numpy as np
        ds_test.PixelData = np.array([[100, 200], [300, 400]], dtype=np.uint16).tobytes()

        test_dicom_path_str = str(MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS / f"test_ct_anterior_{ds_test.SOPInstanceUID[:12]}.dcm")
        test_dicom_path = Path(test_dicom_path_str)

        pydicom.dcmwrite(test_dicom_path_str, ds_test, write_like_original=False)
        logger.info(f"Archivo DICOM de prueba (anterior) creado en: {test_dicom_path_str}")

        pacs_config_for_test = {
            "PACS_IP": MockConfigTesting.PACS_IP,
            "PACS_PORT": MockConfigTesting.PACS_PORT,
            "PACS_AET": MockConfigTesting.PACS_AET,
            "AE_TITLE": MockConfigTesting.CLIENT_AET # Este es el AE_TITLE de nuestro SCU
        }
        logger.info(f"Iniciando prueba de envío (anterior) a PACS: {pacs_config_for_test['PACS_AET']}...")
        success_single = await send_single_dicom_file_async(test_dicom_path_str, pacs_config_for_test)
        print(f"Resultado envío único (anterior): {'Éxito' if success_single else 'Fallo'}")

    except Exception as e:
        logger.error(f"Error en la prueba de pacs_operations (anterior): {e}", exc_info=True)
    finally:
        import shutil
        if test_dicom_path and test_dicom_path.exists():
            try: test_dicom_path.unlink(missing_ok=True)
            except Exception as e_del: logger.warning(f"No se pudo eliminar {test_dicom_path}: {e_del}")
        if MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS.exists():
            try:
                shutil.rmtree(MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS)
            except Exception as e_del_dir: logger.warning(f"No se pudo eliminar el dir {MockConfigTesting.OUTPUT_TEST_DIR_FOR_PACS}: {e_del_dir}")
        logger.info("Limpieza de prueba de pacs_operations (anterior) completada.")

if __name__ == '__main__':
    # import asyncio # Ya importado arriba
    import shutil
    # from pathlib import Path # Ya importado arriba
    import numpy as np

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s [%(threadName)s] - %(levelname)s - %(message)s')

    logger.info("Iniciando prueba de pacs_operations.py (versión anterior)...")
    asyncio.run(_test_pacs_operations())
    logger.info("Prueba de pacs_operations.py (versión anterior) finalizada.")
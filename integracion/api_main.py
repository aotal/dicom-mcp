. He puesto un main# api_main.py
import logging
import re
import io
import os
import json
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.responses import JSONResponse, FileResponse
from typing import Any, List, Optional, Dict, Tuple
import threading
from contextlib import asynccontextmanager
import httpx

# --- NUEVAS IMPORTACIONES ---
import pynetdicom # <--- AÑADIDO PARA RESOLVER EL NameError
from models import (
    StudyResponse,
    SeriesResponse,
    InstanceMetadataResponse,
    LUTExplanationModel,      # <-- AÑADIDO
    PixelDataResponse,        # <-- AÑADIDO
    MoveRequest,
    BulkMoveRequest
)
# ... (el resto del fichero permanece igual que en la versión que te proporcioné)
from pydicom.dataset import Dataset as DicomDataset
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag, tag_for_keyword

import pacs_operations
import config
import dicom_scp

# --- Configuración del Logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        force=True
    )

# --- Lifespan Manager (Corregido) ---
scp_thread: Optional[threading.Thread] = None
# La anotación de tipo ahora funcionará
scp_server_instance: Optional[pynetdicom.AE] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scp_thread, scp_server_instance
    logger.info("Iniciando aplicación FastAPI y servidor DICOM C-STORE SCP...")
    
    # El callback ahora puede asignar la instancia del servidor AE de pynetdicom
    def server_callback(server_instance):
        globals()['scp_server_instance'] = server_instance

    scp_thread = threading.Thread(
        target=dicom_scp.start_scp_server,
        args=(server_callback,),
        daemon=True
    )
    scp_thread.start()

    yield

    logger.info("Deteniendo aplicación FastAPI...")
    
    if scp_server_instance:
        logger.info("Solicitando apagado del servidor SCP...")
        scp_server_instance.shutdown()

    if scp_thread and scp_thread.is_alive():
        logger.info("Esperando que el hilo del SCP termine...")
        scp_thread.join(timeout=5.0)
        if scp_thread.is_alive():
            logger.warning("Advertencia: El hilo del servidor SCP no terminó limpiamente.")
    logger.info("Apagado completado.")


app = FastAPI(
    title="API Gateway para Consultas DICOM (QIDO-RS & DIMSE)",
    version="2.0.1", # Incremento de versión por la corrección
    description="Actúa como un proxy para QIDO-RS y un endpoint para C-FIND/C-MOVE.",
    lifespan=lifespan
)

# ... (El resto del código de api_main.py no necesita cambios)
# --- Definición de "Attribute Sets" para el Proxy ---
FASTAPI_ATTRIBUTE_SETS = {
    "QC_Convencional": [
        "00180060",  # KVP
        "00181151",  # XRayTubeCurrent
        "00181153",  # ExposureInuAs
        "00204000"   # ImageComments
    ],
    "InfoPacienteEstudio": [
        "00100010", # PatientName
        "00081030", # StudyDescription
        "00080090"  # ReferringPhysicianName
    ]
}

# --- ENDPOINTS PARA QIDO-RS (Proxy a dcm4chee-arc-light) ---

async def qido_proxy_handler(request: Request):
    """
    Función genérica para manejar todas las solicitudes de proxy QIDO-RS.
    """
    pacs_url = f"{config.PACS_DICOMWEB_URL}{request.url.path.replace('/dicom-web', '')}"
    
    incoming_params = dict(request.query_params)
    logger.info(f"Solicitud Proxy QIDO entrante para: {request.url}")
    logger.debug(f"Parámetros entrantes: {incoming_params}")

    if 'includefield' in incoming_params:
        expanded_fields = []
        fields_to_check = incoming_params['includefield'].split(',')
        
        for field in fields_to_check:
            field = field.strip()
            if field in FASTAPI_ATTRIBUTE_SETS:
                expanded_fields.extend(FASTAPI_ATTRIBUTE_SETS[field])
                logger.info(f"Expandiendo Attribute Set '{field}' a tags: {FASTAPI_ATTRIBUTE_SETS[field]}")
            else:
                expanded_fields.append(field)
        
        incoming_params['includefield'] = ",".join(expanded_fields)

    logger.info(f"Reenviando solicitud QIDO a: {pacs_url} con parámetros: {incoming_params}")

    async with httpx.AsyncClient() as client:
        try:
            pacs_response = await client.get(
                pacs_url, 
                params=incoming_params,
                headers={"Accept": "application/dicom+json"},
                timeout=30.0
            )
            pacs_response.raise_for_status()
            return JSONResponse(content=pacs_response.json(), status_code=pacs_response.status_code)

        except httpx.HTTPStatusError as exc:
            logger.error(f"Error del PACS real: {exc.response.status_code} - {exc.response.text}")
            raise HTTPException(
                status_code=exc.response.status_code, 
                detail=f"Error del PACS: {exc.response.text}"
            )
        except httpx.RequestError as exc:
            logger.error(f"Error de conexión al contactar el PACS: {exc}")
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo conectar con el PACS en {exc.request.url}."
            )

# Rutas que capturan todas las consultas QIDO-RS
app.add_api_route("/dicom-web/studies", qido_proxy_handler, methods=["GET"])
app.add_api_route("/dicom-web/studies/{study_uid}/series", qido_proxy_handler, methods=["GET"])
app.add_api_route("/dicom-web/studies/{study_uid}/series/{series_uid}/instances", qido_proxy_handler, methods=["GET"])
app.add_api_route("/dicom-web/instances", qido_proxy_handler, methods=["GET"])
app.add_api_route("/dicom-web/series", qido_proxy_handler, methods=["GET"])

# --- Endpoints DIMSE ---
# (Tus endpoints C-FIND, C-MOVE etc., van aquí sin cambios)

# Endpoint para C-MOVE de una sola jerarquía (estudio, serie o instancia única)
@app.post("/retrieve-instance", status_code=202, summary="Solicita al PACS mover un estudio/serie/instancia a esta API")
async def retrieve_instance_via_cmove(item: MoveRequest): # Usa el modelo MoveRequest original
    identifier = DicomDataset()
    identifier.StudyInstanceUID = item.study_instance_uid

    if item.sop_instance_uid:
        if not item.series_instance_uid:
            raise HTTPException(status_code=400, detail="SeriesInstanceUID es requerido para mover una instancia específica.")
        identifier.QueryRetrieveLevel = "IMAGE"
        identifier.SeriesInstanceUID = item.series_instance_uid
        identifier.SOPInstanceUID = item.sop_instance_uid
    elif item.series_instance_uid:
        identifier.QueryRetrieveLevel = "SERIES"
        identifier.SeriesInstanceUID = item.series_instance_uid
        identifier.SOPInstanceUID = "" 
    elif item.study_instance_uid: # Solo StudyInstanceUID
        identifier.QueryRetrieveLevel = "STUDY"
        identifier.SeriesInstanceUID = "" 
        identifier.SOPInstanceUID = ""    
    else:
        # Esto no debería ocurrir si MoveRequest requiere study_instance_uid
        raise HTTPException(status_code=400, detail="Se requiere al menos StudyInstanceUID.")

    logger.info(f"Solicitud C-MOVE para: QueryLevel='{identifier.QueryRetrieveLevel}', StudyUID='{identifier.StudyInstanceUID}', SeriesUID='{identifier.get('SeriesInstanceUID', 'N/A')}', SOPInstanceUID='{identifier.get('SOPInstanceUID', 'N/A')}'")

    pacs_config_dict = {
        "PACS_IP": config.PACS_IP, "PACS_PORT": config.PACS_PORT,
        "PACS_AET": config.PACS_AET, "AE_TITLE": config.CLIENT_AET 
    }
    move_destination = config.API_SCP_AET
    try:
        move_responses = await pacs_operations.perform_c_move_async(
            identifier, pacs_config_dict, move_destination_aet=move_destination, query_model_uid='S' # Asume Study Root
        )
        
        # Interpretar la respuesta C-MOVE
        # La lista move_responses contiene tuplas de (status_dataset, identifier_dataset)
        # El último status_dataset es el que indica el estado final de la operación C-MOVE general.
        final_status_ds = None
        num_completed = 0
        num_failed = 0
        num_warning = 0

        if move_responses:
            for status_ds_item, _ in move_responses: # Iterar para obtener el último estado y contadores
                if status_ds_item:
                    final_status_ds = status_ds_item
                    num_completed = status_ds_item.get("NumberOfCompletedSuboperations", num_completed)
                    num_failed = status_ds_item.get("NumberOfFailedSuboperations", num_failed)
                    num_warning = status_ds_item.get("NumberOfWarningSuboperations", num_warning)
        
        if final_status_ds and hasattr(final_status_ds, 'Status'):
            status_val = final_status_ds.Status
            msg = (
                f"Operación C-MOVE. Estado final del PACS: 0x{status_val:04X}. "
                f"Sub-operaciones: Completadas={num_completed}, Fallidas={num_failed}, Advertencias={num_warning}."
            )
            if status_val == 0x0000: # Éxito
                logger.info(msg)
                return {"message": msg}
            elif status_val == 0xFF00: # Pending (raro como estado final, pero posible si es la única respuesta)
                 logger.info(f"{msg} La operación está pendiente, esperando más respuestas del PACS.")
                 return {"message": f"{msg} La operación está pendiente."}
            else: # Fallo o advertencia
                logger.error(msg)
                raise HTTPException(status_code=502, detail=msg) # Bad Gateway si el PACS reporta fallo
        else:
            logger.error("No se recibió una respuesta de estado final válida o completa del C-MOVE.")
            raise HTTPException(status_code=502, detail="Respuesta C-MOVE incompleta o no exitosa del PACS.")

    except ConnectionError as e:
        logger.error(f"Error de conexión C-MOVE: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Error de conexión al PACS para C-MOVE: {str(e)}")
    except Exception as e:
        logger.error(f"Error al solicitar C-MOVE: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante C-MOVE: {str(e)}")

@app.post("/retrieve-multiple-instances", status_code=202, summary="Solicita al PACS mover múltiples instancias específicas a esta API")
async def retrieve_multiple_instances_via_cmove(request_data: BulkMoveRequest):
    pacs_config_dict = {
        "PACS_IP": config.PACS_IP,
        "PACS_PORT": config.PACS_PORT,
        "PACS_AET": config.PACS_AET,
        "AE_TITLE": config.CLIENT_AET
    }
    move_destination_aet = config.API_SCP_AET
    responses_summary = []
    
    if not request_data.instances_to_move:
        raise HTTPException(status_code=400, detail="La lista 'instances_to_move' no puede estar vacía.")

    for instance_info in request_data.instances_to_move:
        identifier = DicomDataset()
        identifier.QueryRetrieveLevel = "IMAGE"
        identifier.StudyInstanceUID = instance_info.study_instance_uid
        identifier.SeriesInstanceUID = instance_info.series_instance_uid
        identifier.SOPInstanceUID = instance_info.sop_instance_uid
        
        instance_response_summary = {
            "study_instance_uid": instance_info.study_instance_uid,
            "series_instance_uid": instance_info.series_instance_uid,
            "sop_instance_uid": instance_info.sop_instance_uid,
            "status_code_hex": "N/A",
            "message": "No procesado",
            "sub_operations_completed": 0,
            "sub_operations_failed": 0,
            "sub_operations_warning": 0
        }

        try:
            logger.info(f"Iniciando C-MOVE para SOPInstanceUID: {instance_info.sop_instance_uid} hacia {move_destination_aet}")
            move_responses_single = await pacs_operations.perform_c_move_async(
                identifier, pacs_config_dict, move_destination_aet=move_destination_aet, query_model_uid='S'
            )
            
            final_status_ds_single = None
            num_completed_single = 0
            num_failed_single = 0
            num_warning_single = 0

            if move_responses_single:
                for status_ds_item, _ in move_responses_single:
                    if status_ds_item:
                        final_status_ds_single = status_ds_item
                        num_completed_single = status_ds_item.get("NumberOfCompletedSuboperations", num_completed_single)
                        num_failed_single = status_ds_item.get("NumberOfFailedSuboperations", num_failed_single)
                        num_warning_single = status_ds_item.get("NumberOfWarningSuboperations", num_warning_single)

            instance_response_summary["sub_operations_completed"] = num_completed_single
            instance_response_summary["sub_operations_failed"] = num_failed_single
            instance_response_summary["sub_operations_warning"] = num_warning_single

            if final_status_ds_single and hasattr(final_status_ds_single, 'Status'):
                status_val_single = final_status_ds_single.Status
                instance_response_summary["status_code_hex"] = f"0x{status_val_single:04X}"
                instance_response_summary["message"] = f"Estado final del PACS: 0x{status_val_single:04X}."
                if status_val_single == 0x0000:
                    logger.info(f"C-MOVE para {instance_info.sop_instance_uid} exitoso.")
                else:
                    logger.warning(f"C-MOVE para {instance_info.sop_instance_uid} con estado {status_val_single:#04X}.")
            else:
                instance_response_summary["message"] = f"No se recibió estado final claro del PACS para {instance_info.sop_instance_uid}."
                logger.error(instance_response_summary["message"])

        except ConnectionError as e_conn:
            logger.error(f"Error de conexión durante C-MOVE para {instance_info.sop_instance_uid}: {e_conn}", exc_info=True)
            instance_response_summary["message"] = f"Error de conexión: {str(e_conn)}"
            instance_response_summary["status_code_hex"] = "CONN_ERROR"
        except Exception as e_generic:
            logger.error(f"Error genérico durante C-MOVE para {instance_info.sop_instance_uid}: {e_generic}", exc_info=True)
            instance_response_summary["message"] = f"Error interno del servidor: {str(e_generic)}"
            instance_response_summary["status_code_hex"] = "SERVER_ERROR"
        
        responses_summary.append(instance_response_summary)

    return {
        "message": "Procesamiento de C-MOVE masivo completado. Revise los resultados individuales.",
        "results": responses_summary
    }


@app.get("/retrieved-instances/{sop_instance_uid}/pixeldata", response_model=PixelDataResponse, summary="Obtiene datos de píxeles de una instancia recibida localmente")
async def get_retrieved_instance_pixeldata(sop_instance_uid: str):
    # Validar el SOPInstanceUID para evitar traversal attacks, aunque join lo mitiga.
    # Un UID válido no debería contener '..' o '/'.
    if not re.fullmatch(r"[0-9\.]+", sop_instance_uid): # Patrón simple para UIDs DICOM
        raise HTTPException(status_code=400, detail="SOPInstanceUID con formato inválido.")

    # Usar config.DICOM_RECEIVED_DIR que es un Path object
    filepath = config.DICOM_RECEIVED_DIR / (sop_instance_uid + ".dcm")
    logger.info(f"[get_retrieved_instance_pixeldata] Buscando archivo: {filepath}")

    if not filepath.is_file(): # Usar el método de Path
        logger.warning(f"Archivo DICOM no encontrado en el directorio de recepción: {filepath}")
        raise HTTPException(status_code=404, detail="Archivo DICOM no encontrado. Es posible que C-MOVE no haya completado, fallado, o aún no haya llegado.")
    
    try:
        ds = pydicom.dcmread(str(filepath), force=True) # dcmread necesita string
        if not hasattr(ds, 'PixelData') or ds.PixelData is None:
            raise HTTPException(status_code=404, detail="El objeto DICOM no contiene datos de píxeles (PixelData) válidos.")
        
        pixel_array = ds.pixel_array # Esto puede tardar y consumir memoria para imágenes grandes
        logger.info(f"Array de píxeles obtenido del archivo {filepath}: forma={pixel_array.shape}, tipo={pixel_array.dtype}")
        
        preview = None
        # Crear un preview más pequeño para evitar enviar arrays muy grandes en JSON
        if pixel_array.ndim >= 2 and pixel_array.size > 0:
            # Para imágenes 2D (monocromo o un frame de color)
            if pixel_array.ndim == 2:
                rows_preview = min(pixel_array.shape[0], 5)
                cols_preview = min(pixel_array.shape[1], 5)
                preview = pixel_array[:rows_preview, :cols_preview].tolist()
            # Para imágenes 3D (multiframe monocromo o RGB)
            elif pixel_array.ndim == 3:
                # Asumimos primer frame para preview si es multiframe monocromo
                # o un plano si es color (ej. pixel_array[0] sería el plano R si es (planos, filas, cols))
                # pydicom.pixel_array maneja esto y devuelve (filas, cols) o (filas, cols, samples) o (frames, filas, cols)
                # Si es (frames, filas, cols)
                if ds.get("SamplesPerPixel", 1) == 1: # Monocromo multiframe
                     rows_preview = min(pixel_array.shape[1], 5)
                     cols_preview = min(pixel_array.shape[2], 5)
                     preview = pixel_array[0, :rows_preview, :cols_preview].tolist() # Preview del primer frame
                # Si es (filas, cols, samples) -> color
                elif ds.get("SamplesPerPixel", 1) > 1 and pixel_array.shape[-1] == ds.SamplesPerPixel:
                     rows_preview = min(pixel_array.shape[0], 5)
                     cols_preview = min(pixel_array.shape[1], 5)
                     preview = pixel_array[:rows_preview, :cols_preview, 0].tolist() # Preview del primer canal (ej. Rojo)


        return PixelDataResponse(
            sop_instance_uid=sop_instance_uid,
            rows=ds.Rows,
            columns=ds.Columns,
            pixel_array_shape=pixel_array.shape,
            pixel_array_dtype=str(pixel_array.dtype),
            pixel_array_preview=preview,
            message="Pixel data accessed from locally stored C-MOVE file. Preview shown."
        )
    except Exception as e:
        logger.error(f"Error procesando archivo DICOM almacenado {filepath}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al procesar archivo DICOM almacenado: {str(e)}")

# --- Fin de api_main.py ---
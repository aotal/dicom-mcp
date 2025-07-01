# models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple

class StudyResponse(BaseModel):
    StudyInstanceUID: str
    PatientID: Optional[str] = None
    PatientName: Optional[str] = None
    StudyDate: Optional[str] = None
    StudyDescription: Optional[str] = None
    ModalitiesInStudy: Optional[str] = None
    AccessionNumber: Optional[str] = None

class LUTExplanationModel(BaseModel):
    FullText: Optional[str] = None
    Explanation: Optional[str] = None
    InCalibRange: Optional[Tuple[float, float]] = None
    OutLUTRange: Optional[Tuple[float, float]] = None

class InstanceMetadataResponse(BaseModel):
    SOPInstanceUID: str
    InstanceNumber: Optional[str] = None
    dicom_headers: Dict[str, Any]

class SeriesResponse(BaseModel): 
    StudyInstanceUID: str
    SeriesInstanceUID: str
    Modality: Optional[str] = None
    SeriesNumber: Optional[str] = None
    SeriesDescription: Optional[str] = None

class PixelDataResponse(BaseModel):
    sop_instance_uid: str
    rows: int
    columns: int
    pixel_array_shape: Tuple[int, ...]
    pixel_array_dtype: str
    pixel_array_preview: Optional[List[List[Any]]] = None
    message: Optional[str] = None

class MoveRequest(BaseModel):
    study_instance_uid: str
    series_instance_uid: Optional[str] = None
    sop_instance_uid: Optional[str] = None # Para mover una instancia específica
    # El destino será el AET de nuestro propio SCP
    # destination_aet: str # No es necesario si siempre es nuestro propio SCP    

class IndividualInstanceMoveRequest(BaseModel):
    study_instance_uid: str = Field(..., description="Study Instance UID de la instancia a mover.")
    series_instance_uid: str = Field(..., description="Series Instance UID de la instancia a mover.")
    sop_instance_uid: str = Field(..., description="SOP Instance UID de la instancia específica a mover.")

class BulkMoveRequest(BaseModel):
    instances_to_move: List[IndividualInstanceMoveRequest] = Field(..., description="Lista de instancias específicas a mover.")
    # Opcionalmente, podrías añadir aquí el AET de destino si fuese dinámico,
    # pero según tu config.py, es fijo.
    # destination_aet: Optional[str] = None 

# Asegúrate de que el MoveRequest original se mantiene o se elimina/reemplaza según tu necesidad.
# Si quieres mantener la funcionalidad de mover un estudio/serie completo con el endpoint antiguo,
# puedes crear un nuevo endpoint para el movimiento masivo de instancias.  
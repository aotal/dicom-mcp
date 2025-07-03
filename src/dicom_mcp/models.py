# models.py (VERSIÓN FINAL, CORREGIDA Y PERFECCIONADA 4.2)
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple

# --- MODELO BASE CON VALIDADOR UNIVERSAL Y ROBUSTO ---
class DicomResponseBase(BaseModel):
    """
    Un modelo base que convierte automáticamente los tipos de datos que no son
    primitivos de Python a strings. Esto maneja de forma genérica todos los
    tipos especiales de pydicom.
    """
    @field_validator('*', mode='before')
    @classmethod
    def convert_non_primitive_types_to_str(cls, v: Any) -> Any:
        # --- LÓGICA CORREGIDA ---
        # Define un conjunto con los tipos básicos exactos que no queremos tocar.
        base_types = {str, int, float, list, dict, tuple, type(None)}
        
        # Comprueba si el TIPO EXACTO del valor no está en nuestra lista de tipos base.
        if type(v) not in base_types:
            # Si es un tipo especial de pydicom (IS, DS, PN, UID, etc.),
            # lo convertimos a un string puro para Pydantic.
            return str(v)
        
        # Si ya es un tipo básico, lo devolvemos sin cambios.
        return v

    class Config:
        from_attributes = True

# --- MODELOS DE RESPUESTA HEREDANDO DEL MODELO BASE ---
# No necesitan cambios, ya que heredan la lógica correcta.

class StudyResponse(DicomResponseBase):
    StudyInstanceUID: str
    PatientID: Optional[str] = None
    PatientName: Optional[str] = None
    StudyDate: Optional[str] = None
    StudyDescription: Optional[str] = None
    ModalitiesInStudy: Optional[str] = None
    AccessionNumber: Optional[str] = None

class StudyQueryResultsWrapper(BaseModel):
    result: List[StudyResponse]

class SeriesResponse(DicomResponseBase):
    StudyInstanceUID: str
    SeriesInstanceUID: str
    Modality: Optional[str] = None
    SeriesNumber: Optional[str] = None
    SeriesDescription: Optional[str] = None
    PatientName: Optional[str] = None

class SeriesQueryResultsWrapper(BaseModel):
    result: List[SeriesResponse]

# --- El resto de los modelos no necesitan cambios ---
class LUTExplanationModel(BaseModel):
    FullText: Optional[str] = Field(None)
    Explanation: Optional[str] = Field(None)
    InCalibRange: Optional[Tuple[float, float]] = Field(None)
    OutLUTRange: Optional[Tuple[float, float]] = Field(None)

class InstanceMetadataResponse(BaseModel):
    SOPInstanceUID: str
    InstanceNumber: Optional[str] = None
    # Cambiar a Optional y el valor por defecto a None
    dicom_headers: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True # Asegúrate de que esto esté si usas validación desde atributos de objeto

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
    sop_instance_uid: Optional[str] = None

class MoveRequestItem(BaseModel):
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str

class BulkMoveRequest(BaseModel):
    instances_to_move: List[MoveRequestItem]

# --- Nuevos modelos propuestos ---

# Para list_dicom_nodes
class DicomNodeInfo(BaseModel):
    name: str
    description: str

class DicomNodeListResponse(BaseModel):
    current_node: str
    nodes: List[DicomNodeInfo]

# Para switch_dicom_node
class OperationStatusResponse(BaseModel):
    success: bool
    message: str

# Para verify_connection
class ConnectionVerificationResponse(BaseModel):
    message: str

# Para query_patients
class PatientQueryResult(DicomResponseBase):
    PatientID: str
    PatientName: Optional[str] = None
    PatientBirthDate: Optional[str] = None
    PatientSex: Optional[str] = None

class PatientQueryResultsWrapper(BaseModel):
    result: List[PatientQueryResult]
    # Añadir otros campos comunes de paciente que esperes en la respuesta

# Para get_attribute_presets
class AttributePresetDetails(BaseModel):
    minimal: List[str]
    standard: List[str]
    extended: List[str]

class AttributePresetsResponse(BaseModel):
    patient: AttributePresetDetails
    study: AttributePresetDetails
    series: AttributePresetDetails
    instance: AttributePresetDetails

class QidoResponse(DicomResponseBase):
    class Config:
        extra = 'allow'

class QidoQueryResultsWrapper(BaseModel):
    result: List[QidoResponse]
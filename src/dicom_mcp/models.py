# src/dicom_mcp/models.py (Corregido con todos los modelos)

import re
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo

# --- Modelo Base con Configuración Centralizada ---
class DicomResponseBase(BaseModel):
    """Modelo base que convierte tipos no primitivos a strings y permite campos extra."""
    model_config = ConfigDict(
        from_attributes=True,
        extra='allow'
    )

    @field_validator('*', mode='before')
    @classmethod
    def convert_non_primitive_types_to_str(cls, v: Any) -> Any:
        base_types = {str, int, float, list, dict, tuple, type(None)}
        if type(v) not in base_types:
            return str(v)
        return v

# --- Modelos de Petición y Respuesta ---

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

class InstanceMetadataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    SOPInstanceUID: str
    InstanceNumber: Optional[str] = None
    dicom_headers: Optional[Dict[str, Any]] = None

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

class DicomNodeInfo(BaseModel):
    name: str
    description: str

class DicomNodeListResponse(BaseModel):
    current_node: str
    nodes: List[DicomNodeInfo]

class OperationStatusResponse(BaseModel):
    success: bool
    message: str
    
# --- CORRECCIÓN: Clase añadida que faltaba ---
class ConnectionVerificationResponse(BaseModel):
    """Response model for the C-ECHO verification tool."""
    message: str
    
class PatientQueryResult(DicomResponseBase):
    PatientID: str
    PatientName: Optional[str] = None
    PatientBirthDate: Optional[str] = None
    PatientSex: Optional[str] = None

class PatientQueryResultsWrapper(BaseModel):
    result: List[PatientQueryResult]

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
    pass

class QidoQueryResultsWrapper(BaseModel):
    result: List[QidoResponse]

class ModalityLUTSequenceItem(DicomResponseBase):
    LUTDescriptor: Optional[List[int]] = None
    ModalityLUTType: Optional[str] = None
    LUTExplanation: Optional[Dict[str, Any]] = None

    @field_validator('LUTExplanation', mode='before')
    @classmethod
    def parse_lut_explanation(cls, v: Any, info: ValidationInfo) -> Optional[Dict[str, Any]]:
        if isinstance(v, str):
            explanation_str = v
            kerma_match = re.search(r"Kerma\\s*(?P<unit>[a-zA-Z]+)\\s*\\(SF=(?P<dfd>\\d+)\\)", explanation_str)
            in_calib_range_match = re.search(r"InCalibRange:(?P<min>\\d+\\.\\d+)-(?P<max>\\d+\\.\\d+)", explanation_str)
            out_lut_range_match = re.search(r"OutLUTRange:(?P<min>\\d+)-(?P<max>\\d+)", explanation_str)
            parsed_data = {}
            if kerma_match:
                parsed_data["Kerma"] = {"Unidad": kerma_match.group("unit"), "SF": int(kerma_match.group("dfd"))}
            if in_calib_range_match:
                parsed_data["InCalibRange"] = {"min": float(in_calib_range_match.group("min")), "max": float(in_calib_range_match.group("max"))}
            if out_lut_range_match:
                parsed_data["OutLUTRange"] = {"min": int(out_lut_range_match.group("min")), "max": int(out_lut_range_match.group("max"))}
            return parsed_data
        return v

class ModalityLUTSequenceModel(DicomResponseBase):
    ModalityLUTSequence: Optional[List[ModalityLUTSequenceItem]] = None

class MtfResultDetail(BaseModel):
    status: str
    roi_id: str
    pixel_spacing: float
    angle_deg: Optional[float] = None
    frequencies: Optional[List[float]] = None
    mtf: Optional[List[float]] = None

class MtfSingleInstanceResponse(BaseModel):
    status: str
    sop_instance_uid: str
    error_details: Optional[str] = None
    vertical_mtf_result: Optional[MtfResultDetail] = None
    horizontal_mtf_result: Optional[MtfResultDetail] = None

class MtfSeriesAnalysisResponse(DicomResponseBase):
    status: str
    processed_files_count: int
    valid_vertical_rois: int
    valid_horizontal_rois: int
    combined_poly_coeffs: Optional[List[float]] = None
    fit_r_squared: Optional[float] = None
    fit_rmse: Optional[float] = None
    mtf_at_50_percent: Optional[float] = None
    mtf_at_10_percent: Optional[float] = None
    error_details: Optional[str] = None

class FilteredInstanceResult(DicomResponseBase):
    SOPInstanceUID: Optional[str] = None
    InstanceNumber: Optional[str] = None
    ImageComments: Optional[str] = None
    PatientName: Optional[Any] = None
    StudyDescription: Optional[str] = None

class FilteredInstanceResultsWrapper(BaseModel):
    result: List[FilteredInstanceResult]

class NnpsAnalysisResponse(DicomResponseBase):
    status: str
    num_images_processed: Optional[int] = None
    pixel_spacing_mm: Optional[float] = None
    mean_kerma_uGy: Optional[float] = None
    nnps_1d_radial_freq: Optional[List[float]] = None
    nnps_1d_radial_values: Optional[List[float]] = None
    error_details: Optional[str] = None

class NnpsGroupResult(DicomResponseBase):
    status: str
    kerma_group_uGy: float
    num_images_in_group: int
    pixel_spacing_mm: Optional[float] = None
    mean_kerma_uGy: Optional[float] = None
    nnps_1d_radial_freq: Optional[List[float]] = None
    nnps_1d_radial_values: Optional[List[float]] = None
    error_details: Optional[str] = None

class NnpsSeriesAnalysisResponse(DicomResponseBase):
    status: str
    groups_analyzed: List[NnpsGroupResult] = []
    error_details: Optional[str] = None

class ClassifiedInstance(BaseModel):
    sop_instance_uid: str
    instance_number: Optional[str] = None
    image_comments: Optional[str] = None
    calculated_kerma_uGy: Optional[float] = None

class KermaGroup(BaseModel):
    kerma_group_uGy: float
    instances: List[ClassifiedInstance] = []

class SeriesClassificationResponse(DicomResponseBase):
    status: str
    mtf_instances: List[ClassifiedInstance] = []
    tor_instances: List[ClassifiedInstance] = []
    fdt_kerma_groups: List[KermaGroup] = []
    other_instances: List[ClassifiedInstance] = []
    error_details: Optional[str] = None
# models.py
from pydantic import BaseModel, Field, field_validator, FieldValidationInfo
import re
from typing import Optional, List, Dict, Any, Tuple

# --- BASE MODEL WITH UNIVERSAL AND ROBUST VALIDATOR ---
class DicomResponseBase(BaseModel):
    """
    A base model that automatically converts non-primitive Python data types to strings.
    This is crucial for handling various `pydicom` specific types (e.g., IS, DS, PN, UID)
    and ensuring they are JSON serializable.
    """
    @field_validator('*', mode='before')
    @classmethod
    def convert_non_primitive_types_to_str(cls, v: Any) -> Any:
        """
        Converts any value that is not a basic Python type (str, int, float, list, dict, tuple, None)
        into its string representation.

        Args:
            v: The value to validate and potentially convert.

        Returns:
            The original value if it's a base type, otherwise its string representation.
        """
        base_types = {str, int, float, list, dict, tuple, type(None)}
        if type(v) not in base_types:
            return str(v)
        return v

    class Config:
        from_attributes = True

# --- RESPONSE MODELS INHERITING FROM THE BASE MODEL ---
class StudyResponse(DicomResponseBase):
    """Represents a single study record from a DICOM query."""
    StudyInstanceUID: str
    PatientID: Optional[str] = None
    PatientName: Optional[str] = None
    StudyDate: Optional[str] = None
    StudyDescription: Optional[str] = None
    ModalitiesInStudy: Optional[str] = None
    AccessionNumber: Optional[str] = None

class StudyQueryResultsWrapper(BaseModel):
    """A wrapper for a list of study query results."""
    result: List[StudyResponse]

class SeriesResponse(DicomResponseBase):
    """Represents a single series record from a DICOM query."""
    StudyInstanceUID: str
    SeriesInstanceUID: str
    Modality: Optional[str] = None
    SeriesNumber: Optional[str] = None
    SeriesDescription: Optional[str] = None
    PatientName: Optional[str] = None

class SeriesQueryResultsWrapper(BaseModel):
    """A wrapper for a list of series query results."""
    result: List[SeriesResponse]

class LUTExplanationModel(BaseModel):
    """Represents the parsed explanation of a Modality LUT."""
    FullText: Optional[str] = Field(None)
    Explanation: Optional[str] = Field(None)
    InCalibRange: Optional[Tuple[float, float]] = Field(None)
    OutLUTRange: Optional[Tuple[float, float]] = Field(None)

class InstanceMetadataResponse(BaseModel):
    """Represents metadata for a single DICOM instance."""
    SOPInstanceUID: str
    InstanceNumber: Optional[str] = None
    dicom_headers: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class PixelDataResponse(BaseModel):
    """Represents the extracted pixel data and its properties from a DICOM instance."""
    sop_instance_uid: str
    rows: int
    columns: int
    pixel_array_shape: Tuple[int, ...]
    pixel_array_dtype: str
    pixel_array_preview: Optional[List[List[Any]]] = None
    message: Optional[str] = None

class MoveRequest(BaseModel):
    """Represents a request to move a DICOM entity (study, series, or instance)."""
    study_instance_uid: str
    series_instance_uid: Optional[str] = None
    sop_instance_uid: Optional[str] = None

class MoveRequestItem(BaseModel):
    """Represents a single instance to be moved in a bulk operation."""
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str

class BulkMoveRequest(BaseModel):
    """Represents a request to move a list of DICOM instances."""
    instances_to_move: List[MoveRequestItem]

# --- New proposed models ---
class DicomNodeInfo(BaseModel):
    """Represents information about a configured DICOM node."""
    name: str
    description: str

class DicomNodeListResponse(BaseModel):
    """Response model for listing all configured DICOM nodes."""
    current_node: str
    nodes: List[DicomNodeInfo]

class OperationStatusResponse(BaseModel):
    """Generic response model for operations that return a success status and a message."""
    success: bool
    message: str

class ConnectionVerificationResponse(BaseModel):
    """Response model for the C-ECHO verification tool."""
    message: str

class PatientQueryResult(DicomResponseBase):
    """Represents a single patient record from a DICOM query."""
    PatientID: str
    PatientName: Optional[str] = None
    PatientBirthDate: Optional[str] = None
    PatientSex: Optional[str] = None

class PatientQueryResultsWrapper(BaseModel):
    """A wrapper for a list of patient query results."""
    result: List[PatientQueryResult]

class AttributePresetDetails(BaseModel):
    """Details of the DICOM attributes included in each preset level."""
    minimal: List[str]
    standard: List[str]
    extended: List[str]

class AttributePresetsResponse(BaseModel):
    """Response model for listing available attribute presets."""
    patient: AttributePresetDetails
    study: AttributePresetDetails
    series: AttributePresetDetails
    instance: AttributePresetDetails

class QidoResponse(DicomResponseBase):
    """A flexible response model for QIDO-RS queries, allowing any extra fields."""
    class Config:
        extra = 'allow'

class ModalityLUTSequenceItem(DicomResponseBase):
    """Represents an item within the ModalityLUTSequence."""
    LUTDescriptor: Optional[List[int]] = None
    ModalityLUTType: Optional[str] = None
    LUTExplanation: Optional[Dict[str, Any]] = None

    @field_validator('LUTExplanation', mode='before')
    @classmethod
    def parse_lut_explanation(cls, v: Any, info: FieldValidationInfo) -> Optional[Dict[str, Any]]:
        """
        Parses the string value of LUTExplanation into a structured dictionary.
        """
        if isinstance(v, str):
            explanation_str = v
            
            kerma_match = re.search(r"Kerma\\s*(?P<unit>[a-zA-Z]+)\\s*\\(SF=(?P<dfd>\\d+)\\)", explanation_str)
            in_calib_range_match = re.search(r"InCalibRange:(?P<min>\\d+\\.\\d+)-(?P<max>\\d+\\.\\d+)", explanation_str)
            out_lut_range_match = re.search(r"OutLUTRange:(?P<min>\\d+)-(?P<max>\\d+)", explanation_str)

            parsed_data = {}
            if kerma_match:
                parsed_data["Kerma"] = {
                    "Unidad": kerma_match.group("unit"),
                    "SF": int(kerma_match.group("dfd"))
                }
            if in_calib_range_match:
                parsed_data["InCalibRange"] = {
                    "min": float(in_calib_range_match.group("min")),
                    "max": float(in_calib_range_match.group("max"))
                }
            if out_lut_range_match:
                parsed_data["OutLUTRange"] = {
                    "min": int(out_lut_range_match.group("min")),
                    "max": int(out_lut_range_match.group("max"))
                }
            return parsed_data
        return v

class ModalityLUTSequenceModel(DicomResponseBase):
    """Represents the full ModalityLUTSequence."""
    ModalityLUTSequence: Optional[List[ModalityLUTSequenceItem]] = None

class QidoQueryResultsWrapper(BaseModel):
    """A wrapper for a list of QIDO-RS query results."""
    result: List[QidoResponse]

# --- MTF Analysis Models ---
class MtfResultDetail(BaseModel):
    """Contains the detailed results of a single MTF analysis for one ROI."""
    status: str
    roi_id: str
    pixel_spacing: float
    angle_deg: Optional[float] = None
    frequencies: Optional[List[float]] = None
    mtf: Optional[List[float]] = None

class MtfSingleInstanceResponse(BaseModel):
    """Response for the MTF analysis of a single DICOM instance."""
    status: str
    sop_instance_uid: str
    error_details: Optional[str] = None
    vertical_mtf_result: Optional[MtfResultDetail] = None
    horizontal_mtf_result: Optional[MtfResultDetail] = None

class MtfSeriesAnalysisResponse(DicomResponseBase):
    """Modelo para la respuesta del análisis MTF de una serie."""
    processed_files_count: int
    valid_vertical_rois: int
    valid_horizontal_rois: int
    combined_poly_coeffs: Optional[List[float]] = None
    fit_r_squared: Optional[float] = None
    fit_rmse: Optional[float] = None
    mtf_at_50_percent: Optional[float] = None
    mtf_at_10_percent: Optional[float] = None

    class Config:
        extra = 'allow'

class FilteredInstanceResult(DicomResponseBase):
    """Model for a single DICOM instance with its most relevant fields."""
    SOPInstanceUID: Optional[str] = None
    InstanceNumber: Optional[str] = None
    ImageComments: Optional[str] = None
    PatientName: Optional[Any] = None
    StudyDescription: Optional[str] = None
    
    class Config:
        extra = 'allow'

class FilteredInstanceResultsWrapper(BaseModel):
    """Wrapper for the list of filtered instance results."""
    result: List[FilteredInstanceResult]

class NnpsAnalysisResponse(DicomResponseBase):
    """Modelo para la respuesta del análisis NNPS simplificado."""
    num_images_processed: Optional[int] = None
    pixel_spacing_mm: Optional[float] = None
    mean_kerma_uGy: Optional[float] = None # Kerma promedio calculado de la ROI
    nnps_1d_radial_freq: Optional[List[float]] = None
    nnps_1d_radial_values: Optional[List[float]] = None
    error_details: Optional[str] = None

    class Config:
        extra = 'allow'    

class NnpsGroupResult(DicomResponseBase):
    """Modelo para el resultado del análisis NNPS de un único grupo de Kerma."""
    kerma_group_uGy: float
    num_images_in_group: int
    pixel_spacing_mm: Optional[float] = None
    mean_kerma_uGy: Optional[float] = None
    nnps_1d_radial_freq: Optional[List[float]] = None
    nnps_1d_radial_values: Optional[List[float]] = None

class NnpsSeriesAnalysisResponse(DicomResponseBase):
    """
    Wrapper para la respuesta completa del análisis NNPS de una serie.
    Contiene una lista con los resultados de cada grupo de Kerma analizado.
    """
    groups_analyzed: List[NnpsGroupResult] = [] 

class ClassifiedInstance(BaseModel):
    """Represents a single classified instance with its key attributes."""
    sop_instance_uid: str
    instance_number: Optional[str] = None
    image_comments: Optional[str] = None
    calculated_kerma_uGy: Optional[float] = None

class KermaGroup(BaseModel):
    """Represents a group of instances with a similar calculated Kerma value."""
    kerma_group_uGy: float
    instances: List[ClassifiedInstance] = []

class SeriesClassificationResponse(DicomResponseBase):
    """
    Provides a classification of all instances in a series by their ImageComments
    and, for FDT images, groups them by common Kerma values.
    """
    mtf_instances: List[ClassifiedInstance] = []
    tor_instances: List[ClassifiedInstance] = []
    fdt_kerma_groups: List[KermaGroup] = []
    other_instances: List[ClassifiedInstance] = []           

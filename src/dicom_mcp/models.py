# models.py
from pydantic import BaseModel, Field, field_validator, FieldValidationInfo
import re
from typing import Optional, List, Dict, Any, Tuple

# --- BASE MODEL WITH UNIVERSAL AND ROBUST VALIDATOR ---
class DicomResponseBase(BaseModel):
    """
    A base model that automatically converts non-primitive Python data types to strings.
    This generically handles all special pydicom types.
    """
    @field_validator('*', mode='before')
    @classmethod
    def convert_non_primitive_types_to_str(cls, v: Any) -> Any:
        # --- CORRECTED LOGIC ---
        # Define a set with the exact basic types we don't want to touch.
        base_types = {str, int, float, list, dict, tuple, type(None)}
        
        # Check if the EXACT TYPE of the value is not in our list of base types.
        if type(v) not in base_types:
            # If it's a special pydicom type (IS, DS, PN, UID, etc.),
            # we convert it to a pure string for Pydantic.
            return str(v)
        
        # If it's already a basic type, we return it unchanged.
        return v

    class Config:
        from_attributes = True

# --- RESPONSE MODELS INHERITING FROM THE BASE MODEL ---
# No changes needed, as they inherit the correct logic.

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

# --- The rest of the models do not need changes ---
class LUTExplanationModel(BaseModel):
    FullText: Optional[str] = Field(None)
    Explanation: Optional[str] = Field(None)
    InCalibRange: Optional[Tuple[float, float]] = Field(None)
    OutLUTRange: Optional[Tuple[float, float]] = Field(None)

class InstanceMetadataResponse(BaseModel):
    SOPInstanceUID: str
    InstanceNumber: Optional[str] = None
    # Change to Optional and default value to None
    dicom_headers: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True # Ensure this is present if you use validation from object attributes

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

# --- New proposed models ---

# For list_dicom_nodes
class DicomNodeInfo(BaseModel):
    name: str
    description: str

class DicomNodeListResponse(BaseModel):
    current_node: str
    nodes: List[DicomNodeInfo]

# For switch_dicom_node
class OperationStatusResponse(BaseModel):
    success: bool
    message: str

# For verify_connection
class ConnectionVerificationResponse(BaseModel):
    message: str

# For query_patients
class PatientQueryResult(DicomResponseBase):
    PatientID: str
    PatientName: Optional[str] = None
    PatientBirthDate: Optional[str] = None
    PatientSex: Optional[str] = None

class PatientQueryResultsWrapper(BaseModel):
    result: List[PatientQueryResult]
    # Add other common patient fields you expect in the response

# For get_attribute_presets
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

class ModalityLUTSequenceItem(DicomResponseBase):
    LUTDescriptor: Optional[List[int]] = None
    ModalityLUTType: Optional[str] = None
    LUTExplanation: Optional[Dict[str, Any]] = None

    @field_validator('LUTExplanation', mode='before')
    @classmethod
    def parse_lut_explanation(cls, v: Any, info: FieldValidationInfo) -> Optional[Dict[str, Any]]:
        if isinstance(v, str):
            explanation_str = v
            
            kerma_match = re.search(r"Kerma\s*(?P<unit>[a-zA-Z]+)\s*\(SF=(?P<dfd>\d+)\)", explanation_str)
            in_calib_range_match = re.search(r"InCalibRange:(?P<min>\d+\.\d+)-(?P<max>\d+\.\d+)", explanation_str)
            out_lut_range_match = re.search(r"OutLUTRange:(?P<min>\d+)-(?P<max>\d+)", explanation_str)

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
    ModalityLUTSequence: Optional[List[ModalityLUTSequenceItem]] = None

class QidoQueryResultsWrapper(BaseModel):
    result: List[QidoResponse]
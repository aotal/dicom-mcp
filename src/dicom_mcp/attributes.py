"""
DICOM attribute presets for different query levels.

This module defines sets of DICOM attributes that can be requested during
C-FIND queries. Presets for 'minimal', 'standard', and 'extended' levels
of detail are provided for patient, study, series, and instance queries.
"""

from typing import Dict, List, Optional

ATTRIBUTE_PRESETS = {
    "minimal": {
        "patient": [
            "PatientID",
            "PatientName",
        ],
        "study": [
            "StudyInstanceUID",
            "PatientID",
            "StudyDate",
            "StudyDescription",
        ],
        "series": [
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "SeriesNumber",
        ],
        "instance": [
            "SOPInstanceUID",
            "SeriesInstanceUID",
            "InstanceNumber",
        ],
    },
    
    "standard": {
        "patient": [
            "PatientID",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "PatientAge",
        ],
        "study": [
            "StudyInstanceUID",
            "PatientID",
            "PatientName",
            "StudyDate",
            "StudyTime",
            "StudyDescription",
            "AccessionNumber",
            "ReferringPhysicianName",
            "StudyID",
            "ModalitiesInStudy",
            "NumberOfStudyRelatedSeries",
            "NumberOfStudyRelatedInstances",
        ],
        "series": [
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "SeriesNumber",
            "SeriesDescription",
            "BodyPartExamined",
            "PatientPosition",
            "PatientName",
            "NumberOfSeriesRelatedInstances",
        ],
        "instance": [
            "SOPInstanceUID",
            "SeriesInstanceUID",
            "SOPClassUID",
            "InstanceNumber",
            "ContentDate",
            "ContentTime",
            "ImageType",
        ],
    },
    
    "extended": {
        "patient": [
            "PatientID",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "PatientAge",
            "PatientWeight",
            "PatientAddress",
            "PatientComments",
            "IssuerOfPatientID",
            "EthnicGroup",
        ],
        "study": [
            "StudyInstanceUID",
            "PatientID",
            "StudyDate",
            "StudyTime",
            "StudyDescription",
            "AccessionNumber",
            "ReferringPhysicianName",
            "StudyID",
            "ProcedureCodeSequence",
            "NumberOfStudyRelatedSeries",
            "NumberOfStudyRelatedInstances",
            "StudyComments",
            "AdmissionID",
            "ModalitiesInStudy",
            "RequestingPhysician",
            "RequestedProcedureDescription",
        ],
        "series": [
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "SeriesNumber",
            "SeriesDescription",
            "BodyPartExamined",
            "PatientPosition",
            "NumberOfSeriesRelatedInstances",
            "SeriesDate",
            "SeriesTime",
            "PerformingPhysicianName",
            "ProtocolName",
            "OperatorsName",
            "PerformedProcedureStepDescription",
            "AnatomicalOrientationType",
            "InstitutionName",
        ],
        "instance": [
            "SOPInstanceUID",
            "SeriesInstanceUID",
            "SOPClassUID",
            "InstanceNumber",
            "ContentDate",
            "ContentTime",
            "ImageType",
            "AcquisitionDate",
            "AcquisitionTime",
            "ImageComments",
            "BurnedInAnnotation",
            "WindowCenter",
            "WindowWidth",
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "SliceLocation",
            "PixelSpacing",
            "PhotometricInterpretation",
            "BitsAllocated",
            "BitsStored",
        ],
    },
}

def get_attributes_for_level(
    level: str, 
    preset: str = "standard", 
    additional_attrs: Optional[List[str]] = None, 
    exclude_attrs: Optional[List[str]] = None
) -> List[str]:
    """Get the list of attributes for a specific query level and preset.

    This function combines attributes from a specified preset with any additional
    attributes provided, and removes any attributes marked for exclusion.

    Args:
        level: The query level ('patient', 'study', 'series', 'instance').
        preset: The name of the attribute preset ('minimal', 'standard', 'extended').
        additional_attrs: A list of additional attributes to include in the result.
        exclude_attrs: A list of attributes to exclude from the result.

    Returns:
        A list of DICOM attribute names.
    """
    if preset in ATTRIBUTE_PRESETS and level in ATTRIBUTE_PRESETS[preset]:
        attr_list = ATTRIBUTE_PRESETS[preset][level].copy()
    else:
        attr_list = ATTRIBUTE_PRESETS["standard"][level].copy()
    
    if additional_attrs:
        for attr in additional_attrs:
            if attr not in attr_list:
                attr_list.append(attr)
    
    if exclude_attrs:
        attr_list = [attr for attr in attr_list if attr not in exclude_attrs]
    
    return attr_list
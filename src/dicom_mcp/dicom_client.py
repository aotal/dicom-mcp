"""
DICOM Client.

This module provides a clean interface to pynetdicom functionality,
abstracting the details of DICOM networking.
"""
import os
import time
import tempfile
from typing import Dict, List, Any, Tuple

from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, build_role
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelGet,
    StudyRootQueryRetrieveInformationModelGet,
    Verification
)

from .attributes import get_attributes_for_level

class DicomClient:
    """DICOM networking client that handles communication with DICOM nodes."""
    
    def __init__(self, host: str, port: int, calling_aet: str, called_aet: str):
        """Initialize the DICOM client.
        
        Args:
            host: The hostname or IP address of the remote DICOM node.
            port: The port number of the remote DICOM node.
            calling_aet: The Application Entity Title (AET) of this client.
            called_aet: The Application Entity Title (AET) of the remote node.
        """
        self.host = host
        self.port = port
        self.called_aet = called_aet
        self.calling_aet = calling_aet
        
        self.ae = AE(ae_title=calling_aet)
        
        self.ae.add_requested_context(Verification)
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelGet)
        
    
    def verify_connection(self) -> Tuple[bool, str]:
        """Verify connectivity to the DICOM node using a C-ECHO request.
        
        Returns:
            A tuple containing a boolean indicating success and a message.
        """
        assoc = self.ae.associate(self.host, self.port, ae_title=self.called_aet)
        
        if assoc.is_established:
            status = assoc.send_c_echo()
            assoc.release()
            
            if status and status.Status == 0:
                return True, f"Connection successful to {self.host}:{self.port} (Called AE: {self.called_aet}, Calling AE: {self.calling_aet})"
            else:
                return False, f"C-ECHO failed with status: {status.Status if status else 'None'}"
        else:
            return False, f"Failed to associate with DICOM node at {self.host}:{self.port} (Called AE: {self.called_aet}, Calling AE: {self.calling_aet})"
    
    def find(self, query_dataset: Dataset, query_model) -> List[Dict[str, Any]]:
        """Execute a C-FIND request.
        
        Args:
            query_dataset: A pydicom Dataset containing the query parameters.
            query_model: The DICOM query model to use (e.g., PatientRootQueryRetrieveInformationModelFind).
        
        Returns:
            A list of dictionaries, where each dictionary represents a matched record.
        
        Raises:
            Exception: If the association with the DICOM node fails.
        """
        assoc = self.ae.associate(self.host, self.port, ae_title=self.called_aet)
        
        if not assoc.is_established:
            raise Exception(f"Failed to associate with DICOM node at {self.host}:{self.port} (Called AE: {self.called_aet}, Calling AE: {self.calling_aet})")
        
        results = []
        
        try:
            responses = assoc.send_c_find(query_dataset, query_model)
            
            for (status, dataset) in responses:
                if status and status.Status == 0xFF00:  # Pending
                    if dataset:
                        results.append(self._dataset_to_dict(dataset))
        finally:
            assoc.release()
        
        return results
    
    def query_patient(self, patient_id: str = None, name_pattern: str = None, 
                     birth_date: str = None, attribute_preset: str = "standard",
                     additional_attrs: List[str] = None, exclude_attrs: List[str] = None,
                     additional_filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Query for patients matching the specified criteria.
        
        Args:
            patient_id: The patient's ID.
            name_pattern: A pattern for the patient's name (can include wildcards).
            birth_date: The patient's birth date in YYYYMMDD format.
            attribute_preset: The attribute preset to use ('minimal', 'standard', 'extended').
            additional_attrs: A list of additional attributes to include in the result.
            exclude_attrs: A list of attributes to exclude from the result.
            additional_filters: A dictionary of additional DICOM tags to use for filtering.
            
        Returns:
            A list of matching patient records.
        """
        ds = Dataset()
        ds.QueryRetrieveLevel = "PATIENT"
        
        if patient_id:
            ds.PatientID = patient_id
        if name_pattern:
            ds.PatientName = name_pattern
        if birth_date:
            ds.PatientBirthDate = birth_date

        if additional_filters:
            for key, value in additional_filters.items():
                setattr(ds, key, value)
        
        attrs = get_attributes_for_level("patient", attribute_preset, additional_attrs, exclude_attrs)
        for attr in attrs:
            if not hasattr(ds, attr):
                setattr(ds, attr, "")
        
        return self.find(ds, PatientRootQueryRetrieveInformationModelFind)
    
    def query_study(self, patient_id: str = None, study_date: str = None, 
                   modality: str = None, study_description: str = None, 
                   accession_number: str = None, study_instance_uid: str = None,
                   attribute_preset: str = "standard", additional_attrs: List[str] = None, 
                   exclude_attrs: List[str] = None, additional_filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Query for studies matching the specified criteria.
        
        Args:
            patient_id: The patient's ID.
            study_date: The study date or a date range (YYYYMMDD or YYYYMMDD-YYYYMMDD).
            modality: The modality of the study.
            study_description: A pattern for the study description.
            accession_number: The accession number of the study.
            study_instance_uid: The Study Instance UID.
            attribute_preset: The attribute preset to use.
            additional_attrs: A list of additional attributes to include.
            exclude_attrs: A list of attributes to exclude.
            additional_filters: A dictionary of additional DICOM tags for filtering.
            
        Returns:
            A list of matching study records.
        """
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        
        if patient_id:
            ds.PatientID = patient_id
        if study_date:
            ds.StudyDate = study_date
        if modality:
            ds.ModalitiesInStudy = modality
        if study_description:
            ds.StudyDescription = study_description
        if accession_number:
            ds.AccessionNumber = accession_number
        if study_instance_uid:
            ds.StudyInstanceUID = study_instance_uid

        if additional_filters:
            for key, value in additional_filters.items():
                setattr(ds, key, value)
        
        attrs = get_attributes_for_level("study", attribute_preset, additional_attrs, exclude_attrs)
        for attr in attrs:
            if not hasattr(ds, attr):
                setattr(ds, attr, "")
        
        return self.find(ds, StudyRootQueryRetrieveInformationModelFind)
    
    def query_series(self, study_instance_uid: str, series_instance_uid: str = None,
                    modality: str = None, series_number: str = None, 
                    series_description: str = None, attribute_preset: str = "standard",
                    additional_attrs: List[str] = None, exclude_attrs: List[str] = None,
                    additional_filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Query for series matching the specified criteria.
        
        Args:
            study_instance_uid: The Study Instance UID (required).
            series_instance_uid: The Series Instance UID.
            modality: The modality of the series (e.g., 'CT', 'MR').
            series_number: The series number.
            series_description: A pattern for the series description.
            attribute_preset: The attribute preset to use.
            additional_attrs: A list of additional attributes to include.
            exclude_attrs: A list of attributes to exclude.
            additional_filters: A dictionary of additional DICOM tags for filtering.
            
        Returns:
            A list of matching series records.
        """
        ds = Dataset()
        ds.QueryRetrieveLevel = "SERIES"
        ds.StudyInstanceUID = study_instance_uid
        
        if series_instance_uid:
            ds.SeriesInstanceUID = series_instance_uid
        if modality:
            ds.Modality = modality
        if series_number:
            ds.SeriesNumber = series_number
        if series_description:
            ds.SeriesDescription = series_description

        if additional_filters:
            for key, value in additional_filters.items():
                setattr(ds, key, value)
        
        attrs = get_attributes_for_level("series", attribute_preset, additional_attrs, exclude_attrs)
        for attr in attrs:
            if not hasattr(ds, attr):
                setattr(ds, attr, "")
        
        return self.find(ds, StudyRootQueryRetrieveInformationModelFind)
    
    def query_instance(self, series_instance_uid: str, sop_instance_uid: str = None,
                      instance_number: str = None, attribute_preset: str = "standard",
                      additional_attrs: List[str] = None, exclude_attrs: List[str] = None,
                      additional_filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Query for instances matching the specified criteria.
        
        Args:
            series_instance_uid: The Series Instance UID (required).
            sop_instance_uid: The SOP Instance UID.
            instance_number: The instance number.
            attribute_preset: The attribute preset to use.
            additional_attrs: A list of additional attributes to include.
            exclude_attrs: A list of attributes to exclude.
            additional_filters: A dictionary of additional DICOM tags for filtering.
            
        Returns:
            A list of matching instance records.
        """
        ds = Dataset()
        ds.QueryRetrieveLevel = "IMAGE"
        ds.SeriesInstanceUID = series_instance_uid
        
        if sop_instance_uid:
            ds.SOPInstanceUID = sop_instance_uid
        if instance_number:
            ds.InstanceNumber = instance_number

        if additional_filters:
            for key, value in additional_filters.items():
                setattr(ds, key, value)
        
        attrs = get_attributes_for_level("instance", attribute_preset, additional_attrs, exclude_attrs)
        for attr in attrs:
            if not hasattr(ds, attr):
                setattr(ds, attr, "")
        
        return self.find(ds, StudyRootQueryRetrieveInformationModelFind)

    @staticmethod
    def _dataset_to_dict(dataset: Dataset) -> Dict[str, Any]:
        """Convert a pydicom Dataset to a dictionary.
        
        Args:
            dataset: The pydicom Dataset to convert.
            
        Returns:
            A dictionary representation of the dataset.
        """
        if hasattr(dataset, "is_empty") and dataset.is_empty():
            return {}
        
        result = {}
        for elem in dataset:
            if elem.VR == "SQ":
                result[elem.keyword] = [DicomClient._dataset_to_dict(item) for item in elem.value]
            else:
                if hasattr(elem, "keyword"):
                    try:
                        if elem.VM > 1:
                            result[elem.keyword] = list(elem.value)
                        else:
                            result[elem.keyword] = elem.value
                    except Exception:
                        result[elem.keyword] = str(elem.value)
        
        return result
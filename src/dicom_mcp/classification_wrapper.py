# src/dicom_mcp/classification_wrapper.py

import numpy as np
import pydicom
from typing import List, Dict, Any
from collections import defaultdict

def _get_pixel_spacing(ds: pydicom.Dataset) -> float:
    """
    Robustly gets the pixel spacing (in mm) from the DICOM dataset.
    Searches common tags in order of preference.
    """
    for tag in ["PixelSpacing", "ImagerPixelSpacing", "SpatialResolution", "NominalScannedPixelSpacing"]:
        if tag in ds:
            value = ds.get(tag)
            # Handle both single and multi-valued tags
            return float(value[0] if isinstance(value, pydicom.multival.MultiValue) else value)
    raise ValueError("Could not find a valid pixel spacing tag in the dataset.")

def _apply_rescale(ds: pydicom.Dataset) -> np.ndarray:
    """Applies Rescale Slope and Intercept to get pixel values in µGy."""
    pixel_array = ds.pixel_array.astype(np.float64)
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    return pixel_array * slope + intercept

def _calculate_kerma_from_roi(image_in_uGy: np.ndarray, pixel_spacing_mm: float) -> float:
    """Calculates the average Kerma from a central 4x4 cm ROI."""
    roi_size_mm = 40.0
    roi_half_pixels = int(roi_size_mm / (2 * pixel_spacing_mm))
    center_y, center_x = image_in_uGy.shape[0] // 2, image_in_uGy.shape[1] // 2
    roi = image_in_uGy[
        center_y - roi_half_pixels : center_y + roi_half_pixels,
        center_x - roi_half_pixels : center_x + roi_half_pixels,
    ]
    return np.mean(roi)

def classify_instances(datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
    """
    Classifies a list of DICOM datasets by their ImageComments and groups FDT images by Kerma.
    """
    if not datasets:
        return {"mtf_instances": [], "tor_instances": [], "fdt_kerma_groups": [], "other_instances": []}

    # Dictionaries to hold classified instances
    classified_data = { "MTF": [], "TOR": [], "FDT": [], "Other": [] }
    
    pixel_spacing_mm = _get_pixel_spacing(datasets[0]) # Assume consistent spacing across all datasets

    for ds in datasets:
        comments = ds.get("ImageComments", "Other").strip()
        instance_number = ds.get("InstanceNumber", None)
        sop_uid = ds.get("SOPInstanceUID", "Unknown")
        
        instance_info = {
            "sop_instance_uid": str(sop_uid),
            "instance_number": str(instance_number) if instance_number else None,
            "image_comments": comments,
            "calculated_kerma_uGy": None
        }

        if comments == "FDT":
            image_in_uGy = _apply_rescale(ds)
            kerma = _calculate_kerma_from_roi(image_in_uGy, pixel_spacing_mm)
            instance_info["calculated_kerma_uGy"] = kerma
            classified_data["FDT"].append(instance_info)
        elif comments in ["MTF", "TOR"]:
            classified_data[comments].append(instance_info)
        else:
            classified_data["Other"].append(instance_info)

    # Group FDT instances by Kerma, rounded to the first decimal place
    kerma_groups_dict = defaultdict(list)
    for fdt_instance in classified_data["FDT"]:
        if fdt_instance["calculated_kerma_uGy"] is not None:
            kerma_key = round(fdt_instance["calculated_kerma_uGy"], 1)
            kerma_groups_dict[kerma_key].append(fdt_instance)
    
    # Format the groups for the final response
    fdt_kerma_groups = []
    for key, instances in kerma_groups_dict.items():
        fdt_kerma_groups.append({
            "kerma_group_uGy": key,
            "instances": instances
        })

    return {
        "mtf_instances": classified_data["MTF"],
        "tor_instances": classified_data["TOR"],
        "fdt_kerma_groups": fdt_kerma_groups,
        "other_instances": classified_data["Other"]
    }

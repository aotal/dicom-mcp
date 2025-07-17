# src/dicom_mcp/nnps_processor_wrapper.py

import numpy as np
import pydicom
from scipy.fft import fft2, fftshift
from scipy.stats import binned_statistic
from typing import List, Dict, Any
from collections import defaultdict

# --- INTERNAL HELPER FUNCTIONS ---

def _get_pixel_spacing(ds: pydicom.Dataset) -> float:
    """Robustly retrieves the pixel spacing (in mm) from the DICOM dataset."""
    for tag in ["PixelSpacing", "ImagerPixelSpacing", "SpatialResolution", "NominalScannedPixelSpacing"]:
        if tag in ds:
            value = ds.get(tag)
            return float(value[0] if isinstance(value, pydicom.multival.MultiValue) else value)
    raise ValueError("Could not find a valid pixel spacing tag.")

def _apply_rescale(ds: pydicom.Dataset) -> np.ndarray:
    """Applies Rescale Slope and Intercept to pixel data to obtain µGy."""
    pixel_array = ds.pixel_array.astype(np.float64)
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    return pixel_array * slope + intercept

def _calculate_kerma_from_roi(image_in_uGy: np.ndarray, pixel_spacing_mm: float) -> float:
    """Calculates the average Kerma from a central 4x4 cm ROI."""
    roi_size_mm = 40.0
    roi_half_pixels = int(roi_size_mm / (2 * pixel_spacing_mm))
    center_y, center_x = image_in_uGy.shape[0] // 2, image_in_uGy.shape[1] // 2
    roi = image_in_uGy[center_y - roi_half_pixels : center_y + roi_half_pixels, center_x - roi_half_pixels : center_x + roi_half_pixels]
    return np.mean(roi)

def _calculate_nnps_for_group_logic(group_images: List[np.ndarray], pixel_spacing_mm: float, mean_kerma_uGy: float) -> Dict[str, Any]:
    """Contains the mathematical logic to calculate the NNPS for a group of images."""
    ref_shape = group_images[0].shape
    nps_2d_sum = np.zeros(ref_shape, dtype=np.float64)
    for img in group_images:
        img_detrended = img - np.mean(img)
        fft_2d = fftshift(fft2(img_detrended))
        nps_2d_sum += np.abs(fft_2d)**2
    avg_nps_2d = nps_2d_sum / len(group_images)
    
    if mean_kerma_uGy < 1e-9: raise ValueError("Average Kerma is close to zero.")
    
    pixel_area = pixel_spacing_mm**2
    num_pixels = ref_shape[0] * ref_shape[1]
    nnps_2d = (avg_nps_2d * pixel_area) / (num_pixels * mean_kerma_uGy**2)

    freq_x = fftshift(np.fft.fftfreq(ref_shape[1], d=pixel_spacing_mm))
    freq_y = fftshift(np.fft.fftfreq(ref_shape[0], d=pixel_spacing_mm))
    fx_grid, fy_grid = np.meshgrid(freq_x, freq_y)
    fr_grid = np.sqrt(fx_grid**2 + fy_grid**2)
    
    df = freq_x[1] - freq_x[0] if len(freq_x) > 1 else 0
    num_bins = int(np.ceil(np.max(fr_grid) / df)) if df > 0 else 1
    bin_edges = np.linspace(0, num_bins * df, num_bins + 1)

    nnps_radial_mean, _, _ = binned_statistic(fr_grid.ravel(), nnps_2d.ravel(), statistic='mean', bins=bin_edges)
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0

    nan_mask = np.isnan(nnps_radial_mean)
    if np.any(nan_mask):
        nnps_radial_mean[nan_mask] = np.interp(bin_centers[nan_mask], bin_centers[~nan_mask], nnps_radial_mean[~nan_mask])

    return {"nnps_1d_radial_freq": bin_centers.tolist(), "nnps_1d_radial_values": nnps_radial_mean.tolist()}

# --- PROCESSING FUNCTIONS (Entry points for tools) ---

def process_nnps_for_group(datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
    """
    Calculates the NNPS for a single group of datasets, assuming they have a similar Kerma.
    """
    if not datasets or len(datasets) < 2:
        return {"status": "Error", "error_details": "At least 2 images are required for group analysis."}
    
    try:
        pixel_spacing_mm = _get_pixel_spacing(datasets[0])
        images_in_uGy = [_apply_rescale(ds) for ds in datasets]
        
        ref_shape = images_in_uGy[0].shape
        for img in images_in_uGy[1:]:
            if img.shape != ref_shape: raise ValueError("Images in the group have different dimensions.")

        mean_kerma_uGy = np.mean([_calculate_kerma_from_roi(img, pixel_spacing_mm) for img in images_in_uGy])
        nnps_results = _calculate_nnps_for_group_logic(images_in_uGy, pixel_spacing_mm, mean_kerma_uGy)

        return {"status": "OK", "num_images_processed": len(datasets), "pixel_spacing_mm": pixel_spacing_mm, "mean_kerma_uGy": mean_kerma_uGy, **nnps_results, "error_details": None}
    except Exception as e:
        return {"status": "Error", "error_details": str(e)}

def process_series_for_nnps(datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
    """
    Groups datasets by Kerma and calculates the NNPS for each valid group.
    """
    if not datasets:
        return {"status": "Error", "error_details": "The list of datasets is empty."}

    try:
        pixel_spacing_mm = _get_pixel_spacing(datasets[0])
        kerma_groups = defaultdict(list)
        for ds in datasets:
            image_in_uGy = _apply_rescale(ds)
            kerma = _calculate_kerma_from_roi(image_in_uGy, pixel_spacing_mm)
            kerma_key = round(kerma, 1)
            kerma_groups[kerma_key].append(image_in_uGy)

        analysis_results = []
        for kerma_key, images_in_group in kerma_groups.items():
            if len(images_in_group) < 2: continue

            mean_kerma_of_group = np.mean([np.mean(img) for img in images_in_group])
            nnps_results = _calculate_nnps_for_group_logic(images_in_group, pixel_spacing_mm, mean_kerma_of_group)
            
            analysis_results.append({"status": "OK", "kerma_group_uGy": float(kerma_key), "num_images_in_group": len(images_in_group), "pixel_spacing_mm": pixel_spacing_mm, "mean_kerma_uGy": mean_kerma_of_group, **nnps_results})

        return {"status": "OK", "groups_analyzed": analysis_results}
    except Exception as e:
        return {"status": "Error", "error_details": str(e)}

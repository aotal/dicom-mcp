# src/dicom_mcp/mtf_processor_wrapper.py

import numpy as np
import pydicom
from typing import List, Dict, Any

# Import analysis classes from the MTF subfolder
from .MTF.roi_extractor import RoiExtractor
from .MTF.mtf_analyzer import MtfAnalyzer
from .MTF.utils import apply_dicom_linearity

def _get_pixel_spacing(ds: pydicom.Dataset) -> float:
    """
    Robustly retrieves the pixel spacing (in mm) from a DICOM dataset.
    Searches common tags in order of preference.
    """
    # 1. Pixel Spacing (0028,0030) - Most common for projection images
    if "PixelSpacing" in ds:
        spacing = ds.PixelSpacing
        # It's a multi-valued tag (Row, Column), we assume square pixels
        return float(spacing[0])
    
    # 2. Imager Pixel Spacing (0018,1164) - Common in detectors
    if "ImagerPixelSpacing" in ds:
        spacing = ds.ImagerPixelSpacing
        return float(spacing[0])
        
    # 3. Spatial Resolution (0018,1050)
    if "SpatialResolution" in ds:
        # This tag is usually a single value
        return float(ds.SpatialResolution)

    # 4. Nominal Scanned Pixel Spacing (0018,2010) - Another alternative
    if "NominalScannedPixelSpacing" in ds:
        spacing = ds.NominalScannedPixelSpacing
        return float(spacing[0])

    raise ValueError("Could not find a valid pixel spacing tag (PixelSpacing, ImagerPixelSpacing, SpatialResolution, etc.).")

def process_mtf_from_datasets(datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
    """
    Orchestrates the MTF calculation process from a list of DICOM datasets.
    This function extracts pixel spacing, linearizes the image, and averages MTF results.
    """
    if not datasets:
        return {"status": "Error", "error_details": "The list of datasets is empty."}

    try:
        # 1. Extract Pixel Spacing from the first dataset and validate consistency
        pixel_spacing_mm = _get_pixel_spacing(datasets[0])
        ref_shape = datasets[0].pixel_array.shape
        
        vertical_mtf_results = []
        horizontal_mtf_results = []
        processed_files_count = 0

        # Parameters for ROI extraction and MTF analysis
        roi1_offset_mm = (13, 0)
        roi1_shape_yx = (100, 200)
        roi2_offset_mm = (0, -14)
        roi2_shape_yx = (200, 100)
        mtf_params = {'sub_pixel_factor': 0.1, 'smoothing_window_bins': 17, 'smoothing_polyorder': 4}
        analyzer = MtfAnalyzer(**mtf_params)

        for i, ds in enumerate(datasets):
            # Validate consistency of spacing and dimensions
            if _get_pixel_spacing(ds) != pixel_spacing_mm:
                raise ValueError(f"Inconsistent Pixel Spacing in instance #{i+1}.")
            if ds.pixel_array.shape != ref_shape:
                raise ValueError(f"Images have different dimensions in instance #{i+1}.")

            # 2. Apply linearization (RescaleSlope/Intercept)
            linearized_image = apply_dicom_linearity(ds)

            # 3. Extract ROIs
            extractor = RoiExtractor(linearized_image, pixel_spacing_mm, verbose=False)
            rois = extractor.extract_mtf_rois(roi1_offset_mm, roi1_shape_yx, roi2_offset_mm, roi2_shape_yx)
            
            if not rois or len(rois) != 2:
                print(f"Warning: Could not extract ROIs for instance #{i+1}. Skipping.")
                continue

            # 4. Analyze each ROI to obtain MTF curves
            res_v = analyzer.analyze_roi(rois[0], pixel_spacing_mm, roi_id=f"inst_{i+1}_v", verbose=False)
            if res_v and res_v.get("status") == "OK":
                vertical_mtf_results.append(res_v)
            
            res_h = analyzer.analyze_roi(rois[1], pixel_spacing_mm, roi_id=f"inst_{i+1}_h", verbose=False)
            if res_h and res_h.get("status") == "OK":
                horizontal_mtf_results.append(res_h)
            
            processed_files_count += 1

        # 5. Average the results and return
        valid_v_rois = len(vertical_mtf_results)
        valid_h_rois = len(horizontal_mtf_results)

        if valid_v_rois == 0 and valid_h_rois == 0:
            return {"status": "Error", "error_details": "Could not obtain any valid MTF results."}

        # Averaging logic (similar to NNPS)
        vert_avg, horiz_avg, combined_avg, combined_freq = None, None, None, None

        if valid_v_rois > 0:
            vert_freq, vert_avg, _, _ = MtfAnalyzer.calculate_grand_average_mtf(vertical_mtf_results)
            if vert_freq is not None: combined_freq = vert_freq
        
        if valid_h_rois > 0:
            horiz_freq, horiz_avg, _, _ = MtfAnalyzer.calculate_grand_average_mtf(horizontal_mtf_results)
            if combined_freq is None and horiz_freq is not None: combined_freq = horiz_freq
        
        if vert_avg is not None and horiz_avg is not None:
            mtf1_interp = np.interp(combined_freq, vert_freq, vert_avg)
            mtf2_interp = np.interp(combined_freq, horiz_freq, horiz_avg)
            combined_avg = (mtf1_interp + mtf2_interp) / 2.0
        elif vert_avg is not None: combined_avg = vert_avg
        elif horiz_avg is not None: combined_avg = horiz_avg

        if combined_avg is None:
            raise ValueError("Failed to combine MTF curves.")

        coeffs, _, fit_stats = analyzer.fit_average_mtf_polynomial(combined_freq, combined_avg, degree=4)
        
        # --- MTF CALCULATION AT 50% AND 10% ---
        freq_at_50_mtf = np.interp(0.5, combined_avg[::-1], combined_freq[::-1])
        freq_at_10_mtf = np.interp(0.1, combined_avg[::-1], combined_freq[::-1]) # New line

        return {
            "status": "OK",
            "processed_files_count": processed_files_count,
            "valid_vertical_rois": valid_v_rois,
            "valid_horizontal_rois": valid_h_rois,
            "combined_poly_coeffs": coeffs.tolist() if coeffs is not None else None,
            "fit_r_squared": fit_stats.get('r_squared'),
            "fit_rmse": fit_stats.get('rmse'),
            "mtf_at_50_percent": freq_at_50_mtf,
            "mtf_at_10_percent": freq_at_10_mtf, # New field in the result
            "error_details": None
        }

    except Exception as e:
        return {"status": "Error", "error_details": str(e)}

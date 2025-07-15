# src/dicom_mcp/mtf_processor_wrapper.py (VERSIÓN FINAL)

import numpy as np
import pydicom
from typing import List, Dict, Any

from .MTF.roi_extractor import RoiExtractor
from .MTF.mtf_analyzer import MtfAnalyzer
from .MTF.utils import apply_dicom_linearity

# --- Constantes (sin cambios) ---
ROI1_OFFSET_MM = (13, 0)
ROI1_SHAPE_YX = (100, 200)
ROI2_OFFSET_MM = (0, -14)
ROI2_SHAPE_YX = (200, 100)
ROI1_LABEL = "Vertical"
ROI2_LABEL = "Horizontal"
MTF_PARAMS = {
    'sub_pixel_factor': 0.1, 'smoothing_window_bins': 17, 'smoothing_polyorder': 4,
    'baseline_tail_threshold_mm': 7.0, 'window_width_mm': 20.0, 'cutoff_freq': 3.7,
}
NUM_POINTS_INTERP = 250

def _get_pixel_spacing(ds: pydicom.Dataset) -> float:
    spacing = ds.get("PixelSpacing", ds.get("ImagerPixelSpacing"))
    if spacing is None: raise ValueError("No se pudo encontrar el tag PixelSpacing o ImagerPixelSpacing.")
    if len(spacing) > 1 and abs(float(spacing[0]) - float(spacing[1])) > 1e-6:
        print(f"Advertencia: Píxeles anisotrópicos detectados ({spacing}). Se usará el primer valor.")
    return float(spacing[0])

def process_mtf_from_datasets(datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
    """
    Orquesta el cálculo de MTF, manejando correctamente tanto una como múltiples instancias.
    """
    # ... (código de inicialización y bucle principal sin cambios) ...
    if not datasets:
        return {"status": "Error", "error_details": "La lista de datasets de entrada está vacía.", "processed_files_count": 0, "valid_vertical_rois": 0, "valid_horizontal_rois": 0, "combined_poly_coeffs": None, "fit_r_squared": None, "fit_rmse": None, "mtf_at_50_percent": None}
    analyzer = MtfAnalyzer(**MTF_PARAMS)
    vertical_mtf_results, horizontal_mtf_results, processed_files_count = [], [], 0
    for i, ds in enumerate(datasets):
        sop_uid = ds.get("SOPInstanceUID", f"Dataset_{i+1}")
        try:
            pixel_spacing = _get_pixel_spacing(ds)
            linearized_pixel_array = apply_dicom_linearity(ds)
            extractor = RoiExtractor(linearized_pixel_array, pixel_spacing, verbose=False)
            rois = extractor.extract_mtf_rois(ROI1_OFFSET_MM, ROI1_SHAPE_YX, ROI2_OFFSET_MM, ROI2_SHAPE_YX)
            if not rois or len(rois) != 2 or rois[0].size == 0 or rois[1].size == 0: raise ValueError("Fallo al extraer ROIs.")
            results1 = analyzer.analyze_roi(rois[0], pixel_spacing, roi_id=f"{sop_uid}-{ROI1_LABEL}", verbose=False)
            if results1 and "OK" in results1.get("status", ""): vertical_mtf_results.append(results1)
            results2 = analyzer.analyze_roi(rois[1], pixel_spacing, roi_id=f"{sop_uid}-{ROI2_LABEL}", verbose=False)
            if results2 and "OK" in results2.get("status", ""): horizontal_mtf_results.append(results2)
            processed_files_count += 1
        except Exception as e: print(f"ERROR procesando {sop_uid}: {e}")

    # --- LÓGICA DE ANÁLISIS FINAL CORREGIDA ---
    valid_v_rois, valid_h_rois = len(vertical_mtf_results), len(horizontal_mtf_results)
    if valid_v_rois == 0 and valid_h_rois == 0:
        return {"status": "Error", "error_details": "No se obtuvieron resultados MTF válidos.", "processed_files_count": processed_files_count, "valid_vertical_rois": 0, "valid_horizontal_rois": 0, "combined_poly_coeffs": None, "fit_r_squared": None, "fit_rmse": None, "mtf_at_50_percent": None}

    try:
        vert_freq, vert_avg, horiz_freq, horiz_avg, combined_avg, combined_freq = None, None, None, None, None, None

        # --- LÓGICA INTELIGENTE PARA UNA O VARIAS CURVAS ---
        if valid_v_rois == 1:
            vert_freq, vert_avg = vertical_mtf_results[0]['frequencies'], vertical_mtf_results[0]['mtf']
        elif valid_v_rois > 1:
            vert_freq, vert_avg, _, _ = MtfAnalyzer.calculate_grand_average_mtf(vertical_mtf_results, num_points=NUM_POINTS_INTERP, forced_cutoff_freq=MTF_PARAMS['cutoff_freq'])
        
        if valid_h_rois == 1:
            horiz_freq, horiz_avg = horizontal_mtf_results[0]['frequencies'], horizontal_mtf_results[0]['mtf']
        elif valid_h_rois > 1:
            horiz_freq, horiz_avg, _, _ = MtfAnalyzer.calculate_grand_average_mtf(horizontal_mtf_results, num_points=NUM_POINTS_INTERP, forced_cutoff_freq=MTF_PARAMS['cutoff_freq'])
        # --------------------------------------------------

        if vert_avg is not None: combined_freq = vert_freq
        elif horiz_avg is not None: combined_freq = horiz_freq

        if vert_avg is not None and horiz_avg is not None:
            # Interpolar a un eje común si es necesario antes de promediar
            mtf1_interp = np.interp(combined_freq, vert_freq, vert_avg)
            mtf2_interp = np.interp(combined_freq, horiz_freq, horiz_avg)
            combined_avg = (mtf1_interp + mtf2_interp) / 2.0
        elif vert_avg is not None: combined_avg = vert_avg
        elif horiz_avg is not None: combined_avg = horiz_avg

        if combined_avg is None or combined_freq is None: raise ValueError("Fallo al combinar las curvas MTF.")

        coeffs, _, fit_stats = analyzer.fit_average_mtf_polynomial(combined_freq, combined_avg, degree=4)
        if coeffs is None: raise ValueError("El ajuste polinómico falló.")
            
        freq_at_50_mtf = np.interp(0.5, combined_avg[::-1], combined_freq[::-1])

        return {"status": "OK", "processed_files_count": processed_files_count, "valid_vertical_rois": valid_v_rois, "valid_horizontal_rois": valid_h_rois, "combined_poly_coeffs": coeffs.tolist(), "fit_r_squared": fit_stats.get('r_squared'), "fit_rmse": fit_stats.get('rmse'), "mtf_at_50_percent": freq_at_50_mtf, "error_details": None}

    except Exception as e:
        return {"status": "Error", "error_details": f"Fallo en el análisis final y promediado: {e}", "processed_files_count": processed_files_count, "valid_vertical_rois": valid_v_rois, "valid_horizontal_rois": valid_h_rois, "combined_poly_coeffs": None, "fit_r_squared": None, "fit_rmse": None, "mtf_at_50_percent": None}
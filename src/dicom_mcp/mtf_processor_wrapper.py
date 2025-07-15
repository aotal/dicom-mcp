# src/dicom_mcp/mtf_processor_wrapper.py (VERSIÓN FINAL Y ROBUSTA)

import numpy as np
import pydicom
from typing import List, Dict, Any

# Importamos las clases de análisis desde la subcarpeta MTF
from .MTF.roi_extractor import RoiExtractor
from .MTF.mtf_analyzer import MtfAnalyzer
from .MTF.utils import apply_dicom_linearity

def _get_pixel_spacing(ds: pydicom.Dataset) -> float:
    """
    Obtiene el espaciado de píxeles (en mm) del dataset DICOM de forma robusta.
    Busca en los tags más comunes en orden de preferencia.
    """
    # 1. Pixel Spacing (0028,0030) - El más común para imágenes proyectadas
    if "PixelSpacing" in ds:
        spacing = ds.PixelSpacing
        # Es un tag multi-valor (Row, Column), asumimos píxeles cuadrados
        return float(spacing[0])
    
    # 2. Imager Pixel Spacing (0018,1164) - Común en detectores
    if "ImagerPixelSpacing" in ds:
        spacing = ds.ImagerPixelSpacing
        return float(spacing[0])
        
    # 3. Spatial Resolution (0018,1050) - Como mencionaste
    if "SpatialResolution" in ds:
        # Este tag suele ser un valor único
        return float(ds.SpatialResolution)

    # 4. Nominal Scanned Pixel Spacing (0018,2010) - Otra alternativa
    if "NominalScannedPixelSpacing" in ds:
        spacing = ds.NominalScannedPixelSpacing
        return float(spacing[0])

    raise ValueError("No se pudo encontrar un tag de espaciado de píxeles válido (PixelSpacing, ImagerPixelSpacing, SpatialResolution, etc.).")

def process_mtf_from_datasets(datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
    """
    Orquesta el proceso de cálculo de MTF a partir de una lista de datasets DICOM.
    Extrae el Pixel Spacing, linealiza la imagen y promedia los resultados de MTF.
    """
    if not datasets:
        return {"status": "Error", "error_details": "La lista de datasets está vacía."}

    try:
        # 1. Extraer Pixel Spacing del primer dataset y validar consistencia
        pixel_spacing_mm = _get_pixel_spacing(datasets[0])
        ref_shape = datasets[0].pixel_array.shape
        
        vertical_mtf_results = []
        horizontal_mtf_results = []
        processed_files_count = 0

        # Parámetros para la extracción de ROI y análisis MTF
        roi1_offset_mm = (13, 0)
        roi1_shape_yx = (100, 200)
        roi2_offset_mm = (0, -14)
        roi2_shape_yx = (200, 100)
        mtf_params = {'sub_pixel_factor': 0.1, 'smoothing_window_bins': 17, 'smoothing_polyorder': 4}
        analyzer = MtfAnalyzer(**mtf_params)

        for i, ds in enumerate(datasets):
            # Validar consistencia de spacing y dimensiones
            if _get_pixel_spacing(ds) != pixel_spacing_mm:
                raise ValueError(f"Inconsistencia en Pixel Spacing en la instancia #{i+1}.")
            if ds.pixel_array.shape != ref_shape:
                raise ValueError(f"Las imágenes tienen diferentes dimensiones en la instancia #{i+1}.")

            # 2. Aplicar linealización (RescaleSlope/Intercept)
            linearized_image = apply_dicom_linearity(ds)

            # 3. Extraer ROIs
            extractor = RoiExtractor(linearized_image, pixel_spacing_mm, verbose=False)
            rois = extractor.extract_mtf_rois(roi1_offset_mm, roi1_shape_yx, roi2_offset_mm, roi2_shape_yx)
            
            if not rois or len(rois) != 2:
                print(f"Advertencia: No se pudieron extraer las ROIs para la instancia #{i+1}. Se omite.")
                continue

            # 4. Analizar cada ROI para obtener las curvas MTF
            res_v = analyzer.analyze_roi(rois[0], pixel_spacing_mm, roi_id=f"inst_{i+1}_v", verbose=False)
            if res_v and res_v.get("status") == "OK":
                vertical_mtf_results.append(res_v)
            
            res_h = analyzer.analyze_roi(rois[1], pixel_spacing_mm, roi_id=f"inst_{i+1}_h", verbose=False)
            if res_h and res_h.get("status") == "OK":
                horizontal_mtf_results.append(res_h)
            
            processed_files_count += 1

        # 5. Promediar resultados y devolver
        valid_v_rois = len(vertical_mtf_results)
        valid_h_rois = len(horizontal_mtf_results)

        if valid_v_rois == 0 and valid_h_rois == 0:
            return {"status": "Error", "error_details": "No se pudieron obtener resultados MTF válidos."}

        # Lógica de promediado (similar a la de NNPS)
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
            raise ValueError("Fallo al combinar las curvas MTF.")

        coeffs, _, fit_stats = analyzer.fit_average_mtf_polynomial(combined_freq, combined_avg, degree=4)
        
        # --- CÁLCULO DE MTF AL 50% Y 10% ---
        freq_at_50_mtf = np.interp(0.5, combined_avg[::-1], combined_freq[::-1])
        freq_at_10_mtf = np.interp(0.1, combined_avg[::-1], combined_freq[::-1]) # <-- NUEVA LÍNEA

        return {
            "status": "OK",
            "processed_files_count": processed_files_count,
            "valid_vertical_rois": valid_v_rois,
            "valid_horizontal_rois": valid_h_rois,
            "combined_poly_coeffs": coeffs.tolist() if coeffs is not None else None,
            "fit_r_squared": fit_stats.get('r_squared'),
            "fit_rmse": fit_stats.get('rmse'),
            "mtf_at_50_percent": freq_at_50_mtf,
            "mtf_at_10_percent": freq_at_10_mtf, # <-- NUEVO CAMPO EN EL RESULTADO
            "error_details": None
        }

    except Exception as e:
        return {"status": "Error", "error_details": str(e)}

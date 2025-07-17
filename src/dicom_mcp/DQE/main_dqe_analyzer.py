# main_dqe_processor.py
# Script principal para orquestar el cálculo de MTF, NPS y DQE
# Versión final que agrupa por Kerma, usa calibración interactiva,
# ajusta MTF a polinomio (opcional) y guarda JSON.

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob
import pydicom
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple
import warnings
import json
import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox, Frame, Label, Entry, Button, LabelFrame
import math # Asegurar importación
from utils import obtener_datos_calibracion_vmp_k, calculate_linearization_slope, calculate_simple_vmp, convert_to_json_serializable

# Importar las clases y funciones necesarias
from mtf_analyzer import MtfAnalyzer
try: from roi_extractor import RoiExtractor
except ImportError: from mtf.roi_extractor import RoiExtractor # Intentar ruta alternativa
try: from nnps_analyzer import NnpsAnalyzer
except ImportError: print("ERROR: nnps_analyzer.py no encontrado."); exit()
try: from dqe_analyzer import DqeAnalyzer
except ImportError: print("ERROR: dqe_analyzer.py no encontrado."); exit()


# --- Función para Generar Datos de Calibración Interactivamente ---
def generate_calibration_data_interactively(
    nps_folder_path: str,
    required_kvp: float = 70.0,
    processing_tag_addr: Tuple[int, int] = (0x0008, 0x0068), # Presentation Intent Type
    processing_value_expected: str = "FOR PROCESSING",     # Valor exacto esperado
    roi_size_cm: Tuple[float, float] = (4.0, 4.0)
) -> Optional[pd.DataFrame]:
    # ... (print iniciales y búsqueda de nps_files como antes) ...
    print(f"\n--- Iniciando Generación Interactiva de Datos de Calibración ---")
    print(f"Carpeta de imágenes: {nps_folder_path}")
    print(f"Filtros: KVP={required_kvp}, Tag{processing_tag_addr} == '{processing_value_expected}'")
    nps_files = glob.glob(os.path.join(nps_folder_path, "*.dcm"))
    if not nps_files: print(f"Error: No .dcm en:\n{nps_folder_path}"); return None
    vmp_by_mas = defaultdict(list); pixel_spacing = None; files_processed = 0; files_matched = 0
    print("Procesando archivos para VMP y mAs...")

    for fpath in nps_files:
        print(f"  Leyendo: {os.path.basename(fpath)}...") # <-- DEBUG: Nombre archivo
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=False)
            kvp = ds.get("KVP")
            tag_val_raw = ds.get(processing_tag_addr)
            tag_val_str = None # Para almacenar el valor del tag como string

            # --- DEBUG PRINT ---
            print(f"    KVP leído: {kvp} (Tipo: {type(kvp)})")
            print(f"    Tag {processing_tag_addr} leído: {tag_val_raw} (Tipo: {type(tag_val_raw)})")
            # --- FIN DEBUG ---

            # Chequeo de KVP (un poco más flexible con floats)
            kvp_match = False
            if kvp is not None:
                try:
                    kvp_match = np.isclose(float(kvp), required_kvp, atol=0.1) # Tolera +/- 0.1
                except (ValueError, TypeError):
                    kvp_match = False # No es un número
            print(f"    ¿KVP ({required_kvp}) coincide?: {kvp_match}") # <-- DEBUG

            # Chequeo de Tag (exacto, case-insensitive)
            tag_match = False
            if tag_val_raw is not None:
                # Intentar obtener el valor como string
                try:
                    # Manejar multi-valor (como ImageType) o valor simple
                    if isinstance(tag_val_raw.value, list):
                         tag_val_str = "\\".join(map(str, tag_val_raw.value))
                    else:
                         tag_val_str = str(tag_val_raw.value)

                    if isinstance(tag_val_str, str):
                        tag_match = tag_val_str.upper().strip() == processing_value_expected.upper().strip()
                        print(f"    Valor Tag string: '{tag_val_str}'") # <-- DEBUG
                        print(f"    Comparando con '{processing_value_expected.upper().strip()}': {tag_match}") # <-- DEBUG
                except Exception as e_tag:
                     print(f"      Error al procesar valor del tag: {e_tag}")


            if not kvp_match or not tag_match:
                print("      --> Archivo OMITIDO (no cumple filtros)") # <-- DEBUG
                continue # No cumple criterios

            files_matched += 1
            print("      --> Archivo OK (filtros pasados)") # <-- DEBUG

            # Calcular mAs (desde tag 0018,1153)
            exposure_uas_tag = (0x0018, 0x1153)
            exposure_uas = ds.get(exposure_uas_tag)
            if exposure_uas is None or exposure_uas.value == '': warnings.warn(f"Falta Tag {exposure_uas_tag} en {os.path.basename(fpath)}. Se omite."); continue
            try: mAs = float(exposure_uas.value) / 1000.0
            except ValueError: warnings.warn(f"Valor inválido '{exposure_uas.value}' Tag {exposure_uas_tag} en {os.path.basename(fpath)}. Se omite."); continue
            mAs_key = round(mAs, 1)

            # Calcular VMP (como antes)
            px_space = ds.get("PixelSpacing", ds.get("ImagerPixelSpacing"));
            if px_space is None: warnings.warn(f"Falta PixelSpacing en {os.path.basename(fpath)}. Se omite."); continue
            current_ps = float(px_space[0])
            if pixel_spacing is None: pixel_spacing = current_ps
            elif not np.isclose(pixel_spacing, current_ps): warnings.warn(f"Inconsistencia PixelSpacing ({current_ps} vs {pixel_spacing}). Usando {pixel_spacing}")
            img_raw = ds.pixel_array.astype(np.float32); vmp_file = calculate_simple_vmp(img_raw, fraccion_roi=ROI_FRACTION_FOR_VMP)
            if np.isnan(vmp_file): warnings.warn(f"VMP inválido para {os.path.basename(fpath)}. Se omite."); continue
            vmp_by_mas[mAs_key].append(vmp_file); files_processed += 1

        except Exception as e: print(f"Error leyendo o procesando {os.path.basename(fpath)}: {e}")

    # ... (Resto de la función: comprobación final, GUI, etc.) ...
    print(f"Archivos procesados que cumplen criterios: {files_matched}")
    if files_processed == 0 or not vmp_by_mas: print("Error: No se procesaron archivos válidos."); return None
    avg_vmp_by_mas = {mAs: np.mean(vmps) for mAs, vmps in vmp_by_mas.items() if vmps}
    print("\nGrupos de mAs encontrados y VMP promedio:"); [print(f"  mAs ~ {mAs:.1f}: VMP={avg_vmp:.1f} ({len(vmp_by_mas[mAs])})") for mAs, avg_vmp in sorted(avg_vmp_by_mas.items())]
    for mAs, avg_vmp in sorted(avg_vmp_by_mas.items()):
        print(f"  mAs ~ {mAs:.1f}: VMP Promedio = {avg_vmp:.1f} ({len(vmp_by_mas[mAs])} imágenes)")

    # --- Crear GUI para Inputs Manuales ---
    results_gui = {'success': False} # Para almacenar resultados

    def on_ok_pressed():
        try:
            dist_img = float(entry_dist_img.get())
            dist_rad = float(entry_dist_rad.get())
            if dist_img <= 0 or dist_rad <= 0: raise ValueError("Distancias deben ser > 0.")
            kerma_map = {}
            valid = True
            for mAs_k, entry_widget in kerma_entries.items():
                kerma_val_str = entry_widget.get()
                if not kerma_val_str: messagebox.showwarning("Entrada Inválida", f"Introduce Kerma para mAs ~ {mAs_k:.1f}"); valid = False; break
                try: kerma_val = float(kerma_val_str)
                except ValueError: messagebox.showwarning("Entrada Inválida", f"Kerma inválido para mAs ~ {mAs_k:.1f}."); valid = False; break
                if kerma_val < 0: raise ValueError("Kerma >= 0.")
                kerma_map[mAs_k] = kerma_val
            if valid: results_gui['dist_img_cm'] = dist_img; results_gui['dist_rad_cm'] = dist_rad; results_gui['kerma_inputs_uGy'] = kerma_map; results_gui['success'] = True; root.destroy()
            else: results_gui['success'] = False # Mantener ventana abierta si no es válido
        except ValueError as e: messagebox.showwarning("Entrada Inválida", str(e)); results_gui['success'] = False

    def on_cancel():
        results_gui['success'] = False
        root.destroy()

    root = tk.Tk(); root.title("Entrada Datos Calibración")
    main_frame = Frame(root, padx=10, pady=10); main_frame.pack(fill=tk.BOTH, expand=True)
    dist_frame = LabelFrame(main_frame, text="Distancias (cm)", padx=5, pady=5); dist_frame.pack(pady=5, fill=tk.X)
    Label(dist_frame, text="Foco-Detector (Imagen):").grid(row=0, column=0, sticky=tk.W, padx=5); entry_dist_img = Entry(dist_frame, width=10); entry_dist_img.grid(row=0, column=1, padx=5); entry_dist_img.insert(0, "150")
    Label(dist_frame, text="Foco-Detector (Radiación):").grid(row=1, column=0, sticky=tk.W, padx=5); entry_dist_rad = Entry(dist_frame, width=10); entry_dist_rad.grid(row=1, column=1, padx=5); entry_dist_rad.insert(0, "150")
    kerma_frame = LabelFrame(main_frame, text="Kerma Medido (uGy) a Distancia Radiación", padx=5, pady=5); kerma_frame.pack(pady=5, fill=tk.X)
    kerma_entries = {}; row_k = 0
    for mAs_k, avg_vmp in sorted(avg_vmp_by_mas.items()): Label(kerma_frame, text=f"mAs ~ {mAs_k:.1f} (VMP {avg_vmp:.1f}):").grid(row=row_k, column=0, sticky=tk.W, padx=5, pady=2); entry_k = Entry(kerma_frame, width=10); entry_k.grid(row=row_k, column=1, padx=5, pady=2); kerma_entries[mAs_k] = entry_k; row_k += 1
    button_frame = Frame(main_frame); button_frame.pack(pady=10); Button(button_frame, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT, padx=5); Button(button_frame, text="OK", command=on_ok_pressed, default=tk.ACTIVE).pack(side=tk.RIGHT)

    # Centrar ventana
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop() # Esperar interacción

    # --- Procesar resultados de la GUI ---
    if not results_gui.get('success'): print("Operación cancelada."); return None
    final_k_ugy, final_vmp = [], []; dist_img_cm = results_gui['dist_img_cm']; dist_rad_cm = results_gui['dist_rad_cm']; kerma_inputs = results_gui['kerma_inputs_uGy']; dist_corr_factor = 1.0
    if not np.isclose(dist_img_cm, dist_rad_cm):
        if dist_img_cm <= 0: print("Error: Distancia imagen inválida."); return None
        dist_corr_factor = (dist_rad_cm / dist_img_cm) ** 2; print(f"Aplicando factor corrección distancia: {dist_corr_factor:.4f}")
    for mAs_k, avg_vmp in sorted(avg_vmp_by_mas.items()):
        k_manual = kerma_inputs.get(mAs_k);
        if k_manual is None: print(f"Error: Falta Kerma para mAs {mAs_k:.1f}"); return None
        k_corrected = k_manual * dist_corr_factor; final_k_ugy.append(k_corrected); final_vmp.append(avg_vmp)
    if not final_k_ugy or not final_vmp: print("Error: No se generaron datos VMP/Kerma."); return None
    calibration_df = pd.DataFrame({'K_uGy': final_k_ugy, 'VMP': final_vmp}).sort_values(by='K_uGy').reset_index(drop=True) # Ordenar por Kerma
    print("\nDataFrame de Calibración Generado:"); print(calibration_df); print("--- Fin Generación Interactiva ---")
    return calibration_df


# ==============================================================================
if __name__ == "__main__":

    # --- Diccionario Principal ---
    output_data = {
        "run_info": {},
        "configuration": { "paths": {}, "rqa_settings": {}, "mtf_parameters": {}, "nps_parameters": {}, "dqe_parameters": {}, "kerma_grouping": None, "plotting": {} },
        "inputs": { "mtf_files_found": [], "nps_files_found": [], "mtf_files_processed": [], "nps_files_grouped": {} },
        "derived_parameters": {},
        "results": { "average_mtf": {}, "per_kerma_group": {} }
    }
    start_time = datetime.datetime.now()
    output_data["run_info"]["timestamp_start"] = start_time.isoformat()
    output_data["run_info"]["script"] = os.path.basename(__file__)
    print(f"--- Iniciando Procesador DQE Completo (...) [{start_time.strftime('%Y-%m-%d %H:%M:%S')}] ---")

    # ==========================================================================
    # 1. CONFIGURACIÓN GENERAL
    # ==========================================================================
    MTF_FOLDER_PATH = "src/MTF"; NPS_FOLDER_PATH = "src/NNPS"; CALIBRATION_FOLDER_FOR_VMP = NPS_FOLDER_PATH
    output_data["configuration"]["paths"]["mtf_folder"] = MTF_FOLDER_PATH; output_data["configuration"]["paths"]["nps_folder"] = NPS_FOLDER_PATH; output_data["configuration"]["paths"]["calibration_folder_for_vmp"] = CALIBRATION_FOLDER_FOR_VMP
    RQA_SNR_FACTORS_IEC = { 'RQA3': 21759.0, 'RQA5': 30174.0, 'RQA7': 32362.0, 'RQA9': 31077.0, }; RQA_FACTORS_FOR_LINEARIZATION = { rqa: snr_sq / 1000.0 for rqa, snr_sq in RQA_SNR_FACTORS_IEC.items() }
    CURRENT_RQA = 'RQA5'; SNR_IN_SQ_CURRENT_RQA = RQA_SNR_FACTORS_IEC[CURRENT_RQA]; output_data["configuration"]["rqa_settings"] = {"processed_rqa": CURRENT_RQA, "snr_in_sq_iec_table": RQA_SNR_FACTORS_IEC, "factors_for_linearization": RQA_FACTORS_FOR_LINEARIZATION}
    MTF_PARAMS = { 'sub_pixel_factor': 0.1, 'smoothing_window_bins': 17, 'smoothing_polyorder': 4, 'baseline_tail_threshold_mm': 7.0, 'window_width_mm': 20.0 }; ROI1_OFFSET_MM=(13, 0); ROI1_SHAPE_YX=(100, 200); ROI1_LABEL="Vertical"; ROI2_OFFSET_MM=(0, -14); ROI2_SHAPE_YX=(200, 100); ROI2_LABEL="Horizontal"; NUM_POINTS_INTERP_MTF = 250; FIT_MTF_POLYNOMIAL = True; MTF_POLY_DEGREE = 4
    output_data["configuration"]["mtf_parameters"] = MTF_PARAMS.copy(); output_data["configuration"]["mtf_parameters"]["roi1_vertical"] = {'offset_mm': list(ROI1_OFFSET_MM), 'shape_yx': list(ROI1_SHAPE_YX), 'label': ROI1_LABEL}; output_data["configuration"]["mtf_parameters"]["roi2_horizontal"] = {'offset_mm': list(ROI2_OFFSET_MM), 'shape_yx': list(ROI2_SHAPE_YX), 'label': ROI2_LABEL}; output_data["configuration"]["mtf_parameters"]["num_points_interpolation"] = NUM_POINTS_INTERP_MTF; output_data["configuration"]["mtf_parameters"]["fit_polynomial"] = FIT_MTF_POLYNOMIAL; output_data["configuration"]["mtf_parameters"]["polynomial_degree"] = MTF_POLY_DEGREE if FIT_MTF_POLYNOMIAL else None
    NPS_PARAMS = { 'subregion_size': 256, 'overlap': 0.5, 'num_average_rows_1d_side': 7, 'apply_1d_smoothing': True, 'detrending_method': 'global_poly', 'poly_degree': 2, 'verbose': False }; ROI_FRACTION_FOR_VMP = 0.5; ROI_FRACTION_FOR_NPS = 0.8
    output_data["configuration"]["nps_parameters"] = NPS_PARAMS.copy(); output_data["configuration"]["nps_parameters"]["roi_fraction_for_vmp"] = ROI_FRACTION_FOR_VMP; output_data["configuration"]["nps_parameters"]["roi_fraction_for_nps_analysis"] = ROI_FRACTION_FOR_NPS; output_data["configuration"]["kerma_grouping"] = "round(Ka_est, 1), min_files=2"
    DQE_PARAMS = {'verbose': False}; output_data["configuration"]["dqe_parameters"] = DQE_PARAMS.copy()
    PLOT_AVERAGE_MTF = True; PLOT_NPS_PER_KERMA = False; PLOT_FINAL_DQE = True; output_data["configuration"]["plotting"] = {'plot_avg_mtf': PLOT_AVERAGE_MTF, 'plot_nps_per_kerma': PLOT_NPS_PER_KERMA, 'plot_final_dqe': PLOT_FINAL_DQE}

    # ==========================================================================
    # 1b. Búsqueda de Archivos DICOM
    # ==========================================================================
    mtf_files_found = glob.glob(os.path.join(MTF_FOLDER_PATH, "*.dcm")); nps_files_found = glob.glob(os.path.join(NPS_FOLDER_PATH, "*.dcm"))
    output_data["inputs"]["mtf_files_found"] = [os.path.basename(f) for f in mtf_files_found]; output_data["inputs"]["nps_files_found"] = [os.path.basename(f) for f in nps_files_found]
    print(f"Buscando MTF en '{MTF_FOLDER_PATH}': {len(mtf_files_found)} encontrados."); print(f"Buscando NPS en '{NPS_FOLDER_PATH}': {len(nps_files_found)} encontrados.")

    # ==========================================================================
    # 2. GENERACIÓN INTERACTIVA DE DATOS DE CALIBRACIÓN
    # ==========================================================================
    print(f"\nGenerando datos de calibración interactivamente...")
    calibration_df = generate_calibration_data_interactively(
        nps_folder_path=CALIBRATION_FOLDER_FOR_VMP,
        required_kvp=70.0,
        processing_tag_addr=(0x0008, 0x0068),
        processing_value_expected="FOR PROCESSING",
        roi_size_cm=(4.0, 4.0)
    )
    if calibration_df is None or calibration_df.empty: print("Error fatal: No se generaron datos de calibración."); exit()
    print("Datos de calibración generados/obtenidos OK.")
    output_data["configuration"]["calibration_source"] = "Generated Interactively"
    output_data["configuration"]["calibration_data_generated"] = calibration_df.to_dict(orient='records')

    # Calcular pendiente
    print(f"Calculando pendiente de linealización para {CURRENT_RQA}...")
    slope_prime = calculate_linearization_slope(calibration_df, CURRENT_RQA, RQA_FACTORS_FOR_LINEARIZATION)
    if slope_prime is None: print(f"Error fatal: No se pudo calcular la pendiente para {CURRENT_RQA}."); exit()
    print(f"Pendiente (VMP vs q/area) para {CURRENT_RQA}: {slope_prime:.4e}")
    output_data["derived_parameters"]["linearization_slope"] = slope_prime

    # ==========================================================================
    # 3. Cálculo de MTF Promedio y Ajuste Polinomial
    # ==========================================================================
    print("\n--- Iniciando Cálculo MTF Promedio ---")
    mtf_analyzer = None; avg_mtf_h_results = None; avg_mtf_v_results = None; avg_mtf_results_combined = None
    pixel_spacing_mtf = None; mtf_files_processed_names = []; mtf_poly_coeffs = None; mtf_poly_func = None
    vertical_mtf_results_list, horizontal_mtf_results_list = [], []

    try: # --- Bloque try externo para toda la sección MTF ---
        mtf_analyzer = MtfAnalyzer(calibration_df=calibration_df, rqa_factors_dict=RQA_FACTORS_FOR_LINEARIZATION, **MTF_PARAMS)
        print("Instancia de MtfAnalyzer creada.")
        if not mtf_files_found: print("Advertencia: No hay archivos MTF para procesar.")
        else:
            for i, dicom_path in enumerate(mtf_files_found):
                try: # --- Bloque try interno para cada archivo MTF ---
                    extractor = RoiExtractor(dicom_path, verbose=False); current_pixel_spacing = extractor.pixel_spacing
                    if pixel_spacing_mtf is None: pixel_spacing_mtf = current_pixel_spacing
                    elif not np.isclose(pixel_spacing_mtf, current_pixel_spacing): warnings.warn("Inconsistencia Pixel Spacing MTF")
                    rois_raw = extractor.extract_mtf_rois(ROI1_OFFSET_MM, ROI1_SHAPE_YX, ROI2_OFFSET_MM, ROI2_SHAPE_YX)
                    results_v = None; results_h = None
                    if rois_raw and len(rois_raw) > 0 and rois_raw[0].size > 0: results_v = mtf_analyzer.analyze_roi(rois_raw[0], pixel_spacing_mtf, CURRENT_RQA, roi_id=f"{ROI1_LABEL}-{i+1}", verbose=False, plot_angle_fit=False)
                    if results_v and results_v.get("status", "").startswith("OK") and results_v.get("mtf") is not None: vertical_mtf_results_list.append(results_v)
                    if rois_raw and len(rois_raw) > 1 and rois_raw[1].size > 0: results_h = mtf_analyzer.analyze_roi(rois_raw[1], pixel_spacing_mtf, CURRENT_RQA, roi_id=f"{ROI2_LABEL}-{i+1}", verbose=False, plot_angle_fit=False)
                    if results_h and results_h.get("status", "").startswith("OK") and results_h.get("mtf") is not None: horizontal_mtf_results_list.append(results_h)
                    mtf_files_processed_names.append(os.path.basename(dicom_path))
                except Exception as e_mtf: # --- Except interno ---
                    print(f"    Error procesando archivo MTF {os.path.basename(dicom_path)}: {e_mtf}")
                    file_basename = os.path.basename(dicom_path)
                    if file_basename not in mtf_files_processed_names: mtf_files_processed_names.append(file_basename)
        # --- Código después del bucle DENTRO DEL TRY EXTERNO ---
        output_data["inputs"]["mtf_files_processed"] = mtf_files_processed_names
        print(f"\n  Archivos MTF procesados: {len(mtf_files_processed_names)}")
        print(f"  Resultados MTF Vertical válidos: {len(vertical_mtf_results_list)}")
        print(f"  Resultados MTF Horizontal válidos: {len(horizontal_mtf_results_list)}")
        if vertical_mtf_results_list: freq_v_avg, mtf_v_avg, _, _ = MtfAnalyzer.calculate_grand_average_mtf(vertical_mtf_results_list, NUM_POINTS_INTERP_MTF); avg_mtf_v_results = {'frequencies': freq_v_avg, 'mtf': mtf_v_avg, 'direction': ROI1_LABEL}; print(f"MTF Promedio Vertical calculada.")
        if horizontal_mtf_results_list: freq_h_avg, mtf_h_avg, _, _ = MtfAnalyzer.calculate_grand_average_mtf(horizontal_mtf_results_list, NUM_POINTS_INTERP_MTF); avg_mtf_h_results = {'frequencies': freq_h_avg, 'mtf': mtf_h_avg, 'direction': ROI2_LABEL}; print(f"MTF Promedio Horizontal calculada.")
        if avg_mtf_h_results and avg_mtf_v_results and np.allclose(avg_mtf_h_results['frequencies'], avg_mtf_v_results['frequencies']): mtf_comb_avg = (avg_mtf_h_results['mtf'] + avg_mtf_v_results['mtf']) / 2.0; avg_mtf_results_combined = {'frequencies': avg_mtf_h_results['frequencies'], 'mtf': mtf_comb_avg, 'direction': 'Promedio H&V'}; print(f"MTF Combinada (Promedio H&V) calculada.")

        # --- Ajuste Polinomial MTF Combinada ---
        if FIT_MTF_POLYNOMIAL and avg_mtf_results_combined:
            print(f"  Ajustando MTF combinada a polinomio grado {MTF_POLY_DEGREE}...")
            # --- CORRECCIÓN: Separar try/except del print ---
            try:
                mtf_poly_coeffs, mtf_poly_func = mtf_analyzer.fit_average_mtf_polynomial(
                    avg_mtf_results_combined['frequencies'],
                    avg_mtf_results_combined['mtf'],
                    degree=MTF_POLY_DEGREE
                )
                if mtf_poly_coeffs is not None:
                    print(f"    Ajuste polinómico MTF OK. Coefs: {np.round(mtf_poly_coeffs, 5)}")
                    # Guardar en JSON
                    output_data["results"]["average_mtf"]["combined_hv_poly_coeffs"] = mtf_poly_coeffs.tolist()
                    output_data["results"]["average_mtf"]["combined_hv_poly_degree"] = MTF_POLY_DEGREE
                else:
                    print("    Fallo en el ajuste polinómico MTF.")
                    mtf_poly_func = None # Asegurar que no se use para plotear si falló
            except Exception as e_fit:
                print(f"    Error durante el ajuste polinómico MTF: {e_fit}")
                mtf_poly_coeffs = None
                mtf_poly_func = None
            # --- FIN CORRECCIÓN ---

        # Guardar resultados MTF promedio en JSON
        if pixel_spacing_mtf: output_data["derived_parameters"]["pixel_spacing_mtf_mm"] = pixel_spacing_mtf
        if avg_mtf_h_results: output_data["results"]["average_mtf"]["horizontal"] = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in avg_mtf_h_results.items()}
        if avg_mtf_v_results: output_data["results"]["average_mtf"]["vertical"] = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in avg_mtf_v_results.items()}
        if avg_mtf_results_combined: output_data["results"]["average_mtf"]["combined_hv"] = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in avg_mtf_results_combined.items()}

        # Plotear MTF promedio
        if PLOT_AVERAGE_MTF:
             nyquist_freq_mtf = 0.5 / pixel_spacing_mtf if pixel_spacing_mtf else 5.0
             plt.figure("MTF Promedio H vs V", figsize=(8, 5)); plt.clf()
             if avg_mtf_h_results: plt.plot(avg_mtf_h_results['frequencies'], avg_mtf_h_results['mtf'], 'b.-', label=f'MTF Promedio {ROI2_LABEL}')
             if avg_mtf_v_results: plt.plot(avg_mtf_v_results['frequencies'], avg_mtf_v_results['mtf'], 'r.-', label=f'MTF Promedio {ROI1_LABEL}')
             if avg_mtf_results_combined: plt.plot(avg_mtf_results_combined['frequencies'], avg_mtf_results_combined['mtf'], '--', color='gray', alpha=0.7, label='MTF Promedio H&V')
             if FIT_MTF_POLYNOMIAL and mtf_poly_func is not None and avg_mtf_results_combined: mtf_fit_y = np.clip(mtf_poly_func(avg_mtf_results_combined['frequencies']), 0, 1.05); plt.plot(avg_mtf_results_combined['frequencies'], mtf_fit_y, 'g:', lw=2, label=f'Ajuste Poli (G{MTF_POLY_DEGREE})')
             plt.title(f"MTF Promedio ({CURRENT_RQA})"); plt.xlabel("Frecuencia Espacial [c/mm]"); plt.ylabel("MTF"); plt.xlim(0, nyquist_freq_mtf); plt.ylim(0, 1.1); plt.grid(True, which='both', ls=':'); plt.legend(); plt.tight_layout()
    # --- Fin del bloque try externo ---
    except Exception as e: # --- Except externo ---
        print(f"Error fatal durante el análisis MTF: {e}")
        output_data["inputs"]["mtf_files_processed"] = mtf_files_processed_names

    # ==========================================================================
    # 4. Agrupar Archivos NPS por Kerma Estimado
    # ==========================================================================
    print("\n--- Agrupando Archivos NPS por Kerma Estimado (1 decimal) ---")
    nps_files_by_kerma = defaultdict(list)
    if not nps_files_found:
        print("No hay archivos NPS para agrupar.")
    else:
        # --- Inicio Bloque Indentado ---
        for nps_file_path in nps_files_found:
             try:
                 ds = pydicom.dcmread(nps_file_path, stop_before_pixels=False); img_raw = ds.pixel_array.astype(np.float32)
                 vmp_file = calculate_simple_vmp(img_raw, fraccion_roi=ROI_FRACTION_FOR_VMP)
                 if np.isnan(vmp_file): warnings.warn(f"VMP inválido para {os.path.basename(nps_file_path)}. Se omite."); continue
                 if abs(slope_prime * SNR_IN_SQ_CURRENT_RQA) > 1e-12: ka_file_est = vmp_file / (slope_prime * SNR_IN_SQ_CURRENT_RQA); kerma_group_key = round(ka_file_est, 1); nps_files_by_kerma[kerma_group_key].append(os.path.basename(nps_file_path))
                 else: warnings.warn(f"Denom cero Ka {os.path.basename(nps_file_path)}.")
             except Exception as e_group: print(f"Error agrupando {os.path.basename(nps_file_path)}: {e_group}")
        print(f"\nArchivos NPS agrupados en {len(nps_files_by_kerma)} niveles de Kerma estimado:"); output_data["inputs"]["nps_files_grouped"] = {f"{k:.1f}": v for k, v in nps_files_by_kerma.items()}; [print(f"  Kerma ~{k:.1f} uGy: {len(files)} archivos") for k, files in sorted(nps_files_by_kerma.items())]
        # --- Fin Bloque Indentado ---

    # ==========================================================================
    # 5. Calcular NPS y DQE para cada Grupo de Kerma
    # ==========================================================================
    print("\n--- Calculando NPS y DQE por Grupo de Kerma (mínimo 2 archivos por grupo) ---")
    # ... (Código para iterar grupos, calcular NPS/DQE como antes) ...
    # ... (Guardar resultados en kerma_groups_processed) ...
    nps_analyzer, dqe_analyzer = None, None
    kerma_groups_processed = {}
    pixel_spacing_nps = None
    have_mtf_h = avg_mtf_h_results is not None; have_mtf_v = avg_mtf_v_results is not None; have_mtf_r = avg_mtf_results_combined is not None
    if not (have_mtf_h or have_mtf_v): print("Error: No se puede proceder sin MTF promedio válida."); exit()
    if not nps_files_by_kerma: print("No hay grupos de Kerma para procesar.")
    else:
        try:
            nps_analyzer = NnpsAnalyzer(calibration_df=calibration_df, rqa_factors_dict=RQA_FACTORS_FOR_LINEARIZATION, **NPS_PARAMS)
            dqe_analyzer = DqeAnalyzer(rqa_snr_factors=RQA_SNR_FACTORS_IEC, **DQE_PARAMS)
            print("Instancias NnpsAnalyzer y DqeAnalyzer creadas.")
        except Exception as e_init: print(f"Error al inicializar analyzers: {e_init}"); exit()

        for kerma_key, file_paths_group_names in sorted(nps_files_by_kerma.items()):
            file_paths_group = [os.path.join(NPS_FOLDER_PATH, fname) for fname in file_paths_group_names]
            if len(file_paths_group) < 2: print(f"\n--- Omitiendo Grupo Kerma ~ {kerma_key:.1f} uGy (solo {len(file_paths_group)} archivo/s) ---"); continue
            print(f"\n--- Procesando Grupo Kerma ~ {kerma_key:.1f} uGy ({len(file_paths_group)} archivos) ---")
            kerma_key_str = f"{kerma_key:.1f}"; group_output = {"input_files": file_paths_group_names}
            nps_results_group = None; Ka_group_rep = None
            try:
                nps_results_group = nps_analyzer.analyze_flat_field_set(file_paths_group, CURRENT_RQA, ROI_FRACTION_FOR_NPS)
                if nps_results_group and nps_results_group['status'] == 'OK':
                    print(f"  Cálculo NPS grupo {kerma_key:.1f} OK."); group_output["nps_analysis"] = {k: v for k, v in nps_results_group.items() if k not in ['status', 'processed_files', 'error_details', 'nps_2d', 'nnps_2d']}
                    if pixel_spacing_nps is None: pixel_spacing_nps = nps_results_group.get('pixel_spacing')
                    mean_vmp_grp, slope_p_grp = nps_results_group.get('mean_vmp_roi'), nps_results_group.get('slope_prime')
                    if mean_vmp_grp is not None and slope_p_grp is not None:
                        if abs(slope_p_grp * SNR_IN_SQ_CURRENT_RQA) > 1e-12: Ka_group_rep = mean_vmp_grp / (slope_p_grp * SNR_IN_SQ_CURRENT_RQA); nps_results_group['air_kerma_nps'] = Ka_group_rep; group_output["kerma_estimated_uGy"] = Ka_group_rep; print(f"  Ka representativo grupo: {Ka_group_rep:.3f} uGy")
                        else: print(f"Error: Denom cero Ka grupo {kerma_key:.1f}."); nps_results_group=None
                    else: print(f"Error: Faltan VMP/slope grupo {kerma_key:.1f}."); nps_results_group=None
                    if PLOT_NPS_PER_KERMA and nps_results_group: nps_analyzer.plot_nps_1d(nps_results_group, normalize=True, log_scale=True)
                else: print(f"  Error NPS grupo {kerma_key:.1f}: {nps_results_group.get('error_details', 'Desconocido')}"); nps_results_group = None
            except Exception as e_nps_grp: print(f"  Error fatal NPS grupo {kerma_key:.1f}: {e_nps_grp}"); nps_results_group = None

            if nps_results_group and nps_results_group['status'] == 'OK' and 'air_kerma_nps' in nps_results_group:
                 print(f"  Calculando DQE grupo {kerma_key:.1f} (Ka={Ka_group_rep:.3f} uGy)...")
                 group_output["dqe_analysis"] = {}
                 try: # Calcular DQE H,V,R
                    dqe_h, dqe_v, dqe_r = None, None, None
                    if have_mtf_h: dqe_h = dqe_analyzer.calculate_dqe(avg_mtf_h_results, nps_results_group, 'horizontal')
                    if have_mtf_v: dqe_v = dqe_analyzer.calculate_dqe(avg_mtf_v_results, nps_results_group, 'vertical')
                    if have_mtf_r: dqe_r = dqe_analyzer.calculate_dqe(avg_mtf_results_combined, nps_results_group, 'radial')
                    if dqe_h and dqe_h['status'] == 'OK': group_output["dqe_analysis"]['horizontal'] = {k: v for k, v in dqe_h.items() if k!='status'} # Guardar sin status
                    else: print(f"    Error DQE H (grupo {kerma_key:.1f})")
                    if dqe_v and dqe_v['status'] == 'OK': group_output["dqe_analysis"]['vertical'] = {k: v for k, v in dqe_v.items() if k!='status'}
                    else: print(f"    Error DQE V (grupo {kerma_key:.1f})")
                    if dqe_r and dqe_r['status'] == 'OK': group_output["dqe_analysis"]['radial'] = {k: v for k, v in dqe_r.items() if k!='status'}
                    else: print(f"    Error DQE R (grupo {kerma_key:.1f})")
                    if group_output["dqe_analysis"]: print(f"  Cálculo DQE completado para grupo {kerma_key:.1f} uGy.")
                    else: print(f"  Error: Falló DQE para todas las dirs. grupo {kerma_key:.1f}.")
                 except Exception as e_dqe_grp: print(f"  Error fatal DQE grupo {kerma_key:.1f}: {e_dqe_grp}")
            kerma_groups_processed[kerma_key_str] = group_output

        output_data["results"]["per_kerma_group"] = kerma_groups_processed
        if pixel_spacing_nps:
            output_data["derived_parameters"]["pixel_spacing_nps_mm"] = pixel_spacing_nps
            output_data["derived_parameters"]["nyquist_freq_cpmm"] = 0.5 / pixel_spacing_nps


    # ==========================================================================
    # 6. Graficar DQE Final Comparando Kermas
    # ==========================================================================
    if PLOT_FINAL_DQE and output_data["results"]["per_kerma_group"]:
        print("\n--- Generando Gráficos DQE Comparativos por Kerma ---")
        # ... (código plot DQE como antes)...
        directions_to_plot = ['horizontal', 'vertical', 'radial']
        plot_titles = { 'horizontal': f'DQE ({ROI2_LABEL})', 'vertical': f'DQE ({ROI1_LABEL})', 'radial': 'DQE (Radial)'}
        line_styles = ['-', '--', ':', '-.'] * (len(output_data["results"]["per_kerma_group"]) // 4 + 1)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(output_data["results"]["per_kerma_group"])))
        for direction in directions_to_plot:
            plt.figure(f"DQE Comparativa - {plot_titles[direction]}", figsize=(9, 6)); plt.clf()
            plot_count = 0; color_idx = 0
            for kerma_key_str in sorted(output_data["results"]["per_kerma_group"].keys(), key=float):
                dqe_group_results = output_data["results"]["per_kerma_group"][kerma_key_str]
                if "dqe_analysis" in dqe_group_results and direction in dqe_group_results["dqe_analysis"]:
                    result = dqe_group_results["dqe_analysis"][direction]
                    label = f'{kerma_key_str} µGy'
                    freq = np.array(result['frequency']); dqe = np.array(result['dqe'])
                    plt.plot(freq, dqe, linestyle=line_styles[color_idx % len(line_styles)], color=colors[color_idx % len(colors)], marker='.', ms=3, label=label)
                    plot_count += 1; color_idx += 1
            if plot_count > 0:
                plt.xlabel("Frecuencia espacial [c/mm]"); plt.ylabel("DQE"); plt.title(f"{plot_titles[direction]} vs Kerma Estimado ({CURRENT_RQA})"); plt.grid(True, which='both', ls=':'); plt.legend(title="Kerma ~ (µGy)", fontsize='small')
                nyquist_freq = output_data.get("derived_parameters", {}).get("nyquist_freq_cpmm", 5.0)
                plt.xlim(0, nyquist_freq); plt.ylim(0, 1.05); plt.tight_layout()
            else: plt.close(); print(f"No se generó gráfico DQE para '{direction}'.")


    # ==========================================================================
    # 7. Guardar Resultados en JSON (Modificado)
    # ==========================================================================
    output_data["run_info"]["timestamp_end"] = datetime.datetime.now().isoformat()
    output_json_filename = f"dqe_results_{CURRENT_RQA}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    print(f"\n--- Guardando resultados en: {output_json_filename} ---")
    try:
        # --- Convertir todo el diccionario antes de guardar ---
        print("  Convirtiendo datos a formato serializable JSON...")
        serializable_data = convert_to_json_serializable(output_data)
        print("  Conversión completada.")
        # --- FIN ---

        with open(output_json_filename, 'w') as f:
            json.dump(serializable_data, f, indent=4, allow_nan=False)
        print("Resultados guardados exitosamente.")
    except TypeError as e:
        print(f"Error de TIPO al convertir/guardar JSON: {e}")
        print("Verifica 'convert_to_json_serializable'.")
    except Exception as e:
        print(f"Error inesperado al guardar JSON: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # 8. Finalización
    # ==========================================================================
    print("\n--- Proceso DQE Completo Finalizado ---")
    if PLOT_AVERAGE_MTF or (PLOT_FINAL_DQE and output_data.get("results", {}).get("per_kerma_group")):
         plt.show()
# --- Script Principal (ej: main_mtf_processor.py - Modificado para ajuste polinómico combinado) ---
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob

from roi_extractor import RoiExtractor
from mtf_analyzer import MtfAnalyzer # Asumiendo métodos estáticos y de ajuste
from utils import obtener_datos_calibracion_vmp_k

if __name__ == "__main__":

    # --- Configuración ---
    # !#!#! RUTA A LA CARPETA CON LAS IMÁGENES DICOM !#!#!
    dicom_folder = "src/MTF/"
    dicom_files = glob.glob(os.path.join(dicom_folder, '*.dcm'))

    # Validar si se encontraron archivos
    if not dicom_files:
        print(f"Error: No se encontraron archivos .dcm en la carpeta: {dicom_folder}")
        exit()

    print(f"Se encontraron {len(dicom_files)} archivos DICOM en '{dicom_folder}'.")

    # Asunciones de dirección
    roi1_offset_mm = (13, 0); roi1_shape_yx = (100, 200) # -> Vertical MTF
    roi2_offset_mm = (0, -14); roi2_shape_yx = (200, 100) # -> Horizontal MTF
    ROI1_LABEL = "Vertical"
    ROI2_LABEL = "Horizontal"

    # --- Parámetros de análisis MTF ---
    mtf_params = {
        'sub_pixel_factor': 0.1, 'smoothing_window_bins': 17, 'smoothing_polyorder': 4,
        'baseline_tail_threshold_mm': 7.0, 'window_width_mm': 20.0, 'cutoff_freq': 3.7,
    }
    GLOBAL_CUTOFF_FREQ = mtf_params['cutoff_freq']
    NUM_POINTS_INTERP = 250 # Puntos para interpolación y eje común

    # --- Configuración de Calibración ---
    calibration_csv_path = "src/data/linearizacion.csv" # !#!#! ACTUALIZA !#!#!
    RQA_FACTORS = { # !#!#! COMPLETA !#!#!
        'RQA5': 1.23456e6, 'RQA9': 2.34567e6,
        # ... otros tipos RQA ...
    }

    # --- Control de Gráficos y Ajuste ---
    plot_individual_analyses = False
    plot_average_mtf = False
    plot_combined_final_results = True # Activa el gráfico final
    plot_angle_fits = False
    fit_combined_average_poly = True  # Ajustar polinomio AL PROMEDIO COMBINADO V&H
    # Opciones para el gráfico final:
    plot_VH_averages_on_final = True # Mostrar promedios V y H separados (más tenues)
    plot_combined_average_on_final = True # Mostrar el promedio V&H combinado (destacado)
    plot_final_poly_fit = True # Mostrar el ajuste polinómico del promedio V&H
    plot_std_dev_on_final = False # Desactivado para claridad (podría añadirse para V y H si se desea)
    plot_individuals_on_final = False # Desactivado para claridad
    # -----------------------------------

    print("--- Iniciando Procesamiento MTF para Ajuste Polinómico Combinado ---")

    # --- 0. Cargar Datos de Calibración ---
    # ... (igual que antes) ...
    print("  [0] Cargando datos de calibración...")
    calibration_df = obtener_datos_calibracion_vmp_k(calibration_csv_path)
    if calibration_df is None: exit("Error fatal: Faltan datos de calibración.")
    print("  Datos de calibración cargados OK.")


    # --- Crear instancia del analizador MTF ---
    # ... (igual que antes) ...
    print("  [*] Creando instancia de MtfAnalyzer...")
    try:
        analyzer = MtfAnalyzer(calibration_df=calibration_df, rqa_factors_dict=RQA_FACTORS, **mtf_params)
        print("  Instancia de MtfAnalyzer creada OK.")
    except Exception as e: exit(f"Error fatal al inicializar MtfAnalyzer: {e}")


    # --- Listas para acumular resultados VÁLIDOS por dirección ---
    vertical_mtf_results = []
    horizontal_mtf_results = []
    processed_files_count = 0
    error_files_count = 0

    # --- Bucle Principal ---
    # ... (igual que antes, acumula en vertical_mtf_results y horizontal_mtf_results) ...
    for dicom_path in dicom_files:
        print(f"\n>>> Procesando Archivo: {os.path.basename(dicom_path)} <<<")
        file_processed_ok = False
        try:
            # 1. Extraer ROIs RAW
            print("  [1] Extrayendo ROIs RAW...")
            extractor = RoiExtractor(dicom_path, verbose=False) # Menos verboso
            pixel_spacing = extractor.pixel_spacing
            dicom_dataset = extractor.ds
            if pixel_spacing is None: raise ValueError("Pixel spacing no encontrado.")
            rois_raw = extractor.extract_mtf_rois(roi1_offset_mm, roi1_shape_yx, roi2_offset_mm, roi2_shape_yx)
            if not rois_raw or len(rois_raw) != 2 or rois_raw[0].size == 0 or rois_raw[1].size == 0:
                 raise ValueError("Fallo al extraer ROIs RAW.")
            print("  ROIs RAW extraídas OK.")

            # 2. Determinar RQA Type
            print("  [2] Determinando RQA Type...")
            # --- Lógica para determinar current_rqa_type ---
            # !#!#! PLACEHOLDER: Implementa tu lógica aquí !#!#!
            current_rqa_type = 'RQA5' # Ejemplo hardcodeado
            print(f"    RQA Type: {current_rqa_type}")
            if current_rqa_type not in analyzer.rqa_factors_dict:
                 raise ValueError(f"RQA Type '{current_rqa_type}' desconocido.")
            # ----------------------------------------------

            # 3. Analizar ROI 1 (Vertical)
            print(f"  [3] Analizando ROI 1 ({ROI1_LABEL})...")
            results1 = analyzer.analyze_roi(rois_raw[0], pixel_spacing, current_rqa_type,
                                            roi_id=f"{os.path.basename(dicom_path)}-{ROI1_LABEL}",
                                            verbose=False, plot_angle_fit=plot_angle_fits)
            if results1 and "OK" in results1.get("status", "") and results1.get("frequencies") is not None:
                print(f"    ROI 1 ({ROI1_LABEL}) MTF OK.")
                vertical_mtf_results.append(results1)
            else:
                 print(f"    Advertencia: ROI 1 ({ROI1_LABEL}) no produjo MTF válida.")

            # 4. Analizar ROI 2 (Horizontal)
            print(f"  [4] Analizando ROI 2 ({ROI2_LABEL})...")
            results2 = analyzer.analyze_roi(rois_raw[1], pixel_spacing, current_rqa_type,
                                            roi_id=f"{os.path.basename(dicom_path)}-{ROI2_LABEL}",
                                            verbose=False, plot_angle_fit=plot_angle_fits)
            if results2 and "OK" in results2.get("status", "") and results2.get("frequencies") is not None:
                print(f"    ROI 2 ({ROI2_LABEL}) MTF OK.")
                horizontal_mtf_results.append(results2)
            else:
                 print(f"    Advertencia: ROI 2 ({ROI2_LABEL}) no produjo MTF válida.")

            file_processed_ok = True

        except Exception as e:
             print(f"  ERROR procesando {os.path.basename(dicom_path)}: {e}")
             error_files_count += 1

        if file_processed_ok:
            processed_files_count += 1
    # --- Fin Bucle ---

    print(f"\n--- Fin del Bucle de Procesamiento ---")
    print(f"Archivos procesados (intentados): {len(dicom_files)}")
    print(f"Archivos con errores fatales: {error_files_count}")
    print(f"Total ROIs VÁLIDAS acumuladas ({ROI1_LABEL}): {len(vertical_mtf_results)}")
    print(f"Total ROIs VÁLIDAS acumuladas ({ROI2_LABEL}): {len(horizontal_mtf_results)}")

    # --- Paso Final: Calcular Promedios y Ajuste Combinado ---
    final_results = {
        "vert_freq": None, "vert_avg": None,
        "horiz_freq": None, "horiz_avg": None,
        "combined_freq": None, "combined_avg": None,
        "combined_poly_coeffs": None, "combined_poly_func": None,
        "fit_stats": None
    }

    # Calcular promedio Vertical
    if len(vertical_mtf_results) >= 2:
        print(f"\n--- Calculando Promedio Global MTF ({ROI1_LABEL}) ---")
        final_results["vert_freq"], final_results["vert_avg"], _, _ = \
            MtfAnalyzer.calculate_grand_average_mtf(
                vertical_mtf_results, num_points=NUM_POINTS_INTERP, forced_cutoff_freq=GLOBAL_CUTOFF_FREQ
            )
    else:
        print(f"\nNo hay suficientes resultados para calcular promedio global {ROI1_LABEL}.")

    # Calcular promedio Horizontal
    if len(horizontal_mtf_results) >= 2:
        print(f"\n--- Calculando Promedio Global MTF ({ROI2_LABEL}) ---")
        final_results["horiz_freq"], final_results["horiz_avg"], _, _ = \
            MtfAnalyzer.calculate_grand_average_mtf(
                horizontal_mtf_results, num_points=NUM_POINTS_INTERP, forced_cutoff_freq=GLOBAL_CUTOFF_FREQ
            )
    else:
         print(f"\nNo hay suficientes resultados para calcular promedio global {ROI2_LABEL}.")

# Calcular Promedio Combinado (V&H)
    if final_results["vert_avg"] is not None and final_results["horiz_avg"] is not None:
        print("\n--- Calculando Promedio Combinado V&H ---")
        # Validar que los ejes de frecuencia son compatibles usando tolerancia
        if np.allclose(final_results["vert_freq"], final_results["horiz_freq"], rtol=1e-5, atol=1e-8): # <<<--- CAMBIO AQUÍ
            # Si son cercanos, usar uno de ellos como eje combinado (p.ej., el vertical)
            # Pequeñas diferencias no afectarán el promedio de los valores MTF.
            final_results["combined_freq"] = final_results["vert_freq"]
            final_results["combined_avg"] = (final_results["vert_avg"] + final_results["horiz_avg"]) / 2.0
            print("  Promedio combinado V&H calculado.")

            # Ajustar Polinomio al Promedio Combinado
            if fit_combined_average_poly:
                print("\n--- Ajustando Polinomio al Promedio Combinado V&H ---")
                final_results["combined_poly_coeffs"], final_results["combined_poly_func"], final_results["fit_stats"] = \
                    analyzer.fit_average_mtf_polynomial(
                        final_results["combined_freq"], final_results["combined_avg"], degree=4
                    )
                if final_results["combined_poly_func"]:
                    print(f"  Ajuste polinómico final OK. Coefs: {final_results['combined_poly_coeffs']}")
                else:
                    print("  Fallo en el ajuste polinómico final.")
        else:
            # Este mensaje ahora solo debería aparecer si hay una diferencia REAL significativa
            print("Error: Los ejes de frecuencia promedio Vertical y Horizontal difieren significativamente. No se puede calcular el promedio combinado.")
            # Opcional: Añadir debug para ver las diferencias si este error persiste
            # print("DEBUG: Freq Vert Shape:", final_results["vert_freq"].shape, "Freq Horiz Shape:", final_results["horiz_freq"].shape)
            # print("DEBUG: Diferencia Máxima Freq:", np.max(np.abs(final_results["vert_freq"] - final_results["horiz_freq"])))
            # print("DEBUG: Freq Vert (primeros 5):", final_results["vert_freq"][:5])
            # print("DEBUG: Freq Horiz (primeros 5):", final_results["horiz_freq"][:5])


    else:
        print("\nNo se pueden calcular el promedio combinado V&H (faltan datos de V o H).")


    # --- Graficar Resultados Finales Combinados ---
    if plot_combined_final_results and final_results["combined_avg"] is not None:
        print("\n[Graficando Resultados Finales Combinados]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        plt.figure("MTF Final Combinada y Ajuste", figsize=(10, 7)); plt.clf()
        ax = plt.gca()

        # Plot Promedio Vertical (opcional, más tenue)
        if plot_VH_averages_on_final and final_results["vert_avg"] is not None:
            ax.plot(final_results["vert_freq"], final_results["vert_avg"],
                    color='blue', ls=':', lw=1.5, marker='None',
                    label=f'Promedio {ROI1_LABEL} ({len(vertical_mtf_results)} ROIs)')

        # Plot Promedio Horizontal (opcional, más tenue)
        if plot_VH_averages_on_final and final_results["horiz_avg"] is not None:
             ax.plot(final_results["horiz_freq"], final_results["horiz_avg"],
                     color='red', ls=':', lw=1.5, marker='None',
                     label=f'Promedio {ROI2_LABEL} ({len(horizontal_mtf_results)} ROIs)')

        # Plot Promedio Combinado V&H (destacado)
        if plot_combined_average_on_final:
             ax.plot(final_results["combined_freq"], final_results["combined_avg"],
                     color='black', ls='-', lw=2.5, marker='None',
                     label='Promedio Combinado V&H')

        # Plot Ajuste Polinómico Final (del combinado)
        if plot_final_poly_fit and final_results["combined_poly_func"] is not None:
            poly_fit_vals = np.clip(final_results["combined_poly_func"](final_results["combined_freq"]), 0, 1)
            r_squared = final_results.get("fit_stats", {}).get('r_squared')
            fit_label = f'Ajuste Polinómico Final (Grado {len(final_results["combined_poly_coeffs"])-1})'
            if r_squared is not None:
                fit_label += f'\n$R^2={r_squared:.4f}$'

        # --- Configuración del Gráfico ---
        ax.set_title(f"MTF Final Combinada ({ROI1_LABEL} & {ROI2_LABEL}) y Ajuste Polinómico\n{processed_files_count} archivos procesados")
        ax.set_xlabel('Frecuencia Espacial [ciclos/mm]')
        ax.set_ylabel('MTF')
        ax.set_xlim(left=0, right=GLOBAL_CUTOFF_FREQ)
        ax.set_ylim(bottom=-0.05, top=1.1)
        ax.grid(True, which='major', ls='-', lw=0.6); ax.grid(True, which='minor', ls=':', lw=0.4)
        ax.minorticks_on()
        ax.legend(fontsize='medium')
        plt.tight_layout()

        # Mostrar el gráfico final
        print("\nMostrando gráfico final combinado...")
        plt.show()

    elif plot_combined_final_results:
         print("\nNo se generó gráfico final (no se pudo calcular el promedio combinado V&H).")

    # --- Opcional: Guardar/Exportar el ajuste polinómico final ---
    if final_results["combined_poly_coeffs"] is not None:
        print("\n--- Coeficientes del Polinomio Final (Mayor a Menor Grado) ---")
        print(final_results["combined_poly_coeffs"])
        # Aquí podrías guardar los coeficientes en un archivo .npy o .txt
        # np.save("mtf_poly_coeffs.npy", final_results["combined_poly_coeffs"])
        # np.savetxt("mtf_poly_coeffs.txt", final_results["combined_poly_coeffs"])

    print("\n--- Proceso Completado ---") # Usar formato LaTeX para R²
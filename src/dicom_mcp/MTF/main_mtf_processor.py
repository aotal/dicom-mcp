# main_mtf_processor.py
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob

from roi_extractor import RoiExtractor
from mtf_analyzer import MtfAnalyzer
from utils import get_vmp_k_calibration_data

if __name__ == "__main__":
    """Main script to perform MTF analysis on a folder of DICOM images."""

    # --- Configuration ---
    dicom_folder = "src/MTF/"
    dicom_files = glob.glob(os.path.join(dicom_folder, '*.dcm'))

    if not dicom_files:
        print(f"Error: No .dcm files found in the folder: {dicom_folder}")
        exit()

    print(f"Found {len(dicom_files)} DICOM files in '{dicom_folder}'.")

    roi1_offset_mm = (13, 0); roi1_shape_yx = (100, 200)  # -> Vertical MTF
    roi2_offset_mm = (0, -14); roi2_shape_yx = (200, 100) # -> Horizontal MTF
    ROI1_LABEL = "Vertical"
    ROI2_LABEL = "Horizontal"

    mtf_params = {
        'sub_pixel_factor': 0.1, 'smoothing_window_bins': 17, 'smoothing_polyorder': 4,
        'baseline_tail_threshold_mm': 7.0, 'window_width_mm': 20.0, 'cutoff_freq': 3.7,
    }
    GLOBAL_CUTOFF_FREQ = mtf_params['cutoff_freq']
    NUM_POINTS_INTERP = 250

    calibration_csv_path = "src/data/linearizacion.csv"
    RQA_FACTORS = {
        'RQA5': 1.23456e6, 'RQA9': 2.34567e6,
    }

    plot_individual_analyses = False
    plot_average_mtf = False
    plot_combined_final_results = True
    plot_angle_fits = False
    fit_combined_average_poly = True
    plot_VH_averages_on_final = True
    plot_combined_average_on_final = True
    plot_final_poly_fit = True
    plot_std_dev_on_final = False
    plot_individuals_on_final = False

    print("--- Starting MTF Processing for Combined Polynomial Fit ---")

    print("  [0] Loading calibration data...")
    calibration_df = get_vmp_k_calibration_data(calibration_csv_path)
    if calibration_df is None: exit("Fatal Error: Missing calibration data.")
    print("  Calibration data loaded OK.")

    print("  [*] Creating MtfAnalyzer instance...")
    try:
        analyzer = MtfAnalyzer(calibration_df=calibration_df, rqa_factors_dict=RQA_FACTORS, **mtf_params)
        print("  MtfAnalyzer instance created OK.")
    except Exception as e: exit(f"Fatal error initializing MtfAnalyzer: {e}")

    vertical_mtf_results = []
    horizontal_mtf_results = []
    processed_files_count = 0
    error_files_count = 0

    for dicom_path in dicom_files:
        print(f"\n>>> Processing File: {os.path.basename(dicom_path)} <<<")
        file_processed_ok = False
        try:
            print("  [1] Extracting RAW ROIs...")
            extractor = RoiExtractor(dicom_path, verbose=False)
            pixel_spacing = extractor.pixel_spacing
            dicom_dataset = extractor.ds
            if pixel_spacing is None: raise ValueError("Pixel spacing not found.")
            rois_raw = extractor.extract_mtf_rois(roi1_offset_mm, roi1_shape_yx, roi2_offset_mm, roi2_shape_yx)
            if not rois_raw or len(rois_raw) != 2 or rois_raw[0].size == 0 or rois_raw[1].size == 0:
                 raise ValueError("Failed to extract RAW ROIs.")
            print("  RAW ROIs extracted OK.")

            print("  [2] Determining RQA Type...")
            current_rqa_type = 'RQA5'
            print(f"    RQA Type: {current_rqa_type}")
            if current_rqa_type not in analyzer.rqa_factors_dict:
                 raise ValueError(f"RQA Type '{current_rqa_type}' unknown.")

            print(f"  [3] Analyzing ROI 1 ({ROI1_LABEL})...")
            results1 = analyzer.analyze_roi(rois_raw[0], pixel_spacing, current_rqa_type,
                                            roi_id=f"{os.path.basename(dicom_path)}-{ROI1_LABEL}",
                                            verbose=False, plot_angle_fit=plot_angle_fits)
            if results1 and "OK" in results1.get("status", "") and results1.get("frequencies") is not None:
                print(f"    ROI 1 ({ROI1_LABEL}) MTF OK.")
                vertical_mtf_results.append(results1)
            else:
                 print(f"    Warning: ROI 1 ({ROI1_LABEL}) did not produce a valid MTF.")

            print(f"  [4] Analyzing ROI 2 ({ROI2_LABEL})...")
            results2 = analyzer.analyze_roi(rois_raw[1], pixel_spacing, current_rqa_type,
                                            roi_id=f"{os.path.basename(dicom_path)}-{ROI2_LABEL}",
                                            verbose=False, plot_angle_fit=plot_angle_fits)
            if results2 and "OK" in results2.get("status", "") and results2.get("frequencies") is not None:
                print(f"    ROI 2 ({ROI2_LABEL}) MTF OK.")
                horizontal_mtf_results.append(results2)
            else:
                 print(f"    Warning: ROI 2 ({ROI2_LABEL}) did not produce a valid MTF.")

            file_processed_ok = True

        except Exception as e:
             print(f"  ERROR processing {os.path.basename(dicom_path)}: {e}")
             error_files_count += 1

        if file_processed_ok:
            processed_files_count += 1

    print(f"\n--- End of Processing Loop ---")
    print(f"Files processed (attempted): {len(dicom_files)}")
    print(f"Files with fatal errors: {error_files_count}")
    print(f"Total VALID ROIs accumulated ({ROI1_LABEL}): {len(vertical_mtf_results)}")
    print(f"Total VALID ROIs accumulated ({ROI2_LABEL}): {len(horizontal_mtf_results)}")

    final_results = {
        "vert_freq": None, "vert_avg": None,
        "horiz_freq": None, "horiz_avg": None,
        "combined_freq": None, "combined_avg": None,
        "combined_poly_coeffs": None, "combined_poly_func": None,
        "fit_stats": None
    }

    if len(vertical_mtf_results) >= 2:
        print(f"\n--- Calculating Global Average MTF ({ROI1_LABEL}) ---")
        final_results["vert_freq"], final_results["vert_avg"], _, _ = \
            MtfAnalyzer.calculate_grand_average_mtf(
                vertical_mtf_results, num_points=NUM_POINTS_INTERP, forced_cutoff_freq=GLOBAL_CUTOFF_FREQ
            )
    else:
        print(f"\nNot enough results to calculate global average for {ROI1_LABEL}.")

    if len(horizontal_mtf_results) >= 2:
        print(f"\n--- Calculating Global Average MTF ({ROI2_LABEL}) ---")
        final_results["horiz_freq"], final_results["horiz_avg"], _, _ = \
            MtfAnalyzer.calculate_grand_average_mtf(
                horizontal_mtf_results, num_points=NUM_POINTS_INTERP, forced_cutoff_freq=GLOBAL_CUTOFF_FREQ
            )
    else:
         print(f"\nNot enough results to calculate global average for {ROI2_LABEL}.")

    if final_results["vert_avg"] is not None and final_results["horiz_avg"] is not None:
        print("\n--- Calculating Combined V&H Average ---")
        if np.allclose(final_results["vert_freq"], final_results["horiz_freq"], rtol=1e-5, atol=1e-8):
            final_results["combined_freq"] = final_results["vert_freq"]
            final_results["combined_avg"] = (final_results["vert_avg"] + final_results["horiz_avg"]) / 2.0
            print("  Combined V&H average calculated.")

            if fit_combined_average_poly:
                print("\n--- Fitting Polynomial to Combined V&H Average ---")
                final_results["combined_poly_coeffs"], final_results["combined_poly_func"], final_results["fit_stats"] = \
                    analyzer.fit_average_mtf_polynomial(
                        final_results["combined_freq"], final_results["combined_avg"], degree=4
                    )
                if final_results["combined_poly_func"]:
                    print(f"  Final polynomial fit OK. Coeffs: {final_results['combined_poly_coeffs']}")
                else:
                    print("  Final polynomial fit failed.")
        else:
            print("Error: Vertical and Horizontal average frequency axes differ significantly. Cannot calculate combined average.")

    else:
        print("\nCannot calculate combined V&H average (missing V or H data).")

    if plot_combined_final_results and final_results["combined_avg"] is not None:
        print("\n[Plotting Combined Final Results]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        plt.figure("Final Combined MTF and Fit", figsize=(10, 7)); plt.clf()
        ax = plt.gca()

        if plot_VH_averages_on_final and final_results["vert_avg"] is not None:
            ax.plot(final_results["vert_freq"], final_results["vert_avg"],
                    color='blue', ls=':', lw=1.5, marker='None',
                    label=f'Average {ROI1_LABEL} ({len(vertical_mtf_results)} ROIs)')

        if plot_VH_averages_on_final and final_results["horiz_avg"] is not None:
             ax.plot(final_results["horiz_freq"], final_results["horiz_avg"],
                     color='red', ls=':', lw=1.5, marker='None',
                     label=f'Average {ROI2_LABEL} ({len(horizontal_mtf_results)} ROIs)')

        if plot_combined_average_on_final:
             ax.plot(final_results["combined_freq"], final_results["combined_avg"],
                     color='black', ls='-', lw=2.5, marker='None',
                     label='Combined V&H Average')

        if plot_final_poly_fit and final_results["combined_poly_func"] is not None:
            poly_fit_vals = np.clip(final_results["combined_poly_func"](final_results["combined_freq"]), 0, 1)
            r_squared = final_results.get("fit_stats", {}).get('r_squared')
            fit_label = f'Final Polynomial Fit (Degree {len(final_results["combined_poly_coeffs"])-1})'
            if r_squared is not None:
                fit_label += f'\n$R^2={r_squared:.4f}$'

        ax.set_title(f"Final Combined MTF ({ROI1_LABEL} & {ROI2_LABEL}) and Polynomial Fit\n{processed_files_count} files processed")
        ax.set_xlabel('Spatial Frequency [cycles/mm]')
        ax.set_ylabel('MTF')
        ax.set_xlim(left=0, right=GLOBAL_CUTOFF_FREQ)
        ax.set_ylim(bottom=-0.05, top=1.1)
        ax.grid(True, which='major', ls='-', lw=0.6); ax.grid(True, which='minor', ls=':', lw=0.4)
        ax.minorticks_on()
        ax.legend(fontsize='medium')
        plt.tight_layout()

        print("\nShowing final combined plot...")
        plt.show()

    elif plot_combined_final_results:
         print("\nFinal plot not generated (could not calculate combined V&H average).")

    if final_results["combined_poly_coeffs"] is not None:
        print("\n--- Final Polynomial Coefficients (Highest to Lowest Degree) ---")
        print(final_results["combined_poly_coeffs"])

    print("\n--- Process Complete ---")

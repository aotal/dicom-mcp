# mtf_analyzer.py

import numpy as np
import math
import os
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import warnings

try:
    from .utils import linearize_preprocessed_image_from_df
except ImportError:
    from utils import linearize_preprocessed_image_from_df

def estimate_angle_from_midpoints_vectorized(oriented_roi_array, plot_fit=False, verbose=True):
    """Estimates the geometric angle of an edge versus the vertical axis from an oriented ROI.

    This is a vectorized version using NumPy for performance.

    Args:
        oriented_roi_array (np.ndarray): The ROI as a NumPy array.
        plot_fit (bool): If True, plots the linear regression fit.
        verbose (bool): If True, prints detailed messages.

    Returns:
        float: The estimated angle in degrees, or None if it fails.
    """
    if verbose: print(f"\n--- Estimating angle using vectorized midpoint method ---")
    image_data_linear = oriented_roi_array
    if image_data_linear is None or image_data_linear.size == 0:
        print("Error: ROI array is empty or invalid for angle estimation.")
        return None

    rows, cols = image_data_linear.shape
    if verbose: print(f"Analyzing oriented ROI of shape: {rows}x{cols}")

    with np.errstate(all='ignore'):
        row_mins = np.nanmin(image_data_linear, axis=1)
        row_maxs = np.nanmax(image_data_linear, axis=1)
    row_ranges = row_maxs - row_mins

    global_min, global_max = np.nanmin(row_mins), np.nanmax(row_maxs)
    min_range_threshold = (global_max - global_min) * 0.05
    if min_range_threshold < 1e-6: min_range_threshold = 1e-6

    valid_row_mask = (row_ranges >= min_range_threshold) & ~np.isnan(row_ranges)
    
    if not np.any(valid_row_mask):
         print(f"Error: No rows with sufficient dynamic range found.")
         return None

    valid_rows_data = image_data_linear[valid_row_mask, :]
    valid_row_indices = np.where(valid_row_mask)[0]
    valid_row_mins = row_mins[valid_row_mask]
    valid_row_maxs = row_maxs[valid_row_mask]
    mid_vals = (valid_row_mins + valid_row_maxs) / 2.0

    centered_data = valid_rows_data - mid_vals[:, np.newaxis]
    sign_changes = np.diff(np.signbit(centered_data), axis=1)
    crossing_locations = (sign_changes != 0)

    rows_with_single_crossing_mask = np.sum(crossing_locations, axis=1) == 1
    
    if not np.any(rows_with_single_crossing_mask):
        print(f"Error: No rows with a single midpoint crossing found.")
        return None

    single_crossing_filtered_indices = np.where(rows_with_single_crossing_mask)[0]

    final_rows_data = valid_rows_data[rows_with_single_crossing_mask, :]
    final_mid_vals = mid_vals[rows_with_single_crossing_mask]
    final_original_indices = valid_row_indices[rows_with_single_crossing_mask]

    j_indices = np.argmax(crossing_locations[rows_with_single_crossing_mask, :], axis=1)

    row_idx_for_indexing = np.arange(len(final_original_indices))
    val_j = final_rows_data[row_idx_for_indexing, j_indices]
    val_j1 = final_rows_data[row_idx_for_indexing, j_indices + 1]

    delta_val = val_j1 - val_j
    
    x_mid = np.full(final_original_indices.shape, np.nan)

    near_zero_delta_mask = np.abs(delta_val) < 1e-9
    x_mid[near_zero_delta_mask] = j_indices[near_zero_delta_mask] + 0.5

    non_zero_delta_mask = ~near_zero_delta_mask
    fraction = np.full(final_original_indices.shape, np.nan)
    fraction[non_zero_delta_mask] = (final_mid_vals[non_zero_delta_mask] - val_j[non_zero_delta_mask]) / delta_val[non_zero_delta_mask]

    valid_fraction_mask = (fraction >= 0.0) & (fraction <= 1.0) & non_zero_delta_mask

    x_mid[valid_fraction_mask] = j_indices[valid_fraction_mask] + fraction[valid_fraction_mask]

    final_valid_points_mask = (near_zero_delta_mask | valid_fraction_mask) & \
                              (x_mid >= 0) & (x_mid < cols)

    y_coords = final_original_indices[final_valid_points_mask]
    x_coords = x_mid[final_valid_points_mask]

    if len(y_coords) < 5:
        print(f"Error: Not enough valid midpoints found ({len(y_coords)}) after interpolation.")
        return None
    if verbose: print(f"Found {len(y_coords)} valid midpoints.")

    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_coords, x_coords)
        r_squared = r_value**2
        if verbose: print(f"Linear fit: x_mid = {slope:.5f}*y + {intercept:.3f}, R²={r_squared:.4f} (p={p_value:.3g})")
        if r_squared < 0.90: print(f"Warning: Unreliable linear fit (R²={r_squared:.4f}).")

        estimated_angle_deg_vertical_ref = math.degrees(math.atan(slope)) if abs(slope) >= 1e-9 else 0.0
        if verbose: print(f"Geometric angle (vs. vertical): {estimated_angle_deg_vertical_ref:.3f} degrees")

        if plot_fit:
            plt.figure("Midpoint Fit (Vectorized)", figsize=(8, 6)); plt.clf()
            plt.scatter(x_coords, y_coords, label=f'Midpoints ({len(x_coords)})', alpha=0.7, s=10)
            y_fit = np.array([y_coords.min(), y_coords.max()]); x_fit = slope * y_fit + intercept
            plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit\nAngle (vs V) = {estimated_angle_deg_vertical_ref:.2f}°\nR² = {r_squared:.3f}')
            plt.title('Midpoint Fit (Oriented ROI - Vectorized)'); plt.xlabel('Column (x_mid)'); plt.ylabel('Row (y)')
            plt.gca().invert_yaxis(); plt.legend(); plt.grid(True, ls=':'); plt.axis('equal'); plt.tight_layout()

        return estimated_angle_deg_vertical_ref
    except Exception as e:
        print(f"Linear fit error: {e}")
        return None

def estimate_angle_from_midpoints(oriented_roi_array, plot_fit=False, verbose=True):
    """Estimates the geometric angle of an edge versus the vertical axis from an oriented ROI.

    Args:
        oriented_roi_array (np.ndarray): The ROI as a NumPy array.
        plot_fit (bool): If True, plots the linear regression fit.
        verbose (bool): If True, prints detailed messages.

    Returns:
        float: The estimated angle in degrees, or None if it fails.
    """
    if verbose: print(f"\n--- Estimating angle using midpoint method ---")
    image_data_linear = oriented_roi_array
    if image_data_linear is None or image_data_linear.size == 0:
         print("Error: ROI array is empty or invalid for angle estimation.")
         return None
    rows, cols = image_data_linear.shape
    if verbose: print(f"Analyzing oriented ROI of shape: {rows}x{cols}")

    row_indices = []; midpoint_cols = []
    global_min, global_max = np.nanmin(image_data_linear), np.nanmax(image_data_linear)
    min_range_threshold = (global_max - global_min) * 0.05
    if min_range_threshold < 1e-6: min_range_threshold = 1e-6

    for i in range(rows):
        profile = image_data_linear[i, :]
        if np.isnan(profile).any(): continue
        min_val, max_val = np.min(profile), np.max(profile); profile_range = max_val - min_val
        if profile_range < min_range_threshold: continue
        mid_val = (min_val + max_val) / 2.0
        try: crossing_indices = np.where(np.diff(np.signbit(profile - mid_val)))[0]
        except Exception: continue
        if len(crossing_indices) == 1:
            j = crossing_indices[0]; val_j, val_j1 = profile[j], profile[j+1]
            if abs(val_j1 - val_j) < 1e-9: x_mid = j + 0.5
            else: fraction = (mid_val - val_j) / (val_j1 - val_j)
            if 0.0 <= fraction <= 1.0: x_mid = j + fraction
            else: continue
            if 0 <= x_mid < cols: row_indices.append(i); midpoint_cols.append(x_mid)

    if len(row_indices) < 5:
        print(f"Error: Not enough valid midpoints found ({len(row_indices)}).")
        return None
    if verbose: print(f"Found {len(row_indices)} valid midpoints.")
    y_coords, x_coords = np.array(row_indices), np.array(midpoint_cols)

    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_coords, x_coords); r_squared = r_value**2
        if verbose: print(f"Linear fit: x_mid = {slope:.5f}*y + {intercept:.3f}, R²={r_squared:.4f} (p={p_value:.3g})")
        if r_squared < 0.90: print(f"Warning: Unreliable linear fit (R²={r_squared:.4f}).")
        estimated_angle_deg_vertical_ref = math.degrees(math.atan(slope)) if abs(slope) >= 1e-9 else 0.0
        if verbose: print(f"Geometric angle (vs. vertical): {estimated_angle_deg_vertical_ref:.3f} degrees")

        if plot_fit:
            plt.figure("Midpoint Fit", figsize=(8, 6)); plt.clf()
            plt.scatter(x_coords, y_coords, label=f'Midpoints ({len(x_coords)})', alpha=0.7, s=10)
            y_fit = np.array([y_coords.min(), y_coords.max()]); x_fit = slope * y_fit + intercept
            plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit\nAngle (vs V) = {estimated_angle_deg_vertical_ref:.2f}°\nR² = {r_squared:.3f}')
            plt.title('Midpoint Fit (Oriented ROI)'); plt.xlabel('Column (x_mid)'); plt.ylabel('Row (y)')
            plt.gca().invert_yaxis(); plt.legend(); plt.grid(True, ls=':'); plt.axis('equal'); plt.tight_layout()

        return estimated_angle_deg_vertical_ref
    except Exception as e: print(f"Linear fit error: {e}"); return None

def calculate_esf_from_roi(oriented_roi_array, angle_deg, pixel_size_mm,
                           sub_pixel_factor=0.1, smoothing_window_bins=17,
                           smoothing_polyorder=4, verbose=True):
    """Calculates the Edge Spread Function (ESF) from an oriented ROI.

    Args:
        oriented_roi_array (np.ndarray): The oriented ROI as a NumPy array.
        angle_deg (float): The estimated angle of the edge in degrees.
        pixel_size_mm (float): The pixel spacing in millimeters.
        sub_pixel_factor (float): The sub-pixel sampling factor.
        smoothing_window_bins (int): The window size for Savitzky-Golay smoothing.
        smoothing_polyorder (int): The polynomial order for Savitzky-Golay smoothing.
        verbose (bool): If True, prints detailed messages.

    Returns:
        tuple: A tuple containing s_coords, esf_smoothed, esf_raw, and delta_s.
    """
    image_data_linear = oriented_roi_array
    if image_data_linear is None or image_data_linear.size == 0: return None, None, None, None
    rows, cols = image_data_linear.shape
    if pixel_size_mm is None or pixel_size_mm <= 0: print(f"Error: Invalid pixel_size_mm ({pixel_size_mm})"); return None, None, None, None
    if verbose: print(f"\nCalculating ESF for angle={angle_deg:.3f} deg...\n Using ROI of shape: {rows}x{cols}\n Parameters: Pixel={pixel_size_mm:.4f}mm, SubBin={sub_pixel_factor}")
    angle_rad = math.radians(angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
    delta_s = sub_pixel_factor * pixel_size_mm; center_i = (rows - 1) / 2.0; center_j = (cols - 1) / 2.0
    if verbose: print(f"  Bin width (Δs)={delta_s:.5f} mm\n ROI data range: [{image_data_linear.min():.2f}, {image_data_linear.max():.2f}]")

    binned_values = {}
    if verbose: print(" Starting ESF re-projection and binning...")
    for i in range(rows):
        for j in range(cols):
            s = pixel_size_mm * ((j - center_j) * cos_a - (i - center_i) * sin_a)
            bin_index = math.floor(s / delta_s)
            if bin_index not in binned_values: binned_values[bin_index] = []
            binned_values[bin_index].append(image_data_linear[i, j])
    if verbose: print(f" ESF binning complete. Initial bins: {len(binned_values)}")
    if not binned_values: print(f"Error! No ESF bins were created."); return None, None, None, None

    try:
        sorted_bin_indices = sorted(binned_values.keys()); min_idx, max_idx = sorted_bin_indices[0], sorted_bin_indices[-1]
        all_indices = list(range(min_idx, max_idx + 1)); s_coords = np.array([(idx + 0.5) * delta_s for idx in all_indices])
        esf_raw = np.full(len(all_indices), np.nan)
        for k, idx in enumerate(all_indices):
            if idx in binned_values:
                valid_vals = [v for v in binned_values[idx] if not np.isnan(v)]
                if valid_vals: esf_raw[k] = np.mean(valid_vals)
        nan_indices = np.where(np.isnan(esf_raw))[0]
        if len(nan_indices) > 0:
            if verbose: print(f" Warning: {len(nan_indices)} empty bins. Interpolating...")
            is_nan = np.isnan(esf_raw); x_all = np.arange(len(esf_raw))
            if not np.all(is_nan): esf_raw[is_nan] = np.interp(x_all[is_nan], x_all[~is_nan], esf_raw[~is_nan])
            if np.isnan(esf_raw).any():
                if verbose: print("  Could not interpolate all (endpoints?). Trimming.")
                valid_mask = ~np.isnan(esf_raw); s_coords = s_coords[valid_mask]; esf_raw = esf_raw[valid_mask]
        if verbose: print(f" Raw ESF calculated. Points: {len(esf_raw)}. s range: [{s_coords.min():.3f}, {s_coords.max():.3f}] mm\n Raw ESF range: [{np.nanmin(esf_raw):.2f}, {np.nanmax(esf_raw):.2f}]")
    except Exception as e: print(f"Error calculating raw ESF: {e}"); return None, None, None, None

    if len(esf_raw) < max(2, smoothing_polyorder + 1):
         print(f"Error! Raw ESF < {max(2, smoothing_polyorder + 1)} points ({len(esf_raw)}).");
         return s_coords, None, esf_raw, delta_s

    esf_smoothed = None
    effective_smoothing_window = smoothing_window_bins + (smoothing_window_bins % 2 == 0)
    if len(esf_raw) < effective_smoothing_window: effective_smoothing_window = len(esf_raw) - (len(esf_raw) % 2 == 0)
    if effective_smoothing_window < 3: effective_smoothing_window = 0

    if effective_smoothing_window >= 3:
        actual_polyorder = min(smoothing_polyorder, effective_smoothing_window - 1); actual_polyorder = max(1, actual_polyorder)
        if verbose: print(f" Applying Savitzky-Golay (w={effective_smoothing_window}, p={actual_polyorder})...")
        try: esf_smoothed = savgol_filter(esf_raw.astype(np.float64), effective_smoothing_window, actual_polyorder)
        except Exception as e: print(f" Savitzky-Golay smoothing error: {e}. Returning raw ESF."); esf_smoothed = esf_raw.copy()
    else:
        if verbose: print(" Skipping smoothing (too few points or invalid window)."); esf_smoothed = esf_raw.copy()

    if esf_smoothed is None: esf_smoothed = esf_raw.copy()
    if verbose and esf_smoothed is not esf_raw: print(f" Smoothed ESF calculated. Range: [{esf_smoothed.min():.2f}, {esf_smoothed.max():.2f}]")
    return s_coords, esf_smoothed, esf_raw, delta_s

def calculate_lsf( s_coords, esf_smoothed, delta_s,
                  baseline_tail_threshold_mm=7.0, window_width_mm=20.0, verbose=True):
    """Calculates the Line Spread Function (LSF) from the ESF.

    This function finds the absolute peak and inverts the LSF if it's negative.

    Args:
        s_coords (np.ndarray): The spatial coordinates.
        esf_smoothed (np.ndarray): The smoothed ESF.
        delta_s (float): The spatial sampling interval.
        baseline_tail_threshold_mm (float): The threshold for defining the LSF tails.
        window_width_mm (float): The width of the Hanning window for the LSF.
        verbose (bool): If True, prints detailed messages.

    Returns:
        tuple: A tuple containing the windowed spatial coordinates and the final LSF.
    """
    if esf_smoothed is None or s_coords is None or len(esf_smoothed) < 2 or delta_s is None: return None, None
    if verbose: print(f"\n--- Calculating LSF --- \n Spacing (Δs) for differentiation: {delta_s:.5f} mm")
    try: lsf_raw = np.gradient(esf_smoothed, delta_s)
    except Exception as e: print(f"Error in np.gradient for LSF: {e}"); return None, None
    if verbose: print(f" Raw LSF (diff.) Points: {len(lsf_raw)}. Range: [{lsf_raw.min():.2f}, {lsf_raw.max():.2f}]")

    lsf_baseline_corrected = lsf_raw.copy(); s_windowed = None; lsf_windowed = None
    try:
        peak_index = np.argmax(np.abs(lsf_raw))
        s_peak = s_coords[peak_index]; lsf_peak_value = lsf_raw[peak_index]
        if verbose: print(f" Raw LSF peak (max abs) at s={s_peak:.3f}mm (idx {peak_index}, val={lsf_peak_value:.2f})")
        tail_indices = np.where(np.abs(s_coords - s_peak) > baseline_tail_threshold_mm)[0]
        if len(tail_indices) > 5:
            s_tails = s_coords[tail_indices]; lsf_tails = lsf_raw[tail_indices]; coeffs = np.polyfit(s_tails, lsf_tails, 1)
            baseline = np.polyval(coeffs, s_coords); lsf_baseline_corrected = lsf_raw - baseline
            if verbose: print(f" Baseline fit (tail>{baseline_tail_threshold_mm}mm): Slope={coeffs[0]:.4f}, Int={coeffs[1]:.4f}\n Baseline subtracted.")
        else:
            if verbose: print(f" Warning: Too few points ({len(tail_indices)}) in tails. Skipping subtraction.")
    except Exception as e: print(f" Baseline subtraction error: {e}. Using raw LSF."); lsf_baseline_corrected = lsf_raw.copy()
    if verbose: print(f" Baseline-corrected LSF. Range: [{lsf_baseline_corrected.min():.2f}, {lsf_baseline_corrected.max():.2f}]")

    try:
        peak_index_corrected = np.argmax(np.abs(lsf_baseline_corrected))
        s_peak_corrected = s_coords[peak_index_corrected]; lsf_peak_corrected_value = lsf_baseline_corrected[peak_index_corrected]
        if verbose: print(f" Corrected LSF peak (max abs) at s={s_peak_corrected:.3f}mm (idx {peak_index_corrected}, val={lsf_peak_corrected_value:.2f})")
        window_half_width = window_width_mm/2.0; window_indices = np.where(np.abs(s_coords - s_peak_corrected) <= window_half_width)[0]
        if len(window_indices) < 2: print(f" Error: Too few LSF points ({len(window_indices)}) in window {window_width_mm}mm."); return None, None
        s_windowed = s_coords[window_indices]; lsf_to_window = lsf_baseline_corrected[window_indices]
        hanning_win = np.hanning(len(lsf_to_window)); lsf_windowed = lsf_to_window * hanning_win
        if verbose:
            w_width_act = s_windowed[-1] - s_windowed[0] if len(s_windowed) > 1 else 0
            print(f" Hanning window applied (nominal width={window_width_mm}mm, actual~{w_width_act:.1f}mm, {len(lsf_windowed)} pts).")
            print(f" Final (windowed) LSF. Range: [{lsf_windowed.min():.2f}, {lsf_windowed.max():.2f}]")
    except Exception as e: print(f" LSF windowing error: {e}."); return None, None

    if lsf_windowed is None: return None, None
    peak_val_windowed = lsf_windowed[np.argmax(np.abs(lsf_windowed))] if len(lsf_windowed) > 0 else 0
    if peak_val_windowed < -1e-9:
        if verbose: print(" Warning: Windowed LSF peak is negative. Inverting LSF for MTF.")
        lsf_windowed = -lsf_windowed
    return s_windowed, lsf_windowed

def calculate_mtf(s_lsf, lsf_final, delta_s, verbose=True, cutoff_freq=3.7):
    """Calculates the Modulation Transfer Function (MTF) from the final LSF.

    Args:
        s_lsf (np.ndarray): The spatial coordinates for the LSF.
        lsf_final (np.ndarray): The final, windowed LSF.
        delta_s (float): The spatial sampling interval.
        verbose (bool): If True, prints detailed messages.
        cutoff_freq (float): The cutoff frequency for the MTF.

    Returns:
        tuple: A tuple containing the frequencies and the MTF.
    """
    if lsf_final is None or s_lsf is None or len(lsf_final) < 2 or delta_s is None or delta_s <= 0: print("Error: Invalid LSF or delta_s for MTF."); return None, None
    if verbose: print("\n--- Calculating MTF ---")
    try:
        area_lsf = trapezoid(lsf_final, dx=delta_s)
        if abs(area_lsf) < 1e-9: print("Error: LSF area is close to zero."); return None, None
        lsf_normalized = lsf_final / area_lsf
        if verbose: print(f" Original LSF area: {area_lsf:.4f}"); area_check=trapezoid(lsf_normalized, dx=delta_s); print(f" Normalized LSF. Area check: {area_check:.4f}")
    except Exception as e: print(f"LSF normalization error: {e}"); return None, None
    try:
        N = len(lsf_normalized); otf = fft(lsf_normalized)
        if verbose: print(f" OTF calculated (FFT, N={N} points).")
    except Exception as e: print(f"FFT calculation error: {e}"); return None, None
    try:
        frequencies_all = fftfreq(N, d=delta_s); nyquist_freq = 0.5 / delta_s
        if verbose: print(f" Frequency axis generated (d={delta_s:.5f}mm). Nyquist={nyquist_freq:.2f} c/mm")
    except Exception as e: print(f"Frequency axis generation error: {e}"); return None, None
    try:
        mtf_all = np.abs(otf); valid_freq_mask = (frequencies_all >= -1e-9) & (frequencies_all <= cutoff_freq + 1e-9)
        frequencies_filtered = frequencies_all[valid_freq_mask]; mtf_filtered = mtf_all[valid_freq_mask]
        sort_indices = np.argsort(frequencies_filtered); frequencies = frequencies_filtered[sort_indices]; mtf = mtf_filtered[sort_indices]
        if len(mtf) > 0:
            mtf0 = mtf[0]
            if abs(mtf0) > 1e-9:
                if verbose and not np.isclose(mtf0, 1.0, atol=1e-3): print(f" Normalizing MTF(0) (was {mtf0:.4f}) to 1.0.")
                mtf = mtf / mtf0
            elif verbose: print(" Warning: MTF(0) is close to zero, setting to 1."); mtf[0] = 1.0
            mtf = np.clip(mtf, 0.0, 1.0)
        else: print(f"Error: No valid frequencies in [0, {cutoff_freq:.2f}] c/mm."); return None, None
        if verbose: print(f" MTF calculated and filtered [0, {cutoff_freq:.2f}] c/mm ({len(frequencies)} points).")
    except Exception as e: print(f"Error calculating/filtering MTF: {e}"); return None, None
    return frequencies, mtf

class MtfAnalyzer:
    """A class to perform a complete MTF analysis from a raw ROI, including internal linearization."""
    def __init__(self, sub_pixel_factor=0.1,
                 smoothing_window_bins=17,
                 smoothing_polyorder=4,
                 baseline_tail_threshold_mm=7.0,
                 window_width_mm=20.0,
                 cutoff_freq=3.7):
        """Initializes the analyzer with the analysis parameters.

        Args:
            sub_pixel_factor (float): The sub-sampling factor for the ESF.
            smoothing_window_bins (int): The window size for Savitzky-Golay ESF smoothing.
            smoothing_polyorder (int): The polynomial order for ESF smoothing.
            baseline_tail_threshold_mm (float): The threshold for defining the LSF tails.
            window_width_mm (float): The width of the Hanning window for the LSF.
            cutoff_freq (float): The cutoff frequency for the MTF.
        """
        self.params = {
            'sub_pixel_factor': sub_pixel_factor,
            'smoothing_window_bins': smoothing_window_bins,
            'smoothing_polyorder': smoothing_polyorder,
            'baseline_tail_threshold_mm': baseline_tail_threshold_mm,
            'window_width_mm': window_width_mm,
            'cutoff_freq': cutoff_freq,
        }
        self.sub_pixel_factor = sub_pixel_factor
        self.smoothing_window_bins = smoothing_window_bins
        self.smoothing_polyorder = smoothing_polyorder
        self.baseline_tail_threshold_mm = baseline_tail_threshold_mm
        self.window_width_mm = window_width_mm
        self.cutoff_freq = cutoff_freq

        print(f"DEBUG: MtfAnalyzer initialized with analysis parameters.")


    def analyze_roi(self, linearized_roi_array: np.ndarray, pixel_spacing: float,
                    roi_id="ROI", verbose=True, plot_angle_fit=False):
        """Performs a complete ESF/LSF/MTF analysis for a single, pre-linearized ROI.

        Args:
            linearized_roi_array (np.ndarray): The linearized ROI as a NumPy array.
            pixel_spacing (float): The pixel spacing in millimeters.
            roi_id (str): An identifier for logging and plotting.
            verbose (bool): If True, prints detailed messages.
            plot_angle_fit (bool): If True, plots the angle fit.

        Returns:
            dict: A dictionary containing the analysis results.
        """
        print(f"\n=== Analyzing {roi_id} ===")
        if linearized_roi_array is None or linearized_roi_array.size == 0:
            print(f"Error: Input array for {roi_id} is empty.")
            return {"status": "Error - Empty input ROI", "roi_id": roi_id}

        if pixel_spacing is None or pixel_spacing <= 0:
            print(f"Error: Invalid pixel spacing ({pixel_spacing}) for {roi_id}.")
            return {"status": "Error - Invalid Pixel Spacing", "roi_id": roi_id}

        original_shape = linearized_roi_array.shape
        print(f" {roi_id} - Input shape: {original_shape}")

        oriented_roi = linearized_roi_array
        rows_orig, cols_orig = oriented_roi.shape
        rotated_flag = False
        if rows_orig > cols_orig:
            if verbose: print(f"  [Step 1] {roi_id} - Rotating 90 degrees...")
            oriented_roi = np.rot90(oriented_roi)
            rotated_flag = True
            if verbose: print(f"     Oriented shape: {oriented_roi.shape}")
        else:
            if verbose: print(f"  [Step 1] {roi_id} - No rotation required.")

        if verbose: print(f"  [Step 2] Estimating angle (oriented ROI)...")
        angle_deg = estimate_angle_from_midpoints_vectorized(
            oriented_roi, plot_fit=plot_angle_fit, verbose=verbose
        )
        if angle_deg is None:
            print(f"Error: Angle estimation failed for {roi_id}.")
            return {"status": "Error - Angle Estimation", "roi_id": roi_id, "oriented_shape": oriented_roi.shape, "pixel_spacing": pixel_spacing}

        if verbose: print(f"  [Step 3] Calculating ESF...")
        s_esf, esf_smooth, esf_r, delta_s = calculate_esf_from_roi(
            oriented_roi_array=oriented_roi,
            angle_deg=angle_deg,
            pixel_size_mm=pixel_spacing,
            sub_pixel_factor=self.params['sub_pixel_factor'],
            smoothing_window_bins=self.params['smoothing_window_bins'],
            smoothing_polyorder=self.params['smoothing_polyorder'],
            verbose=verbose
        )
        if esf_smooth is None:
            print(f"Error: ESF calculation failed for {roi_id}.")
            return {"status": "Error - ESF Calculation", "roi_id": roi_id, "angle_deg": angle_deg}

        if verbose: print(f"  [Step 4] Calculating LSF...")
        s_lsf, lsf_final = calculate_lsf(
            s_coords=s_esf,
            esf_smoothed=esf_smooth,
            delta_s=delta_s,
            baseline_tail_threshold_mm=self.params['baseline_tail_threshold_mm'],
            window_width_mm=self.params['window_width_mm'],
            verbose=verbose
        )
        if lsf_final is None:
            print(f"Error: LSF calculation failed for {roi_id}.")
            return {"status": "Error - LSF Calculation", "roi_id": roi_id, "angle_deg": angle_deg, "s_esf": s_esf, "esf_smooth": esf_smooth, "esf_r": esf_r}

        if verbose: print(f"  [Step 5] Calculating MTF...")
        actual_delta_s_lsf = s_lsf[1] - s_lsf[0] if len(s_lsf) > 1 else delta_s
        
        frequencies, mtf = None, None
        if actual_delta_s_lsf is not None and len(s_lsf) > 1:
            frequencies, mtf = calculate_mtf(
                s_lsf=s_lsf,
                lsf_final=lsf_final,
                delta_s=actual_delta_s_lsf,
                verbose=verbose,
                cutoff_freq=self.params['cutoff_freq']
            )
        elif verbose:
            print(f"  Warning: Skipping MTF calculation due to insufficient LSF (points={len(s_lsf)}) or invalid delta_s.")

        if mtf is None:
            print(f"Warning: MTF calculation failed for {roi_id} (or insufficient LSF).")
            final_status = "Warning - MTF Failed"
        else:
            final_status = "OK"

        print(f"=== Analysis of {roi_id} complete ({final_status}) ===")
        return {
            "status": final_status,
            "roi_id": roi_id,
            "original_shape": original_shape, "pixel_spacing": pixel_spacing,
            "linearized_roi_stats": {"min": np.min(linearized_roi_array), "max": np.max(linearized_roi_array), "mean": np.mean(linearized_roi_array)},
            "rotated_flag": rotated_flag, "oriented_shape": oriented_roi.shape,
            "angle_deg": angle_deg,
            "s_esf": s_esf, "esf_smooth": esf_smooth, "esf_r": esf_r, "delta_s": delta_s,
            "s_lsf": s_lsf, "lsf_final": lsf_final,
            "frequencies": frequencies, "mtf": mtf
        }

    def calculate_average_mtf(self, results1, results2, num_points=250):
        """Calculates the average MTF from two analysis results."""
        print("\n--- Calculating Average MTF ---")
        mtf1 = results1.get("mtf") if results1 else None
        mtf2 = results2.get("mtf") if results2 else None
        freqs1 = results1.get("frequencies") if results1 else None
        freqs2 = results2.get("frequencies") if results2 else None

        if mtf1 is None or mtf2 is None or len(mtf1)<2 or len(mtf2)<2:
            print("Error: Cannot average MTFs (missing data or insufficient points).")
            return None, None

        try:
            max_freq = min(freqs1[-1], freqs2[-1], self.params['cutoff_freq'])
            common_freq_axis = np.linspace(0, max_freq, num=num_points)
            mtf1_interp = np.interp(common_freq_axis, freqs1, mtf1, left=1.0, right=0.0)
            mtf2_interp = np.interp(common_freq_axis, freqs2, mtf2, left=1.0, right=0.0)
            mtf_avg = (mtf1_interp + mtf2_interp) / 2.0
            print(" Average MTF calculated.")
            return common_freq_axis, mtf_avg
        except Exception as e:
            print(f"Error during MTF interpolation/averaging: {e}")
            return None, None
        
    def fit_average_mtf_polynomial(self, common_freq_axis, mtf_avg, degree=4):
        """Fits the average MTF curve to a polynomial, calculating R² and RMSE.

        Handles NaN values in the input.

        Args:
            common_freq_axis (np.ndarray): The common frequency axis.
            mtf_avg (np.ndarray): The average MTF values.
            degree (int): The degree of the polynomial to fit.

        Returns:
            tuple: A tuple containing the polynomial coefficients, the polynomial function, and fit statistics.
        """
        print(f"\n--- Fitting Average MTF to a Polynomial of Degree {degree} ---")
        if common_freq_axis is None or mtf_avg is None:
            print("Error: Missing frequency or average MTF data for fitting.")
            return None, None, None

        valid_mask = ~np.isnan(mtf_avg)
        freq_axis_valid = common_freq_axis[valid_mask]
        mtf_avg_valid = mtf_avg[valid_mask]

        if len(freq_axis_valid) < degree + 1:
            print(f"Error: Not enough valid points ({len(freq_axis_valid)}) to fit a polynomial of degree {degree}.")
            return None, None, None

        try:
            coeffs = np.polyfit(freq_axis_valid, mtf_avg_valid, degree)
            poly1d_func = np.poly1d(coeffs)

            mtf_predicted = poly1d_func(freq_axis_valid)
            ss_res = np.sum((mtf_avg_valid - mtf_predicted) ** 2)
            ss_tot = np.sum((mtf_avg_valid - np.mean(mtf_avg_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean((mtf_avg_valid - mtf_predicted) ** 2))
            
            fit_stats = {'r_squared': r_squared, 'rmse': rmse}

            if self.params.get('verbose', True):
                 print(f" Polynomial coefficients (degree {degree}): {coeffs}")
                 print(f" Goodness of Fit: R² = {r_squared:.5f}, RMSE = {rmse:.5f}")

            return coeffs, poly1d_func, fit_stats

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Error during polynomial fitting: {e}")
            return None, None, None
        except Exception as e_gen:
             print(f"Unexpected error during polynomial fitting: {e_gen}")
             return None, None, None

    def plot_single_roi_analysis(self, results, title_prefix=""):
        """Generates a 3-panel plot (ESF/LSF/MTF) for a single ROI."""
        if not results or results.get("status", "").startswith("Error"):
            print(f"DEBUG: Skipping plot for {results.get('roi_id','Unknown ROI')} due to status: {results.get('status')}")
            return

        roi_id = results.get("roi_id", "N/A")
        angle_deg = results.get("angle_deg")
        s_esf = results.get("s_esf"); esf_r = results.get("esf_r"); esf_smooth = results.get("esf_smooth")
        s_lsf = results.get("s_lsf"); lsf_final = results.get("lsf_final")
        frequencies = results.get("frequencies"); mtf = results.get("mtf")
        orig_shape = results.get("original_shape", "N/A"); rotated_flag = results.get("rotated_flag", False)
        cutoff_freq = self.params['cutoff_freq']
        ventana_suavizado_esf = self.params['smoothing_window_bins']
        polyorder_suavizado_esf = self.params['smoothing_polyorder']
        lsf_window_width_mm = self.params['window_width_mm']

        print(f"\n[Plotting Individual Results for {roi_id}]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
        try: fig.canvas.manager.set_window_title(f"Analysis {roi_id} - {title_prefix}")
        except AttributeError: pass

        title_roi_str = f"ROI #{roi_id} (Orig: {orig_shape})"
        if rotated_flag: title_roi_str += " [Rotated]"
        title_angle_str = f"Angle={angle_deg:.2f}°" if angle_deg is not None else "Angle?"

        ax0 = axes[0]
        if s_esf is not None and esf_smooth is not None and esf_r is not None:
            ax0.plot(s_esf, esf_r, 'c-', alpha=0.4, lw=1, label='Raw ESF')
            ax0.plot(s_esf, esf_smooth, 'b-', lw=1.5, label=f'Smoothed ESF (w={ventana_suavizado_esf}, p={polyorder_suavizado_esf})')
            ax0.set_title(f'ESF - {title_roi_str} - {title_angle_str}')
            ax0.set_ylabel('Linear Intensity'); ax0.legend(); ax0.grid(True, ls=':')
        else: ax0.text(0.5, 0.5, "ESF not calculated/available", **{'ha':'center', 'va':'center', 'transform':ax0.transAxes, 'color':'orange'})
        ax0.set_title(f'ESF - {title_roi_str}')

        ax1 = axes[1]
        if s_lsf is not None and lsf_final is not None:
            ax1.plot(s_lsf, lsf_final, 'r-', lw=1.5, label=f'Final LSF (Window {lsf_window_width_mm}mm)')
            ax1.set_title(f'Line Spread Function (LSF)')
            ax1.set_xlabel('Distance (s) [mm]'); ax1.set_ylabel('Derivative')
            ax1.legend(); ax1.grid(True, ls=':')
            try:
                lsf_min, lsf_max = np.min(lsf_final), np.max(lsf_final); margin = (lsf_max - lsf_min) * 0.1
                if margin < 1e-6: margin = max(abs(lsf_min), abs(lsf_max)) * 0.1 + 1e-6
                ax1.set_ylim(lsf_min - margin, lsf_max + margin); ax1.sharex(ax0); ax1.tick_params(axis='x', labelbottom=True)
            except ValueError: pass
        else: ax1.text(0.5, 0.5, "LSF not calculated/available", **{'ha':'center', 'va':'center', 'transform':ax1.transAxes, 'color':'orange'})
        ax1.set_title('LSF'); ax1.set_xlabel('Distance (s) [mm]')

        ax2 = axes[2]
        if frequencies is not None and mtf is not None:
            ax2.plot(frequencies, mtf, 'g.-', ms=4, label=f'MTF {roi_id}')
            ax2.set_title(f'Modulation Transfer Function (MTF) - up to {cutoff_freq} c/mm')
            ax2.set_xlabel('Spatial Frequency [c/mm]'); ax2.set_ylabel('MTF')
            ax2.grid(True, which='major', ls='-', lw=0.6); ax2.grid(True, which='minor', ls=':', lw=0.4); ax2.minorticks_on()
            ax2.set_xlim(left=0, right=cutoff_freq); ax2.set_ylim(bottom=-0.05, top=1.1)
            try:
                idx_mtf05 = np.where(mtf < 0.5)[0];
                if len(idx_mtf05) > 0:
                    idx_mtf05 = idx_mtf05[0]
                    freq_mtf05 = np.interp(0.5, mtf[idx_mtf05::-1], frequencies[idx_mtf05::-1]) if idx_mtf05 > 0 else frequencies[idx_mtf05]
                    ax2.axhline(0.5, color='grey', ls='--', lw=0.8, label='MTF=0.5')
                    ax2.axvline(freq_mtf05, color='grey', ls='--', lw=0.8)
                    label_x_pos = freq_mtf05 * 1.05; ha = 'left'
                    if label_x_pos > cutoff_freq * 0.8: label_x_pos = freq_mtf05 * 0.95; ha = 'right'
                    ax2.text(label_x_pos, 0.51, f'{freq_mtf05:.2f} c/mm', **{'ha':'center', 'va':'center', 'transform':ax2.transAxes, 'color':'orange'})
                else: ax2.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5 (not reached)')
            except Exception as e_mtf05: print(f"  {roi_id} - Could not mark MTF=0.5: {e_mtf05}")
            ax2.legend(loc='upper right')
        else: ax2.text(0.5, 0.5, "MTF not calculated/available", **{'ha':'center', 'va':'center', 'transform':ax2.transAxes, 'color':'orange'})
        ax2.set_title('MTF'); ax2.set_xlabel('Freq [c/mm]'); ax2.set_ylabel('MTF')

        try:
            fig.tight_layout(pad=2.0, h_pad=3.0)
            fig.suptitle(f"{title_prefix} - Individual Analysis {roi_id}", fontsize=14, y=0.99)
            plt.subplots_adjust(top=0.92)
        except Exception as e_layout: print(f"Layout/suptitle error {roi_id}: {e_layout}")


    def plot_average_mtf(self, results1, results2, common_freq_axis, mtf_avg,
                         title_prefix="", poly_fit_func=None, plot_poly_fit=True):
        """Plots the individual MTFs, the average, and optionally the polynomial fit."""
        if mtf_avg is None or common_freq_axis is None:
             print("Warning: Cannot plot average MTF (missing data).")
             return

        print("\n[Plotting Comparison and Average MTF]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        plt.figure("Comparative and Average MTF", figsize=(10, 6)); plt.clf()

        roi1_id = results1.get("roi_id", "ROI1") if results1 else "ROI1"
        if results1 and results1.get("status","Error").startswith("OK") and results1.get("frequencies") is not None:
            plt.plot(results1['frequencies'], results1['mtf'], marker='.', ms=4, ls='', 
                     label=f'MTF {roi1_id} (Data)', alpha=0.5)

        roi2_id = results2.get("roi_id", "ROI2") if results2 else "ROI2"
        if results2 and results2.get("status","Error").startswith("OK") and results2.get("frequencies") is not None:
            plt.plot(results2['frequencies'], results2['mtf'], marker='.', ms=4, ls='', 
                     label=f'MTF {roi2_id} (Data)', alpha=0.5)

        plt.plot(common_freq_axis, mtf_avg, color='black', ls='-', lw=1.5, label='Average MTF')

        if poly_fit_func is not None and plot_poly_fit:
            print("  Plotting polynomial fit...")
            mtf_poly_fit_values = poly_fit_func(common_freq_axis)
            mtf_poly_fit_values = np.clip(mtf_poly_fit_values, 0, 1)
            plt.plot(common_freq_axis, mtf_poly_fit_values, color='red', linestyle='--', linewidth=1.5,
                     label=f'Polynomial Fit (Degree {len(poly_fit_func.coeffs)-1})')

        plt.title(f'{title_prefix} - MTF Comparison and Average')
        plt.xlabel('Spatial Frequency [cycles/mm]'); plt.ylabel('MTF')
        plt.xlim(left=0, right=self.params['cutoff_freq'])
        plt.ylim(bottom=-0.05, top=1.1)
        plt.grid(True, which='major', ls='-', lw=0.6); plt.grid(True, which='minor', ls=':', lw=0.4); plt.minorticks_on()

        try:
             idx_mtf05_avg = np.where(mtf_avg < 0.5)[0]
             if len(idx_mtf05_avg) > 0:
                 idx_mtf05_avg = idx_mtf05_avg[0]
                 freq_mtf05_avg = np.interp(0.5, mtf_avg[idx_mtf05_avg::-1], common_freq_axis[idx_mtf05_avg::-1]) if idx_mtf05_avg > 0 else common_freq_axis[idx_mtf05_avg]
                 plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5')
                 plt.axvline(freq_mtf05_avg, color='grey', ls=':', lw=1.0)
                 label_x = freq_mtf05_avg + 0.1; ha = 'left'
                 if label_x > self.params['cutoff_freq'] * 0.85: label_x = freq_mtf05_avg - 0.1; ha = 'right'
                 plt.text(label_x, 0.51, f'Avg: {freq_mtf05_avg:.2f} c/mm', **{'va':'bottom', 'ha':ha, 'fontsize':9, 'color':'black'})
             else: plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5 (not reached)')
        except Exception as e_avg: print(f"Could not mark average MTF=0.5: {e_avg}")


        plt.legend(); plt.tight_layout()

    @staticmethod
    def calculate_grand_average_mtf(list_of_mtf_results, num_points=250, forced_cutoff_freq=None):
        """Calculates the grand average MTF from a list of individual MTF results.

        Args:
            list_of_mtf_results (list): A list of dictionaries, each containing 'frequencies' and 'mtf' NumPy arrays.
            num_points (int): The number of points for the common interpolated frequency axis.
            forced_cutoff_freq (float, optional): The maximum frequency for the common axis.

        Returns:
            tuple: A tuple containing common_freq_axis, mtf_grand_avg, mtf_std_dev, and interpolated_mtfs.
        """
        print(f"\n--- Calculating Grand Average MTF ---")
        valid_results = [res for res in list_of_mtf_results if
                         res and isinstance(res.get('frequencies'), np.ndarray) and
                         isinstance(res.get('mtf'), np.ndarray) and
                         len(res['frequencies']) > 1 and len(res['mtf']) > 1 and
                         len(res['frequencies']) == len(res['mtf'])]

        if len(valid_results) < 2:
            print(f"Error: At least 2 valid MTF results are required for averaging. Found {len(valid_results)}.")
            return None, None, None, None

        print(f" Averaging {len(valid_results)} valid MTF curves.")

        try:
            max_freqs = [res['frequencies'][-1] for res in valid_results]
            min_of_max_freqs = min(max_freqs)

            if forced_cutoff_freq is not None:
                 cutoff_freq = forced_cutoff_freq
                 print(f"  Using forced cutoff frequency: {cutoff_freq:.2f} c/mm")
            else:
                 max_freqs = [res['frequencies'][-1] for res in valid_results]
                 cutoff_freq = min(max_freqs)
                 print(f"  Using minimum cutoff frequency from inputs: {cutoff_freq:.2f} c/mm")

            if cutoff_freq <= 0:
                 print(f"Error: Calculated cutoff frequency ({cutoff_freq:.2f}) is invalid.")
                 return None, None, None, None

            common_freq_axis = np.linspace(0, cutoff_freq, num=num_points)
            interpolated_mtfs = []

            for i, res in enumerate(valid_results):
                try:
                    mtf_interp = np.full_like(common_freq_axis, np.nan)

                    min_freq_res = res['frequencies'][0]
                    max_freq_res = res['frequencies'][-1]
                    
                    valid_mask = (common_freq_axis >= min_freq_res) & (common_freq_axis <= max_freq_res)

                    if np.any(valid_mask):
                        mtf_interp[valid_mask] = np.interp(
                            common_freq_axis[valid_mask],
                            res['frequencies'],
                            res['mtf']
                        )
                    
                    if common_freq_axis[0] == 0:
                        mtf_interp[0] = 1.0

                    interpolated_mtfs.append(mtf_interp)
                except Exception as e_interp:
                    print(f"Warning: Could not process MTF #{i} for averaging: {e_interp}")

            if len(interpolated_mtfs) < 2:
                 print(f"Error: Not enough MTFs left after interpolation ({len(interpolated_mtfs)}).")
                 return None, None, None, None

            mtf_array = np.array(interpolated_mtfs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mtf_grand_avg = np.nanmean(mtf_array, axis=0)
                mtf_std_dev = np.nanstd(mtf_array, axis=0)

            if mtf_std_dev is not None:
                mtf_std_dev[np.isnan(mtf_std_dev)] = 0.0

            print(f" Grand average MTF calculated ({len(interpolated_mtfs)} curves used).")
            return common_freq_axis, mtf_grand_avg, mtf_std_dev, interpolated_mtfs

        except Exception as e:
            print(f"Error during grand average MTF calculation: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None


    @staticmethod
    def plot_grand_average_mtf(common_freq_axis, mtf_grand_avg, mtf_std_dev=None,
                               interpolated_mtfs=None, poly_fit_func=None,
                               title="Grand Average MTF", cutoff_freq=None,
                               plot_individuals=False, plot_std_dev=True):
        """Plots the grand average MTF.

        Args:
            common_freq_axis (np.ndarray): The common frequency axis.
            mtf_grand_avg (np.ndarray): The average MTF.
            mtf_std_dev (np.ndarray, optional): The standard deviation for shading.
            interpolated_mtfs (list, optional): A list of individual interpolated MTFs to plot.
            poly_fit_func (np.poly1d, optional): The fitted polynomial function to plot.
            title (str): The title of the plot.
            cutoff_freq (float, optional): The maximum frequency to display on the x-axis.
            plot_individuals (bool): If True, plots the individual interpolated curves.
            plot_std_dev (bool): If True, shades the area of +/- 1 standard deviation.
        """
        if common_freq_axis is None or mtf_grand_avg is None:
            print("Warning: Cannot plot grand average MTF (missing data).")
            return

        print(f"\n[Plotting Grand Average MTF: {title}]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        plt.figure("Grand Average MTF", figsize=(10, 6)); plt.clf()

        if plot_individuals and interpolated_mtfs:
            print(f"  Plotting {len(interpolated_mtfs)} individual interpolated curves...")
            for i, mtf_interp in enumerate(interpolated_mtfs):
                plt.plot(common_freq_axis, mtf_interp, color='grey', alpha=0.15, lw=0.8,
                         label='_nolegend_')

        if plot_std_dev and mtf_std_dev is not None:
            print("  Plotting +/- 1 standard deviation...")
            plt.fill_between(common_freq_axis, mtf_grand_avg - mtf_std_dev,
                             mtf_grand_avg + mtf_std_dev, color='skyblue', alpha=0.4,
                             label='±1 Standard Deviation')

        plt.plot(common_freq_axis, mtf_grand_avg, color='black', ls='-', lw=2.0,
                 label=f'Average MTF ({len(interpolated_mtfs) if interpolated_mtfs else "N/A"} curves)')

        if poly_fit_func is not None:
            print("  Plotting polynomial fit...")
            mtf_poly_fit_values = poly_fit_func(common_freq_axis)
            mtf_poly_fit_values = np.clip(mtf_poly_fit_values, 0, 1)
            plt.plot(common_freq_axis, mtf_poly_fit_values, color='red', linestyle='--', linewidth=1.5,
                     label=f'Polynomial Fit (Degree {len(poly_fit_func.coeffs)-1})')

        plt.title(title)
        plt.xlabel('Spatial Frequency [cycles/mm]'); plt.ylabel('MTF')
        plot_cutoff = cutoff_freq if cutoff_freq is not None else common_freq_axis[-1]
        plt.xlim(left=0, right=plot_cutoff)
        plt.ylim(bottom=-0.05, top=1.1)
        plt.grid(True, which='major', ls='-', lw=0.6); plt.grid(True, which='minor', ls=':', lw=0.4)
        plt.minorticks_on()

        try:
             idx_mtf05_avg = np.where(mtf_grand_avg < 0.5)[0]
             if len(idx_mtf05_avg) > 0:
                 idx_mtf05_avg = idx_mtf05_avg[0]
                 if idx_mtf05_avg > 0:
                      freq_mtf05_avg = np.interp(0.5, mtf_grand_avg[idx_mtf05_avg::-1], common_freq_axis[idx_mtf05_avg::-1])
                 else:
                      freq_mtf05_avg = common_freq_axis[idx_mtf05_avg]

                 plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5')
                 plt.axvline(freq_mtf05_avg, color='grey', ls=':', lw=1.0)
                 label_x = freq_mtf05_avg + 0.05 * plot_cutoff; ha = 'left'
                 if label_x > plot_cutoff * 0.8: label_x = freq_mtf05_avg - 0.05 * plot_cutoff; ha = 'right'
                 plt.text(label_x, 0.51, f'{freq_mtf05_avg:.2f} c/mm', **{'va':'bottom', 'ha':ha, 'fontsize':9, 'color':'black'})
             else: plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5 (not reached)')
        except Exception as e_avg: print(f"Warning: Could not mark average MTF=0.5: {e_avg}")


        plt.legend(); plt.tight_layout()
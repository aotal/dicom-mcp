# --- Archivo: mtf_analyzer.py ---

import numpy as np
import math
import os
import pandas as pd # Necesario para validar el DataFrame
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import warnings

# --- Importar la función de linealización ---
# Asegúrate de que utils.py esté accesible
try:
    from utils import linearize_preprocessed_image_from_df
except ImportError:
    print("ERROR CRÍTICO: No se pudo importar 'linearize_preprocessed_image_from_df' desde 'utils.py'.")
    # Podrías definir una función dummy o lanzar un error para evitar que continúe
    def linearize_preprocessed_image_from_df(*args, **kwargs):
        raise ImportError("Función de linealización no disponible.")

# ==============================================================================
# 1. DEFINICIONES DE LAS FUNCIONES DE CÁLCULO MTF
# (Versiones limpias que aceptan arrays y usan nombres estándar)
# ==============================================================================

import numpy as np
from scipy.stats import linregress
import math
import matplotlib.pyplot as plt # Si plot_fit=True

def estimate_angle_from_midpoints_vectorized(oriented_roi_array, plot_fit=False, verbose=True):
    """
    Estima el ángulo geométrico del borde vs vertical desde una ROI orientada (array NumPy).
    Versión vectorizada con NumPy.
    """
    if verbose: print(f"\n--- Estimando ángulo por método de puntos medios (Vectorizado) ---")
    image_data_linear = oriented_roi_array
    if image_data_linear is None or image_data_linear.size == 0:
        print("Error: Array de ROI vacío o inválido para estimar ángulo.")
        return None

    rows, cols = image_data_linear.shape
    if verbose: print(f"Analizando ROI orientada de forma: {rows}x{cols}")

    # --- Inicio de la Vectorización ---

    # 1. Calcular Mínimos, Máximos y Rango por fila (ignorando NaNs)
    # Añadir manejo de filas completamente NaN que resultarían en NaN para min/max
    with np.errstate(all='ignore'): # Suprimir warnings de filas totalmente NaN
        row_mins = np.nanmin(image_data_linear, axis=1)
        row_maxs = np.nanmax(image_data_linear, axis=1)
    row_ranges = row_maxs - row_mins

    # 2. Calcular umbral de rango y máscara de filas válidas inicial
    global_min, global_max = np.nanmin(row_mins), np.nanmax(row_maxs) # Usar min/max de filas
    min_range_threshold = (global_max - global_min) * 0.05
    if min_range_threshold < 1e-6: min_range_threshold = 1e-6

    # Máscara de filas válidas: rango suficiente Y no son completamente NaN (min != max o rango no es NaN)
    valid_row_mask = (row_ranges >= min_range_threshold) & ~np.isnan(row_ranges)
    
    # Si no hay filas válidas después del filtrado inicial
    if not np.any(valid_row_mask):
         print(f"Error: No se encontraron filas con suficiente rango dinámico.")
         return None

    # 3. Filtrar datos y calcular Mid-Values para filas válidas
    valid_rows_data = image_data_linear[valid_row_mask, :]
    valid_row_indices = np.where(valid_row_mask)[0] # Índices originales de las filas válidas
    valid_row_mins = row_mins[valid_row_mask]
    valid_row_maxs = row_maxs[valid_row_mask]
    mid_vals = (valid_row_mins + valid_row_maxs) / 2.0

    # 4. Encontrar cruces del Mid-Value
    # Restar el mid_val correspondiente a cada fila válida (Broadcasting)
    centered_data = valid_rows_data - mid_vals[:, np.newaxis]
    # Encontrar cambios de signo a lo largo de las columnas (axis=1)
    sign_changes = np.diff(np.signbit(centered_data), axis=1)
    # Identificar dónde ocurren los cambios (True donde hay cambio)
    crossing_locations = (sign_changes != 0)

    # 5. Filtrar filas que tienen EXACTAMENTE UN cruce
    rows_with_single_crossing_mask = np.sum(crossing_locations, axis=1) == 1
    
    # Si no quedan filas con un solo cruce
    if not np.any(rows_with_single_crossing_mask):
        print(f"Error: No se encontraron filas con un único cruce del punto medio.")
        return None

    # Índices (dentro del array filtrado 'valid_rows_data') de las filas con un solo cruce
    single_crossing_filtered_indices = np.where(rows_with_single_crossing_mask)[0]

    # Obtener los datos y valores necesarios SOLO para estas filas
    final_rows_data = valid_rows_data[rows_with_single_crossing_mask, :]
    final_mid_vals = mid_vals[rows_with_single_crossing_mask]
    final_original_indices = valid_row_indices[rows_with_single_crossing_mask] # Índices originales finales

    # Encontrar la columna 'j' (índice ANTES del cruce) para cada fila final
    # argmax encuentra el *primer* True en cada fila de crossing_locations[rows_with_single_crossing_mask, :]
    j_indices = np.argmax(crossing_locations[rows_with_single_crossing_mask, :], axis=1)

    # 6. Realizar Interpolación Lineal Vectorizada
    # Obtener valores en j y j+1 para todas las filas finales a la vez
    # Usamos indexación avanzada
    row_idx_for_indexing = np.arange(len(final_original_indices)) # Índices 0, 1, 2... para las filas finales
    val_j = final_rows_data[row_idx_for_indexing, j_indices]
    val_j1 = final_rows_data[row_idx_for_indexing, j_indices + 1] # Asume j+1 siempre es válido

    delta_val = val_j1 - val_j
    
    # Inicializar x_mid con NaN o un valor por defecto
    x_mid = np.full(final_original_indices.shape, np.nan)

    # Máscara para casos donde val_j1 es muy cercano a val_j (evitar división por cero)
    near_zero_delta_mask = np.abs(delta_val) < 1e-9
    x_mid[near_zero_delta_mask] = j_indices[near_zero_delta_mask] + 0.5 # Midpoint es j + 0.5

    # Calcular fracción para los casos no cercanos a cero
    non_zero_delta_mask = ~near_zero_delta_mask
    fraction = np.full(final_original_indices.shape, np.nan)
    # Calcular fracción solo donde delta_val no es cero
    fraction[non_zero_delta_mask] = (final_mid_vals[non_zero_delta_mask] - val_j[non_zero_delta_mask]) / delta_val[non_zero_delta_mask]

    # Máscara para fracciones válidas (entre 0 y 1)
    valid_fraction_mask = (fraction >= 0.0) & (fraction <= 1.0) & non_zero_delta_mask

    # Calcular x_mid para fracciones válidas
    x_mid[valid_fraction_mask] = j_indices[valid_fraction_mask] + fraction[valid_fraction_mask]

    # 7. Filtrar Puntos Medios Válidos Finales
    # Combinar máscaras: necesitan tener un delta cercano a cero O una fracción válida
    # Y el x_mid resultante debe estar dentro de los límites de las columnas
    final_valid_points_mask = (near_zero_delta_mask | valid_fraction_mask) & \
                              (x_mid >= 0) & (x_mid < cols)

    y_coords = final_original_indices[final_valid_points_mask]
    x_coords = x_mid[final_valid_points_mask]

    # --- Fin de la Vectorización ---

    if len(y_coords) < 5: # Usar y_coords o x_coords, tienen la misma longitud
        print(f"Error: No se encontraron suficientes puntos medios válidos ({len(y_coords)}) después de la interpolación.")
        return None
    if verbose: print(f"Se encontraron {len(y_coords)} puntos medios válidos.")

    # Realizar la regresión lineal (igual que antes)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_coords, x_coords)
        r_squared = r_value**2
        if verbose: print(f"Ajuste lineal: x_mid = {slope:.5f}*y + {intercept:.3f}, R²={r_squared:.4f} (p={p_value:.3g})")
        if r_squared < 0.90: print(f"Advertencia: Ajuste lineal poco fiable (R²={r_squared:.4f}).")

        # Calcular el ángulo (igual que antes)
        estimated_angle_deg_vertical_ref = math.degrees(math.atan(slope)) if abs(slope) >= 1e-9 else 0.0
        if verbose: print(f"Ángulo geométrico (ref. vertical): {estimated_angle_deg_vertical_ref:.3f} grados")

        if plot_fit:
            plt.figure("Midpoint Fit (Vectorized)", figsize=(8, 6)); plt.clf()
            plt.scatter(x_coords, y_coords, label=f'Puntos Medios ({len(x_coords)})', alpha=0.7, s=10)
            y_fit = np.array([y_coords.min(), y_coords.max()]); x_fit = slope * y_fit + intercept
            plt.plot(x_fit, y_fit, color='red', label=f'Ajuste Lineal\nÁngulo (vs V) = {estimated_angle_deg_vertical_ref:.2f}°\nR² = {r_squared:.3f}')
            plt.title('Ajuste Puntos Medios (ROI Orientada - Vectorizado)'); plt.xlabel('Col (x_mid)'); plt.ylabel('Fila (y)')
            plt.gca().invert_yaxis(); plt.legend(); plt.grid(True, ls=':'); plt.axis('equal'); plt.tight_layout()
            # Consider adding plt.show() if running outside an interactive environment like Jupyter

        return estimated_angle_deg_vertical_ref
    except Exception as e:
        print(f"Error ajuste lineal: {e}")
        return None

# --- Ejemplo de uso (necesitarías crear un array 'oriented_roi_array' de ejemplo) ---
# rows, cols = 100, 50
# data = np.zeros((rows, cols))
# # Crear un borde diagonal simple con algo de ruido
# for r in range(rows):
#     center_col = int(cols * 0.3 + r * 0.4) # Pendiente
#     data[r, max(0, center_col-5):min(cols, center_col+5)] = 100
# data += np.random.rand(rows, cols) * 10 # Añadir ruido
# data = np.gaussian_filter(data, sigma=1.5) # Suavizar

# angle = estimate_angle_from_midpoints_vectorized(data, plot_fit=True, verbose=True)
# if angle is not None:
#     print(f"\nÁngulo estimado final: {angle:.3f} grados")
# else:
#     print("\nNo se pudo estimar el ángulo.")
# if plt.fignum_exists("Midpoint Fit (Vectorized)"): # Solo muestra si se creó la figura
#      plt.show()

def estimate_angle_from_midpoints(oriented_roi_array, plot_fit=False, verbose=True):
    """
    Estima el ángulo geométrico del borde vs vertical desde una ROI orientada (array NumPy).
    """
    if verbose: print(f"\n--- Estimando ángulo por método de puntos medios ---")
    image_data_linear = oriented_roi_array
    if image_data_linear is None or image_data_linear.size == 0:
         print("Error: Array de ROI vacío o inválido para estimar ángulo.")
         return None
    rows, cols = image_data_linear.shape
    if verbose: print(f"Analizando ROI orientada de forma: {rows}x{cols}")

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
        print(f"Error: No se encontraron suficientes puntos medios válidos ({len(row_indices)}).")
        return None
    if verbose: print(f"Se encontraron {len(row_indices)} puntos medios válidos.")
    y_coords, x_coords = np.array(row_indices), np.array(midpoint_cols)

    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_coords, x_coords); r_squared = r_value**2
        if verbose: print(f"Ajuste lineal: x_mid = {slope:.5f}*y + {intercept:.3f}, R²={r_squared:.4f} (p={p_value:.3g})")
        if r_squared < 0.90: print(f"Advertencia: Ajuste lineal poco fiable (R²={r_squared:.4f}).")
        estimated_angle_deg_vertical_ref = math.degrees(math.atan(slope)) if abs(slope) >= 1e-9 else 0.0
        if verbose: print(f"Ángulo geométrico (ref. vertical): {estimated_angle_deg_vertical_ref:.3f} grados")

        if plot_fit:
            plt.figure("Midpoint Fit", figsize=(8, 6)); plt.clf()
            plt.scatter(x_coords, y_coords, label=f'Puntos Medios ({len(x_coords)})', alpha=0.7, s=10)
            y_fit = np.array([y_coords.min(), y_coords.max()]); x_fit = slope * y_fit + intercept
            plt.plot(x_fit, y_fit, color='red', label=f'Ajuste Lineal\nÁngulo (vs V) = {estimated_angle_deg_vertical_ref:.2f}°\nR² = {r_squared:.3f}')
            plt.title('Ajuste Puntos Medios (ROI Orientada)'); plt.xlabel('Col (x_mid)'); plt.ylabel('Fila (y)')
            plt.gca().invert_yaxis(); plt.legend(); plt.grid(True, ls=':'); plt.axis('equal'); plt.tight_layout()

        return estimated_angle_deg_vertical_ref
    except Exception as e: print(f"Error ajuste lineal: {e}"); return None

# ------------------------------------------------------------------------------

def calculate_esf_from_roi(oriented_roi_array, angle_deg, pixel_size_mm,
                           sub_pixel_factor=0.1, smoothing_window_bins=17,
                           smoothing_polyorder=4, verbose=True):
    """Calcula ESF desde ROI orientada (array)."""
    image_data_linear = oriented_roi_array
    if image_data_linear is None or image_data_linear.size == 0: return None, None, None, None
    rows, cols = image_data_linear.shape
    if pixel_size_mm is None or pixel_size_mm <= 0: print(f"Error: pixel_size_mm inválido ({pixel_size_mm})"); return None, None, None, None
    if verbose: print(f"\nCalculando ESF para ángulo={angle_deg:.3f} deg...\n Usando ROI de forma: {rows}x{cols}\n Parámetros: Píxel={pixel_size_mm:.4f}mm, SubBin={sub_pixel_factor}")
    angle_rad = math.radians(angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
    delta_s = sub_pixel_factor * pixel_size_mm; center_i = (rows - 1) / 2.0; center_j = (cols - 1) / 2.0
    if verbose: print(f"  Ancho Bin (Δs)={delta_s:.5f} mm\n Rango datos ROI: [{image_data_linear.min():.2f}, {image_data_linear.max():.2f}]")

    binned_values = {}
    if verbose: print(" Iniciando reproyección y binning ESF...")
    for i in range(rows):
        for j in range(cols):
            s = pixel_size_mm * ((j - center_j) * cos_a - (i - center_i) * sin_a)
            bin_index = math.floor(s / delta_s)
            if bin_index not in binned_values: binned_values[bin_index] = []
            binned_values[bin_index].append(image_data_linear[i, j])
    if verbose: print(f" Binning ESF completado. Bins iniciales: {len(binned_values)}")
    if not binned_values: print(f"¡Error! No se crearon bins ESF."); return None, None, None, None

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
            if verbose: print(f" Advertencia: {len(nan_indices)} bins vacíos. Interpolando...")
            is_nan = np.isnan(esf_raw); x_all = np.arange(len(esf_raw))
            if not np.all(is_nan): esf_raw[is_nan] = np.interp(x_all[is_nan], x_all[~is_nan], esf_raw[~is_nan])
            if np.isnan(esf_raw).any():
                if verbose: print("  No se pudieron interpolar todos (extremos?). Recortando.")
                valid_mask = ~np.isnan(esf_raw); s_coords = s_coords[valid_mask]; esf_raw = esf_raw[valid_mask]
        if verbose: print(f" ESF Raw calculada. Puntos: {len(esf_raw)}. Rango s: [{s_coords.min():.3f}, {s_coords.max():.3f}] mm\n Rango ESF Raw: [{np.nanmin(esf_raw):.2f}, {np.nanmax(esf_raw):.2f}]")
    except Exception as e: print(f"Error calculando ESF raw: {e}"); return None, None, None, None

    if len(esf_raw) < max(2, smoothing_polyorder + 1):
         print(f"¡Error! ESF Raw < {max(2, smoothing_polyorder + 1)} puntos ({len(esf_raw)}).");
         return s_coords, None, esf_raw, delta_s # Devolver raw aunque no se pueda suavizar

    esf_smoothed = None
    effective_smoothing_window = smoothing_window_bins + (smoothing_window_bins % 2 == 0)
    if len(esf_raw) < effective_smoothing_window: effective_smoothing_window = len(esf_raw) - (len(esf_raw) % 2 == 0)
    if effective_smoothing_window < 3: effective_smoothing_window = 0

    if effective_smoothing_window >= 3:
        actual_polyorder = min(smoothing_polyorder, effective_smoothing_window - 1); actual_polyorder = max(1, actual_polyorder)
        if verbose: print(f" Aplicando Savitzky-Golay (w={effective_smoothing_window}, p={actual_polyorder})...")
        try: esf_smoothed = savgol_filter(esf_raw.astype(np.float64), effective_smoothing_window, actual_polyorder)
        except Exception as e: print(f" Error suavizado Savitzky-Golay: {e}. Devolviendo ESF raw."); esf_smoothed = esf_raw.copy()
    else:
        if verbose: print(" Omitiendo suavizado (pocos puntos/ventana inválida)."); esf_smoothed = esf_raw.copy()

    if esf_smoothed is None: esf_smoothed = esf_raw.copy()
    if verbose and esf_smoothed is not esf_raw: print(f" ESF Suavizada calculada. Rango: [{esf_smoothed.min():.2f}, {esf_smoothed.max():.2f}]")
    return s_coords, esf_smoothed, esf_raw, delta_s

# ------------------------------------------------------------------------------

def calculate_lsf( s_coords, esf_smoothed, delta_s,
                  baseline_tail_threshold_mm=7.0, window_width_mm=20.0, verbose=True):
    """ Calcula la LSF desde ESF, buscando pico absoluto e invirtiendo si es negativo."""
    if esf_smoothed is None or s_coords is None or len(esf_smoothed) < 2 or delta_s is None: return None, None
    if verbose: print(f"\n--- Calculando LSF --- \n Espaciado (Δs) para diferenciación: {delta_s:.5f} mm")
    try: lsf_raw = np.gradient(esf_smoothed, delta_s)
    except Exception as e: print(f"Error en np.gradient para LSF: {e}"); return None, None
    if verbose: print(f" LSF Raw (dif.) Puntos: {len(lsf_raw)}. Rango: [{lsf_raw.min():.2f}, {lsf_raw.max():.2f}]")

    lsf_baseline_corrected = lsf_raw.copy(); s_windowed = None; lsf_windowed = None
    try:
        peak_index = np.argmax(np.abs(lsf_raw))
        s_peak = s_coords[peak_index]; lsf_peak_value = lsf_raw[peak_index]
        if verbose: print(f" Pico LSF Raw (máx abs) en s={s_peak:.3f}mm (idx {peak_index}, val={lsf_peak_value:.2f})")
        tail_indices = np.where(np.abs(s_coords - s_peak) > baseline_tail_threshold_mm)[0]
        if len(tail_indices) > 5:
            s_tails = s_coords[tail_indices]; lsf_tails = lsf_raw[tail_indices]; coeffs = np.polyfit(s_tails, lsf_tails, 1)
            baseline = np.polyval(coeffs, s_coords); lsf_baseline_corrected = lsf_raw - baseline
            if verbose: print(f" Ajuste línea base (cola>{baseline_tail_threshold_mm}mm): Pend={coeffs[0]:.4f}, Int={coeffs[1]:.4f}\n Línea base restada.")
        else:
            if verbose: print(f" Advertencia: Pocos puntos ({len(tail_indices)}) en colas. Omitiendo sustracción.")
    except Exception as e: print(f" Error sustracción línea base: {e}. Usando LSF Raw."); lsf_baseline_corrected = lsf_raw.copy()
    if verbose: print(f" LSF con línea base corregida. Rango: [{lsf_baseline_corrected.min():.2f}, {lsf_baseline_corrected.max():.2f}]")

    try:
        peak_index_corrected = np.argmax(np.abs(lsf_baseline_corrected))
        s_peak_corrected = s_coords[peak_index_corrected]; lsf_peak_corrected_value = lsf_baseline_corrected[peak_index_corrected]
        if verbose: print(f" Pico LSF corregida (máx abs) en s={s_peak_corrected:.3f}mm (idx {peak_index_corrected}, val={lsf_peak_corrected_value:.2f})")
        window_half_width = window_width_mm/2.0; window_indices = np.where(np.abs(s_coords - s_peak_corrected) <= window_half_width)[0]
        if len(window_indices) < 2: print(f" Error: Pocos puntos LSF ({len(window_indices)}) en ventana {window_width_mm}mm."); return None, None
        s_windowed = s_coords[window_indices]; lsf_to_window = lsf_baseline_corrected[window_indices]
        hanning_win = np.hanning(len(lsf_to_window)); lsf_windowed = lsf_to_window * hanning_win
        if verbose:
            w_width_act = s_windowed[-1] - s_windowed[0] if len(s_windowed) > 1 else 0
            print(f" Ventana Hanning aplicada (ancho nom={window_width_mm}mm, real~{w_width_act:.1f}mm, {len(lsf_windowed)} pts).")
            print(f" LSF final (ventaneada). Rango: [{lsf_windowed.min():.2f}, {lsf_windowed.max():.2f}]")
    except Exception as e: print(f" Error ventaneo LSF: {e}."); return None, None

    if lsf_windowed is None: return None, None
    peak_val_windowed = lsf_windowed[np.argmax(np.abs(lsf_windowed))] if len(lsf_windowed) > 0 else 0
    if peak_val_windowed < -1e-9:
        if verbose: print(" Advertencia: Pico LSF ventaneada negativo. Invirtiendo LSF para MTF.")
        lsf_windowed = -lsf_windowed
    return s_windowed, lsf_windowed

# ------------------------------------------------------------------------------

def calculate_mtf(s_lsf, lsf_final, delta_s, verbose=True, cutoff_freq=3.7):
    """Calcula MTF desde LSF final."""
    if lsf_final is None or s_lsf is None or len(lsf_final) < 2 or delta_s is None or delta_s <= 0: print("Error: LSF inválida o delta_s para MTF."); return None, None
    if verbose: print("\n--- Calculando MTF ---")
    try:
        area_lsf = trapezoid(lsf_final, dx=delta_s)
        if abs(area_lsf) < 1e-9: print("Error: Área LSF cercana a cero."); return None, None
        lsf_normalized = lsf_final / area_lsf
        if verbose: print(f" Área LSF original: {area_lsf:.4f}"); area_check=trapezoid(lsf_normalized, dx=delta_s); print(f" LSF normalizada. Área check: {area_check:.4f}")
    except Exception as e: print(f"Error normalización LSF: {e}"); return None, None
    try:
        N = len(lsf_normalized); otf = fft(lsf_normalized)
        if verbose: print(f" OTF calculada (FFT, N={N} puntos).")
    except Exception as e: print(f"Error cálculo FFT: {e}"); return None, None
    try:
        frequencies_all = fftfreq(N, d=delta_s); nyquist_freq = 0.5 / delta_s
        if verbose: print(f" Eje frecuencias generado (d={delta_s:.5f}mm). Nyquist={nyquist_freq:.2f} c/mm")
    except Exception as e: print(f"Error generando eje frecuencias: {e}"); return None, None
    try:
        mtf_all = np.abs(otf); valid_freq_mask = (frequencies_all >= -1e-9) & (frequencies_all <= cutoff_freq + 1e-9)
        frequencies_filtered = frequencies_all[valid_freq_mask]; mtf_filtered = mtf_all[valid_freq_mask]
        sort_indices = np.argsort(frequencies_filtered); frequencies = frequencies_filtered[sort_indices]; mtf = mtf_filtered[sort_indices]
        if len(mtf) > 0:
            mtf0 = mtf[0]
            if abs(mtf0) > 1e-9: # Renormalizar si mtf(0) no es 1
                if verbose and not np.isclose(mtf0, 1.0, atol=1e-3): print(f" Normalizando MTF(0) (era {mtf0:.4f}) a 1.0.")
                mtf = mtf / mtf0
            elif verbose: print(" Advertencia: MTF(0) cercano a cero, fijando a 1."); mtf[0] = 1.0
            mtf = np.clip(mtf, 0.0, 1.0)
        else: print(f"Error: No hay frecuencias válidas en [0, {cutoff_freq:.2f}] c/mm."); return None, None
        if verbose: print(f" MTF calculada y filtrada [0, {cutoff_freq:.2f}] c/mm ({len(frequencies)} puntos).")
    except Exception as e: print(f"Error calculando/filtrando MTF: {e}"); return None, None
    return frequencies, mtf


# ==============================================================================
# 2. DEFINICIÓN DE LA CLASE MtfAnalyzer
# ==============================================================================

class MtfAnalyzer:
    """
    Clase para realizar el análisis MTF completo desde una ROI RAW,
    incluyendo la linealización interna.
    """
    def __init__(self, sub_pixel_factor=0.1,
                 smoothing_window_bins=17,
                 smoothing_polyorder=4,
                 baseline_tail_threshold_mm=7.0,
                 window_width_mm=20.0,
                 cutoff_freq=3.7):
        """
        Inicializa con los parámetros del análisis.
        Los datos de calibración ya no son necesarios aquí, ya que la linealización
        se realiza antes de llamar a esta clase.

        Args:
            sub_pixel_factor (float): Factor de submuestreo para ESF.
            smoothing_window_bins (int): Ventana para suavizado Savitzky-Golay ESF.
            smoothing_polyorder (int): Orden polinomio para suavizado ESF.
            baseline_tail_threshold_mm (float): Umbral para definir colas LSF.
            window_width_mm (float): Ancho ventana Hanning para LSF.
            cutoff_freq (float): Frecuencia de corte para MTF.
        """
        # Almacenar otros parámetros de análisis MTF
        self.params = {
            'sub_pixel_factor': sub_pixel_factor,
            'smoothing_window_bins': smoothing_window_bins,
            'smoothing_polyorder': smoothing_polyorder,
            'baseline_tail_threshold_mm': baseline_tail_threshold_mm,
            'window_width_mm': window_width_mm,
            'cutoff_freq': cutoff_freq,
        }
        # Guardar atributos individuales (útil)
        self.sub_pixel_factor = sub_pixel_factor
        self.smoothing_window_bins = smoothing_window_bins
        self.smoothing_polyorder = smoothing_polyorder
        self.baseline_tail_threshold_mm = baseline_tail_threshold_mm
        self.window_width_mm = window_width_mm
        self.cutoff_freq = cutoff_freq

        print(f"DEBUG: MtfAnalyzer inicializado con parámetros de análisis.")


    def analyze_roi(self, linearized_roi_array: np.ndarray, pixel_spacing: float,
                    roi_id="ROI", verbose=True, plot_angle_fit=False):
        """
        Realiza el análisis ESF/LSF/MTF completo para una única ROI YA LINEALIZADA.

        Args:
            linearized_roi_array (np.ndarray): El array NumPy de la ROI, ya linealizada.
            pixel_spacing (float): Tamaño del píxel en mm.
            roi_id (str): Identificador para logging y gráficos.
            verbose (bool): Si imprimir mensajes detallados.
            plot_angle_fit (bool): Si graficar el ajuste del ángulo.

        Returns:
            dict: Un diccionario con los resultados del análisis.
        """
        print(f"\n=== Analizando {roi_id} ===")
        if linearized_roi_array is None or linearized_roi_array.size == 0:
            print(f"Error: Array para {roi_id} está vacío.")
            return {"status": "Error - ROI de entrada vacía", "roi_id": roi_id}

        if pixel_spacing is None or pixel_spacing <= 0:
            print(f"Error: Pixel spacing inválido ({pixel_spacing}) para {roi_id}.")
            return {"status": "Error - Pixel Spacing inválido", "roi_id": roi_id}

        original_shape = linearized_roi_array.shape
        print(f" {roi_id} - Forma de entrada: {original_shape}")

        # --- 1. Orientar ROI (ya está linealizada) ---
        oriented_roi = linearized_roi_array
        rows_orig, cols_orig = oriented_roi.shape
        rotated_flag = False
        if rows_orig > cols_orig:
            if verbose: print(f"  [Paso 1] {roi_id} - Rotando 90 grados...")
            oriented_roi = np.rot90(oriented_roi)
            rotated_flag = True
            if verbose: print(f"     Forma orientada: {oriented_roi.shape}")
        else:
            if verbose: print(f"  [Paso 1] {roi_id} - No requiere rotación.")

        # --- 2. Estimar Ángulo ---
        if verbose: print(f"  [Paso 2] Estimando ángulo (ROI orientada)...")
        angle_deg = estimate_angle_from_midpoints_vectorized(
            oriented_roi, plot_fit=plot_angle_fit, verbose=verbose
        )
        if angle_deg is None:
            print(f"Error: Falló estimación de ángulo para {roi_id}.")
            return {"status": "Error - Estimación Ángulo", "roi_id": roi_id, "oriented_shape": oriented_roi.shape, "pixel_spacing": pixel_spacing}

        # --- 3. Calcular ESF ---
        if verbose: print(f"  [Paso 3] Calculando ESF...")
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
            print(f"Error: Falló cálculo ESF para {roi_id}.")
            return {"status": "Error - Cálculo ESF", "roi_id": roi_id, "angle_deg": angle_deg}

        # --- 4. Calcular LSF ---
        if verbose: print(f"  [Paso 4] Calculando LSF...")
        s_lsf, lsf_final = calculate_lsf(
            s_coords=s_esf,
            esf_smoothed=esf_smooth,
            delta_s=delta_s,
            baseline_tail_threshold_mm=self.params['baseline_tail_threshold_mm'],
            window_width_mm=self.params['window_width_mm'],
            verbose=verbose
        )
        if lsf_final is None:
            print(f"Error: Falló cálculo LSF para {roi_id}.")
            return {"status": "Error - Cálculo LSF", "roi_id": roi_id, "angle_deg": angle_deg, "s_esf": s_esf, "esf_smooth": esf_smooth, "esf_r": esf_r}

        # --- 5. Calcular MTF ---
        if verbose: print(f"  [Paso 5] Calculando MTF...")
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
            print(f"  Advertencia: Omitiendo cálculo MTF debido a LSF insuficiente (puntos={len(s_lsf)}) o delta_s inválido.")

        if mtf is None:
            print(f"Advertencia: Falló cálculo MTF para {roi_id} (o LSF insuficiente).")
            final_status = "Warning - MTF Failed"
        else:
            final_status = "OK"

        print(f"=== Análisis {roi_id} Completado ({final_status}) ===")
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
        """Calcula la MTF promedio desde dos resultados de análisis."""
        print("\n--- Calculando MTF Promedio ---")
        mtf1 = results1.get("mtf") if results1 else None
        mtf2 = results2.get("mtf") if results2 else None
        freqs1 = results1.get("frequencies") if results1 else None
        freqs2 = results2.get("frequencies") if results2 else None

        if mtf1 is None or mtf2 is None or len(mtf1)<2 or len(mtf2)<2:
            print("Error: No se pueden promediar MTFs (faltan datos o puntos insuficientes).")
            return None, None

        try:
            max_freq = min(freqs1[-1], freqs2[-1], self.params['cutoff_freq']) # Usar clave correcta
            common_freq_axis = np.linspace(0, max_freq, num=num_points)
            mtf1_interp = np.interp(common_freq_axis, freqs1, mtf1, left=1.0, right=0.0)
            mtf2_interp = np.interp(common_freq_axis, freqs2, mtf2, left=1.0, right=0.0)
            mtf_avg = (mtf1_interp + mtf2_interp) / 2.0
            print(" MTF promedio calculada.")
            return common_freq_axis, mtf_avg
        except Exception as e:
            print(f"Error durante interpolación/promedio MTF: {e}")
            return None, None
        
    def fit_average_mtf_polynomial(self, common_freq_axis, mtf_avg, degree=4):
        """
        Ajusta la curva MTF promedio a un polinomio, calcula R² y RMSE.
        Maneja valores NaN en la entrada.
        """
        print(f"\n--- Ajustando MTF Promedio a Polinomio Grado {degree} ---")
        if common_freq_axis is None or mtf_avg is None:
            print("Error: Faltan datos de frecuencia o MTF promedio para el ajuste.")
            return None, None, None

        valid_mask = ~np.isnan(mtf_avg)
        freq_axis_valid = common_freq_axis[valid_mask]
        mtf_avg_valid = mtf_avg[valid_mask]

        if len(freq_axis_valid) < degree + 1:
            print(f"Error: No hay suficientes puntos válidos ({len(freq_axis_valid)}) para ajustar un polinomio de grado {degree}.")
            return None, None, None

        try:
            coeffs = np.polyfit(freq_axis_valid, mtf_avg_valid, degree)
            poly1d_func = np.poly1d(coeffs)

            # Calcular bondad de ajuste (R² y RMSE)
            mtf_predicted = poly1d_func(freq_axis_valid)
            ss_res = np.sum((mtf_avg_valid - mtf_predicted) ** 2)
            ss_tot = np.sum((mtf_avg_valid - np.mean(mtf_avg_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean((mtf_avg_valid - mtf_predicted) ** 2))
            
            fit_stats = {'r_squared': r_squared, 'rmse': rmse}

            if self.params.get('verbose', True):
                 print(f" Coeficientes del polinomio (grado {degree}): {coeffs}")
                 print(f" Bondad de Ajuste: R² = {r_squared:.5f}, RMSE = {rmse:.5f}")

            return coeffs, poly1d_func, fit_stats

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Error durante el ajuste polinómico: {e}")
            return None, None, None
        except Exception as e_gen:
             print(f"Error inesperado durante el ajuste polinómico: {e_gen}")
             return None, None, None

    def plot_single_roi_analysis(self, results, title_prefix=""):
        """Genera el gráfico de 3 paneles ESF/LSF/MTF para una ROI."""
        if not results or results.get("status", "").startswith("Error"):
            print(f"DEBUG: Omitiendo plot para {results.get('roi_id','ROI desconocida')} debido a estado: {results.get('status')}")
            return

        # --- Extracción segura de variables ---
        roi_id = results.get("roi_id", "N/A")
        angle_deg = results.get("angle_deg")
        s_esf = results.get("s_esf"); esf_r = results.get("esf_r"); esf_smooth = results.get("esf_smooth")
        s_lsf = results.get("s_lsf"); lsf_final = results.get("lsf_final")
        frequencies = results.get("frequencies"); mtf = results.get("mtf")
        orig_shape = results.get("original_shape", "N/A"); rotated_flag = results.get("rotated_flag", False)
        # Parámetros de ploteo desde self.params
        cutoff_freq = self.params['cutoff_freq']
        ventana_suavizado_esf = self.params['smoothing_window_bins']
        polyorder_suavizado_esf = self.params['smoothing_polyorder']
        lsf_window_width_mm = self.params['window_width_mm']

        print(f"\n[Graficando Resultados Individuales para {roi_id}]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
        try: fig.canvas.manager.set_window_title(f"Análisis {roi_id} - {title_prefix}")
        except AttributeError: pass

        title_roi_str = f"ROI #{roi_id} (Orig: {orig_shape})"
        if rotated_flag: title_roi_str += " [Rotada]"
        title_angle_str = f"Ángulo={angle_deg:.2f}°" if angle_deg is not None else "Ángulo?"

        # --- Gráfico ESF ---
        ax0 = axes[0]
        if s_esf is not None and esf_smooth is not None and esf_r is not None:
            ax0.plot(s_esf, esf_r, 'c-', alpha=0.4, lw=1, label='ESF Raw')
            ax0.plot(s_esf, esf_smooth, 'b-', lw=1.5, label=f'ESF Suavizada (w={ventana_suavizado_esf}, p={polyorder_suavizado_esf})')
            ax0.set_title(f'ESF - {title_roi_str} - {title_angle_str}')
            ax0.set_ylabel('Intensidad Lineal'); ax0.legend(); ax0.grid(True, ls=':')
        else: ax0.text(0.5, 0.5, "ESF no calculada/disponible", **{'ha':'center', 'va':'center', 'transform':ax0.transAxes, 'color':'orange'})
        ax0.set_title(f'ESF - {title_roi_str}')

        # --- Gráfico LSF ---
        ax1 = axes[1]
        if s_lsf is not None and lsf_final is not None:
            ax1.plot(s_lsf, lsf_final, 'r-', lw=1.5, label=f'LSF Final (Ventana {lsf_window_width_mm}mm)')
            ax1.set_title(f'Line Spread Function (LSF)')
            ax1.set_xlabel('Distancia (s) [mm]'); ax1.set_ylabel('Derivada')
            ax1.legend(); ax1.grid(True, ls=':')
            try: # Ajustar límites Y
                lsf_min, lsf_max = np.min(lsf_final), np.max(lsf_final); margin = (lsf_max - lsf_min) * 0.1
                if margin < 1e-6: margin = max(abs(lsf_min), abs(lsf_max)) * 0.1 + 1e-6
                ax1.set_ylim(lsf_min - margin, lsf_max + margin); ax1.sharex(ax0); ax1.tick_params(axis='x', labelbottom=True)
            except ValueError: pass # Ignorar si no hay datos para min/max
        else: ax1.text(0.5, 0.5, "LSF no calculada/disponible", **{'ha':'center', 'va':'center', 'transform':ax1.transAxes, 'color':'orange'})
        ax1.set_title('LSF'); ax1.set_xlabel('Distancia (s) [mm]')

        # --- Gráfico MTF ---
        ax2 = axes[2]
        if frequencies is not None and mtf is not None:
            ax2.plot(frequencies, mtf, 'g.-', ms=4, label=f'MTF {roi_id}')
            ax2.set_title(f'Modulation Transfer Function (MTF) - hasta {cutoff_freq} c/mm')
            ax2.set_xlabel('Frecuencia Espacial [c/mm]'); ax2.set_ylabel('MTF')
            ax2.grid(True, which='major', ls='-', lw=0.6); ax2.grid(True, which='minor', ls=':', lw=0.4); ax2.minorticks_on()
            ax2.set_xlim(left=0, right=cutoff_freq); ax2.set_ylim(bottom=-0.05, top=1.1)
            try: # Marcar MTF 0.5
                idx_mtf05 = np.where(mtf < 0.5)[0];
                if len(idx_mtf05) > 0:
                    idx_mtf05 = idx_mtf05[0]
                    freq_mtf05 = np.interp(0.5, mtf[idx_mtf05::-1], frequencies[idx_mtf05::-1]) if idx_mtf05 > 0 else frequencies[idx_mtf05]
                    ax2.axhline(0.5, color='grey', ls='--', lw=0.8, label='MTF=0.5')
                    ax2.axvline(freq_mtf05, color='grey', ls='--', lw=0.8)
                    label_x_pos = freq_mtf05 * 1.05; ha = 'left'
                    if label_x_pos > cutoff_freq * 0.8: label_x_pos = freq_mtf05 * 0.95; ha = 'right'
                    ax2.text(label_x_pos, 0.51, f'{freq_mtf05:.2f} c/mm', **{'va':'bottom', 'ha':ha, 'fontsize':9})
                else: ax2.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5 (no alcanzado)')
            except Exception as e_mtf05: print(f"  {roi_id} - No se pudo marcar MTF=0.5: {e_mtf05}")
            ax2.legend(loc='upper right')
        else: ax2.text(0.5, 0.5, "MTF no calculada/disponible", **{'ha':'center', 'va':'center', 'transform':ax2.transAxes, 'color':'orange'})
        ax2.set_title('MTF'); ax2.set_xlabel('Freq [c/mm]'); ax2.set_ylabel('MTF')

        try:
            fig.tight_layout(pad=2.0, h_pad=3.0)
            fig.suptitle(f"{title_prefix} - Análisis Individual {roi_id}", fontsize=14, y=0.99)
            plt.subplots_adjust(top=0.92)
        except Exception as e_layout: print(f"Error layout/suptitle {roi_id}: {e_layout}")


    def plot_average_mtf(self, results1, results2, common_freq_axis, mtf_avg,
                         title_prefix="", poly_fit_func=None, plot_poly_fit=True): # Nuevos args
        """Grafica las MTFs individuales, la promedio y opcionalmente el ajuste polinómico."""
        if mtf_avg is None or common_freq_axis is None:
             print("Advertencia: No se puede graficar MTF promedio (datos ausentes).")
             return

        print("\n[Graficando Comparación y Promedio MTF]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        plt.figure("MTF Comparativa y Promedio", figsize=(10, 6)); plt.clf()

        # Plot MTF ROI 1 (si existe y OK)
        roi1_id = results1.get("roi_id", "ROI1") if results1 else "ROI1"
        if results1 and results1.get("status","Error").startswith("OK") and results1.get("frequencies") is not None:
            plt.plot(results1['frequencies'], results1['mtf'], marker='.', ms=4, ls='', # Sin línea, solo puntos
                     label=f'MTF {roi1_id} (Datos)', alpha=0.5) # Etiqueta cambiada

        # Plot MTF ROI 2 (si existe y OK)
        roi2_id = results2.get("roi_id", "ROI2") if results2 else "ROI2"
        if results2 and results2.get("status","Error").startswith("OK") and results2.get("frequencies") is not None:
            plt.plot(results2['frequencies'], results2['mtf'], marker='.', ms=4, ls='', # Sin línea, solo puntos
                     label=f'MTF {roi2_id} (Datos)', alpha=0.5) # Etiqueta cambiada

        # Plot Promedio
        plt.plot(common_freq_axis, mtf_avg, color='black', ls='-', lw=1.5, label='MTF Promedio')

        # --- Plot AJUSTE POLINÓMICO (si existe y se solicita) ---
        if poly_fit_func is not None and plot_poly_fit:
            print("  Graficando ajuste polinómico...")
            # Evaluar el polinomio en el eje común
            mtf_poly_fit_values = poly_fit_func(common_freq_axis)
            # Recortar el ajuste para que no sea > 1 o < 0 (visualmente)
            mtf_poly_fit_values = np.clip(mtf_poly_fit_values, 0, 1)
            plt.plot(common_freq_axis, mtf_poly_fit_values, color='red', linestyle='--', linewidth=1.5,
                     label=f'Ajuste Polinomio (Grado {len(poly_fit_func.coeffs)-1})')
        # -------------------------------------------------------

        plt.title(f'{title_prefix} - Comparación de MTF y Promedio')
        plt.xlabel('Frecuencia Espacial [ciclos/mm]'); plt.ylabel('MTF')
        plt.xlim(left=0, right=self.params['cutoff_freq'])
        plt.ylim(bottom=-0.05, top=1.1)
        plt.grid(True, which='major', ls='-', lw=0.6); plt.grid(True, which='minor', ls=':', lw=0.4); plt.minorticks_on()

        # ... (Marcar MTF 0.5 promedio como antes) ...
        try: # Marcar MTF=0.5 promedio
             idx_mtf05_avg = np.where(mtf_avg < 0.5)[0]
             if len(idx_mtf05_avg) > 0:
                 idx_mtf05_avg = idx_mtf05_avg[0]
                 freq_mtf05_avg = np.interp(0.5, mtf_avg[idx_mtf05_avg::-1], common_freq_axis[idx_mtf05_avg::-1]) if idx_mtf05_avg > 0 else common_freq_axis[idx_mtf05_avg]
                 plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5')
                 plt.axvline(freq_mtf05_avg, color='grey', ls=':', lw=1.0)
                 label_x = freq_mtf05_avg + 0.1; ha = 'left'
                 if label_x > self.params['cutoff_freq'] * 0.85: label_x = freq_mtf05_avg - 0.1; ha = 'right'
                 plt.text(label_x, 0.51, f'Avg: {freq_mtf05_avg:.2f} c/mm', **{'va':'bottom', 'ha':ha, 'fontsize':9, 'color':'black'})
             else: plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5 (no alcanzado)')
        except Exception as e_avg: print(f"No se pudo marcar MTF=0.5 promedio: {e_avg}")


        plt.legend(); plt.tight_layout()

    @staticmethod
    def calculate_grand_average_mtf(list_of_mtf_results, num_points=250, forced_cutoff_freq=None):
        """
        Calcula la MTF promedio a partir de una lista de resultados de MTF individuales.

        Args:
            list_of_mtf_results (list): Una lista de diccionarios. Cada diccionario
                                         debe contener al menos las claves 'frequencies'
                                         y 'mtf' con arrays NumPy válidos.
            num_points (int): Número de puntos para el eje de frecuencia común interpolado.
            forced_cutoff_freq (float, optional): Frecuencia máxima para el eje común.
                                                  Si es None, usa el mínimo de los máximos
                                                  de las MTFs de entrada.

        Returns:
            tuple: (common_freq_axis, mtf_grand_avg, mtf_std_dev, interpolated_mtfs)
                   - common_freq_axis (np.array): Eje de frecuencias común.
                   - mtf_grand_avg (np.array): MTF promedio.
                   - mtf_std_dev (np.array): Desviación estándar de las MTFs interpoladas.
                   - interpolated_mtfs (list): Lista de todas las MTFs interpoladas.
                   Retorna (None, None, None, None) si no hay suficientes datos válidos.
        """
        print(f"\n--- Calculando MTF Promedio Global ---")
        valid_results = [res for res in list_of_mtf_results if
                         res and isinstance(res.get('frequencies'), np.ndarray) and
                         isinstance(res.get('mtf'), np.ndarray) and
                         len(res['frequencies']) > 1 and len(res['mtf']) > 1 and
                         len(res['frequencies']) == len(res['mtf'])]

        if len(valid_results) < 2:
            print(f"Error: Se necesitan al menos 2 resultados MTF válidos para promediar. Se encontraron {len(valid_results)}.")
            return None, None, None, None

        print(f" Promediando {len(valid_results)} curvas MTF válidas.")

        try:
            # Determinar el eje de frecuencia común
            max_freqs = [res['frequencies'][-1] for res in valid_results]
            min_of_max_freqs = min(max_freqs)

            if forced_cutoff_freq is not None:
                 cutoff_freq = forced_cutoff_freq
                 print(f"  Usando frecuencia de corte forzada: {cutoff_freq:.2f} c/mm")
            else:
                 max_freqs = [res['frequencies'][-1] for res in valid_results]
                 cutoff_freq = min(max_freqs)
                 print(f"  Usando frecuencia de corte mínima de las entradas: {cutoff_freq:.2f} c/mm")

            if cutoff_freq <= 0:
                 print(f"Error: Frecuencia de corte calculada ({cutoff_freq:.2f}) es inválida.")
                 return None, None, None, None

            common_freq_axis = np.linspace(0, cutoff_freq, num=num_points)
            interpolated_mtfs = []

            # Interpolar cada MTF válida, manejando los extremos con NaN para un promedio robusto
            for i, res in enumerate(valid_results):
                try:
                    # Crear un array para los valores interpolados, inicializado con NaN
                    mtf_interp = np.full_like(common_freq_axis, np.nan)

                    # Definir el rango del eje común que está cubierto por los datos de esta MTF
                    min_freq_res = res['frequencies'][0]
                    max_freq_res = res['frequencies'][-1]
                    
                    # Máscara para el eje común donde tenemos datos válidos para interpolar
                    valid_mask = (common_freq_axis >= min_freq_res) & (common_freq_axis <= max_freq_res)

                    # Interpolar solo en el rango válido
                    if np.any(valid_mask):
                        mtf_interp[valid_mask] = np.interp(
                            common_freq_axis[valid_mask],
                            res['frequencies'],
                            res['mtf']
                        )
                    
                    # Forzar MTF(0) a ser 1.0 si el eje común empieza en 0
                    if common_freq_axis[0] == 0:
                        mtf_interp[0] = 1.0

                    interpolated_mtfs.append(mtf_interp)
                except Exception as e_interp:
                    print(f"Advertencia: No se pudo procesar MTF #{i} para el promedio: {e_interp}")

            if len(interpolated_mtfs) < 2:
                 print(f"Error: No quedaron suficientes MTFs después de la interpolación ({len(interpolated_mtfs)}).")
                 return None, None, None, None

            # Calcular promedio y desviación estándar usando nanmean/nanstd para ignorar los NaNs
            mtf_array = np.array(interpolated_mtfs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # Ignorar "mean of empty slice"
                mtf_grand_avg = np.nanmean(mtf_array, axis=0)
                mtf_std_dev = np.nanstd(mtf_array, axis=0)

            if mtf_std_dev is not None:
                mtf_std_dev[np.isnan(mtf_std_dev)] = 0.0

            print(f" MTF promedio global calculada ({len(interpolated_mtfs)} curvas usadas).")
            return common_freq_axis, mtf_grand_avg, mtf_std_dev, interpolated_mtfs

        except Exception as e:
            print(f"Error durante el cálculo del promedio global MTF: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None


    @staticmethod
    def plot_grand_average_mtf(common_freq_axis, mtf_grand_avg, mtf_std_dev=None,
                               interpolated_mtfs=None, poly_fit_func=None,
                               title="MTF Promedio Global", cutoff_freq=None,
                               plot_individuals=False, plot_std_dev=True):
        """
        Grafica la MTF promedio global.

        Args:
            common_freq_axis (np.array): Eje de frecuencias común.
            mtf_grand_avg (np.array): MTF promedio.
            mtf_std_dev (np.array, optional): Desviación estándar para sombrear.
            interpolated_mtfs (list, optional): Lista de MTFs individuales interpoladas
                                                para graficar en segundo plano.
            poly_fit_func (np.poly1d, optional): Función polinómica ajustada para graficar.
            title (str): Título del gráfico.
            cutoff_freq (float, optional): Frecuencia máxima a mostrar en el eje X.
                                            Si es None, se usa el máximo del eje común.
            plot_individuals (bool): Si graficar las curvas individuales interpoladas
                                     débilmente en el fondo.
            plot_std_dev (bool): Si sombrear el área de +/- 1 desviación estándar.
        """
        if common_freq_axis is None or mtf_grand_avg is None:
            print("Advertencia: No se puede graficar MTF promedio global (datos ausentes).")
            return

        print(f"\n[Graficando MTF Promedio Global: {title}]")
        try: plt.style.use('seaborn-v0_8-darkgrid')
        except OSError: plt.style.use('default')

        plt.figure("MTF Promedio Global", figsize=(10, 6)); plt.clf()

        # Graficar individuales (si se pide y existen)
        if plot_individuals and interpolated_mtfs:
            print(f"  Graficando {len(interpolated_mtfs)} curvas individuales interpoladas...")
            for i, mtf_interp in enumerate(interpolated_mtfs):
                plt.plot(common_freq_axis, mtf_interp, color='grey', alpha=0.15, lw=0.8,
                         label='_nolegend_') # Evitar leyenda para individuales

        # Graficar desviación estándar (si se pide y existe)
        if plot_std_dev and mtf_std_dev is not None:
            print("  Graficando +/- 1 desviación estándar...")
            plt.fill_between(common_freq_axis, mtf_grand_avg - mtf_std_dev,
                             mtf_grand_avg + mtf_std_dev, color='skyblue', alpha=0.4,
                             label='±1 Desv. Estándar')

        # Graficar promedio
        plt.plot(common_freq_axis, mtf_grand_avg, color='black', ls='-', lw=2.0,
                 label=f'MTF Promedio ({len(interpolated_mtfs) if interpolated_mtfs else "N/A"} curvas)')

        # Graficar ajuste polinómico (si existe)
        if poly_fit_func is not None:
            print("  Graficando ajuste polinómico...")
            mtf_poly_fit_values = np.clip(poly_fit_func(common_freq_axis), 0, 1)
            plt.plot(common_freq_axis, mtf_poly_fit_values, color='red', linestyle='--', linewidth=1.5,
                     label=f'Ajuste Polinomio (Grado {len(poly_fit_func.coeffs)-1})')

        # Configuración del gráfico
        plt.title(title)
        plt.xlabel('Frecuencia Espacial [ciclos/mm]'); plt.ylabel('MTF')
        plot_cutoff = cutoff_freq if cutoff_freq is not None else common_freq_axis[-1]
        plt.xlim(left=0, right=plot_cutoff)
        plt.ylim(bottom=-0.05, top=1.1)
        plt.grid(True, which='major', ls='-', lw=0.6); plt.grid(True, which='minor', ls=':', lw=0.4)
        plt.minorticks_on()

        # Marcar MTF 0.5 promedio
        try:
             idx_mtf05_avg = np.where(mtf_grand_avg < 0.5)[0]
             if len(idx_mtf05_avg) > 0:
                 idx_mtf05_avg = idx_mtf05_avg[0]
                 # Interpolar para encontrar la frecuencia exacta
                 if idx_mtf05_avg > 0:
                      freq_mtf05_avg = np.interp(0.5, mtf_grand_avg[idx_mtf05_avg::-1], common_freq_axis[idx_mtf05_avg::-1])
                 else: # Si el primer punto ya está por debajo de 0.5
                      freq_mtf05_avg = common_freq_axis[idx_mtf05_avg]

                 plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5')
                 plt.axvline(freq_mtf05_avg, color='grey', ls=':', lw=1.0)
                 label_x = freq_mtf05_avg + 0.05 * plot_cutoff; ha = 'left'
                 if label_x > plot_cutoff * 0.8: label_x = freq_mtf05_avg - 0.05 * plot_cutoff; ha = 'right'
                 plt.text(label_x, 0.51, f'{freq_mtf05_avg:.2f} c/mm', **{'va':'bottom', 'ha':ha, 'fontsize':9, 'color':'black'})
             else: plt.axhline(0.5, color='grey', ls=':', lw=1.0, label='MTF=0.5 (no alcanzado)')
        except Exception as e_avg: print(f"Advertencia: No se pudo marcar MTF=0.5 promedio: {e_avg}")


        plt.legend(); plt.tight_layout()
# --- Fin de mtf_analyzer.py ---
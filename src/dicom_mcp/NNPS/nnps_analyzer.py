# nnps_analyzer.py (Adaptado a IEC 62220-1:2003)

import numpy as np
import pydicom
from scipy.fft import fft2, fftshift, fftfreq
from scipy.stats import binned_statistic
# Para el ajuste polinómico 2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings # Para advertencias sobre ajustes, etc.
from typing import List, Tuple, Dict, Optional, Any, Union

# --- Importaciones de utils ---
try:
    from utils import linearize_preprocessed_image_from_df, obtener_datos_calibracion_vmp_k
except ImportError:
    print("ERROR CRÍTICO: No se pudieron importar funciones necesarias desde 'utils.py'.")
    # Definir funciones dummy o lanzar error (como antes)
    def linearize_preprocessed_image_from_df(*args, **kwargs): raise ImportError("Linealización no disponible.")
    def obtener_datos_calibracion_vmp_k(*args, **kwargs): raise ImportError("Carga de calibración no disponible.")

class NnpsAnalyzer:
    """
    Clase para calcular el Espectro de Potencia de Ruido (NPS) y el NPS Normalizado (NNPS)
    según las directrices de IEC 62220-1:2003.
    Calcula NPS 2D (W_out), perfiles 1D (H/V) y el perfil radial 1D.
    También proporciona NNPS normalizado.
    """
    def __init__(self,
                 calibration_df: pd.DataFrame,
                 rqa_factors_dict: Dict[str, float],
                 subregion_size: int = 256,
                 overlap: float = 0.5,
                 num_average_rows_1d_side: int = 7, # <--- Parámetro de entrada
                 apply_1d_smoothing: bool = True,
                 detrending_method: str = 'global_poly',
                 poly_degree: int = 2,
                 num_radial_bins: Optional[int] = None,
                 verbose: bool = True):
        """
        Inicializa el analizador NNPS.

        Args:
            calibration_df (pd.DataFrame): DataFrame con datos de calibración (VMP vs K_uGy).
            rqa_factors_dict (dict): Diccionario mapeando RQA types a sus factores (SNR_in^2 / 1000).
            subregion_size (int): Tamaño (lado) de las subregiones cuadradas. IEC 62220-1 usa 256.
            overlap (float): Fracción de solapamiento entre subregiones (0.0 a < 1.0). IEC 62220-1 usa 0.5.
            num_average_rows_1d_side (int): Número de filas/columnas a cada lado del eje central
                                            a promediar para NPS 1D H/V. IEC 62220-1 usa 7.
            apply_1d_smoothing (bool): Si aplicar el suavizado/binning final a los perfiles 1D H/V
                                      según IEC 62220-1 (usando f_int).
            detrending_method (str): Método para eliminar tendencia:
                                     'global_poly' (ajuste polinómico 2D a la ROI grande, IEC opcional),
                                     'local_mean' (restar media de cada subregión).
            poly_degree (int): Grado del polinomio para 'global_poly' detrending (IEC sugiere 2).
            num_radial_bins (int, optional): Número de bins para NPS radial. Default: usa df.
            verbose (bool): Si imprimir mensajes detallados.
        """
        # --- Validaciones ---
        if not isinstance(calibration_df, pd.DataFrame): raise TypeError("calibration_df debe ser DataFrame.")
        if not all(col in calibration_df.columns for col in ['VMP', 'K_uGy']): raise ValueError("calibration_df requiere 'VMP', 'K_uGy'.")
        if not isinstance(rqa_factors_dict, dict) or not rqa_factors_dict: raise ValueError("rqa_factors_dict debe ser dict no vacío.")
        if not isinstance(subregion_size, int) or subregion_size <= 0: raise ValueError("subregion_size debe ser int positivo.")
        if subregion_size != 256: warnings.warn(f"IEC 62220-1 especifica subregion_size=256 (usado: {subregion_size}).")
        if not isinstance(overlap, (int, float)) or not (0.0 <= overlap < 1.0): raise ValueError("overlap debe estar en [0.0, 1.0).")
        if not np.isclose(overlap, 0.5): warnings.warn(f"IEC 62220-1 especifica overlap=0.5 (usado: {overlap}).")
        if not isinstance(num_average_rows_1d_side, int) or num_average_rows_1d_side <= 0: raise ValueError("num_average_rows_1d_side debe ser int positivo.")
        if num_average_rows_1d_side != 7: warnings.warn(f"IEC 62220-1 especifica promediar 7 filas/columnas a cada lado (usado: {num_average_rows_1d_side}).")
        if detrending_method not in ['global_poly', 'local_mean']: raise ValueError("detrending_method debe ser 'global_poly' o 'local_mean'.")
        if not isinstance(poly_degree, int) or poly_degree < 0: raise ValueError("poly_degree debe ser int no negativo.")
        if detrending_method == 'global_poly' and poly_degree != 2: warnings.warn(f"IEC 62220-1 sugiere poly_degree=2 para detrending (usado: {poly_degree}).")
        if num_radial_bins is not None and (not isinstance(num_radial_bins, int) or num_radial_bins <= 0): raise ValueError("num_radial_bins debe ser int positivo si se especifica.")

        self.calibration_df = calibration_df
        self.rqa_factors_dict = rqa_factors_dict
        self.subregion_size = subregion_size
        self.overlap = overlap
        # --- CORRECCIÓN AQUÍ ---
        self.num_average_rows_1d_side = num_average_rows_1d_side # Guardar el valor
        # --- FIN CORRECCIÓN ---
        self.num_average_rows_1d_total = num_average_rows_1d_side * 2 # Calculamos el total también
        self.apply_1d_smoothing = apply_1d_smoothing
        self.detrending_method = detrending_method
        self.poly_degree = poly_degree
        self.num_radial_bins = num_radial_bins
        self.verbose = verbose

        if self.verbose:
            print("NnpsAnalyzer inicializado (conforme a IEC 62220-1):")
            print(f"  Subregion Size: {self.subregion_size}x{self.subregion_size}")
            print(f"  Overlap: {self.overlap*100:.1f}%")
            print(f"  1D H/V Avg Rows: {self.num_average_rows_1d_total} ({num_average_rows_1d_side} a cada lado)")
            print(f"  Apply 1D H/V Smoothing (f_int): {self.apply_1d_smoothing}")
            print(f"  Detrending Method: {self.detrending_method}" + (f" (Degree {self.poly_degree})" if self.detrending_method == 'global_poly' else ""))
            print(f"  1D Radial Bins: {'Default (df)' if self.num_radial_bins is None else self.num_radial_bins}")
            print(f"  Calibración: {len(self.calibration_df)} puntos, {len(self.rqa_factors_dict)} factores RQA cargados.")

    # --------------------------------------------------------------------------
    # Método Principal de Análisis
    # --------------------------------------------------------------------------
    def _calculate_linearization_slope(self, rqa_type: str) -> Optional[float]:
        """ --- NUEVO HELPER ---
        Calcula la pendiente (VMP vs quanta/area) para un RQA dado.
        Basado en la lógica de utils.linearize_preprocessed_image_from_df
        """
        if self.verbose: print(f"    Calculando pendiente de linealización para {rqa_type}...")
        try:
            if rqa_type not in self.rqa_factors_dict:
                 raise ValueError(f"RQA type '{rqa_type}' no en rqa_factors_dict.")
            if not all(col in self.calibration_df.columns for col in ['K_uGy', 'VMP']):
                 raise ValueError("calibration_df debe contener 'K_uGy' y 'VMP'.")

            snr_in_squared_factor = self.rqa_factors_dict[rqa_type] * 1000.0 # OJO: Asume que el dict tiene SNR_in^2/1000
            epsilon = 1e-9

            # Filtrar puntos con Kerma > 0
            valid_cal_data = self.calibration_df[self.calibration_df['K_uGy'] > epsilon].copy()
            if valid_cal_data.empty:
                 raise ValueError("No hay puntos de calibración válidos (Kerma > 0).")

            # Calcular x = Quanta/Area, y = VMP
            valid_cal_data['quanta_per_area'] = valid_cal_data['K_uGy'] * snr_in_squared_factor
            x_values = valid_cal_data['quanta_per_area'].values
            y_values = valid_cal_data['VMP'].values # VMP directo

            # Calcular la pendiente (y = slope' * x)
            # Evitar división por cero si x_values tiene ceros (aunque debería filtrarse por K_uGy > epsilon)
            valid_x_mask = np.abs(x_values) > epsilon
            if not np.any(valid_x_mask):
                 raise ValueError("No hay valores de quanta/area válidos (> epsilon).")

            slopes_prime = y_values[valid_x_mask] / x_values[valid_x_mask]
            slope_prime = np.mean(slopes_prime) # Media simple de pendientes individuales

            # Alternativa más robusta (mínimos cuadrados forzando intercepto a 0):
            # x_col = x_values[valid_x_mask][:, np.newaxis]
            # y_vals_f = y_values[valid_x_mask]
            # slope_prime = np.linalg.lstsq(x_col, y_vals_f, rcond=None)[0][0]

            if abs(slope_prime) < epsilon:
                 raise ValueError(f"Pendiente calculada ({slope_prime:.2e}) demasiado cercana a cero.")
            if self.verbose: print(f"    Pendiente (VMP vs q/area) calculada: {slope_prime:.4e}")
            return float(slope_prime)

        except Exception as e:
            warnings.warn(f"No se pudo calcular la pendiente de linealización: {e}")
            return None

    def analyze_flat_field_set(self,
                               flat_field_paths: List[str],
                               rqa_type: str,
                               # air_kerma_nps: float, # <--- ELIMINADO
                               roi_extract_fraction: float = 0.8) -> Dict[str, Any]:
        """
        Calcula NPS y NNPS. Estima Ka a partir del VMP promedio.

        Args:
            flat_field_paths (List[str]): Rutas a los archivos DICOM flat-field.
            rqa_type (str): Tipo de RQA ('RQA5', etc.).
            roi_extract_fraction (float): Fracción central de la imagen a usar.

        Returns:
            Dict[str, Any]: Resultados incluyendo 'mean_vmp_roi' y 'slope_prime',
                            pero SIN 'air_kerma_nps'.
        """
        if self.verbose: print(f"\n=== Iniciando Análisis NPS/NNPS para RQA: {rqa_type} (IEC 62220-1) ===")
        # if not isinstance(air_kerma_nps...): # ELIMINADO

        # --- Paso 1: Cargar Imágenes ---
        status, load_data = self._load_and_validate_images(flat_field_paths)
        if status != "OK": return self._format_error(load_data["error_details"])
        all_pixel_arrays_raw = load_data["pixel_arrays"]
        pixel_spacing = load_data["pixel_spacing"]
        ref_shape = load_data["ref_shape"]
        processed_files = load_data["processed_files"]

        # --- Paso 2: Extraer ROI ---
        status, roi_data = self._extract_central_rois(all_pixel_arrays_raw, ref_shape, roi_extract_fraction)
        if status != "OK": return self._format_error(roi_data["error_details"])
        rois_raw = roi_data["rois"]
        roi_shape = roi_data["roi_shape"]

        # --- NUEVO: Calcular VMP Promedio de ROIs Crudas ---
        if not rois_raw: return self._format_error("No se extrajeron ROIs crudas.")
        all_vmps = [np.mean(roi) for roi in rois_raw]
        mean_vmp_roi = np.mean(all_vmps)
        if self.verbose: print(f"  Mean VMP (Raw ROIs): {mean_vmp_roi:.2f}")

        # --- Calcular Pendiente de Linealización ---
        slope_prime = self._calculate_linearization_slope(rqa_type)
        if slope_prime is None:
             return self._format_error("Fallo al calcular la pendiente de linealización necesaria.")

        # --- Paso 3: Linealizar (usando la pendiente calculada) ---
        # Modificamos _linearize_rois para aceptar la pendiente o la recalculamos aquí
        if self.verbose: print(f"  [Paso 3] Linealizando {len(rois_raw)} ROIs usando slope={slope_prime:.4e}...")
        rois_linearized = []
        all_mean_linear_signals = []
        try:
            for i, roi_raw in enumerate(rois_raw):
                # Aplicar linealización directa: I = VMP / pendiente
                linearized_roi = roi_raw.astype(np.float64) / slope_prime # Asegurar float64 para precisión
                rois_linearized.append(linearized_roi)
                all_mean_linear_signals.append(np.mean(linearized_roi))
            mean_linear_signal_overall = np.mean(all_mean_linear_signals)
            if self.verbose: print(f"  Mean Linear Signal (overall avg): {mean_linear_signal_overall:.3e} quanta/area")
        except Exception as e:
             return self._format_error(f"Error durante la linealización manual: {e}")


        # --- Paso 4: Calcular y promediar NPS 2D (W_out) ---
        # (Sin cambios aquí, usa rois_linearized)
        status, nps2d_data = self._calculate_and_average_nps2d(rois_linearized, pixel_spacing)
        if status != "OK": return self._format_error(nps2d_data["error_details"])
        avg_nps_2d = nps2d_data["avg_nps_2d"]
        freq_x = nps2d_data["freq_x"]
        freq_y = nps2d_data["freq_y"]
        num_rois_averaged = nps2d_data["num_rois_averaged"]

        # Calcular NNPS 2D
        avg_nnps_2d = avg_nps_2d / (mean_linear_signal_overall**2) if mean_linear_signal_overall > 1e-9 else np.full_like(avg_nps_2d, np.nan)

        # --- Paso 5: Calcular NPS 1D H/V ---
        # (Sin cambios aquí)
        try:
            nps_1d_h, nps_1d_v, freq_1d_h, freq_1d_v, smoothing_applied = self._extract_nps_1d_hv(avg_nps_2d, freq_x, freq_y, pixel_spacing)
        except Exception as e: return self._format_error(f"Error calculando NPS 1D H/V: {e}")
        nnps_1d_h = nps_1d_h / (mean_linear_signal_overall**2) if mean_linear_signal_overall > 1e-9 else np.full_like(nps_1d_h, np.nan)
        nnps_1d_v = nps_1d_v / (mean_linear_signal_overall**2) if mean_linear_signal_overall > 1e-9 else np.full_like(nps_1d_v, np.nan)

        # --- Paso 6: Calcular NPS 1D Radial ---
        # (Sin cambios aquí)
        try:
            nps_1d_radial, freq_1d_radial = self._calculate_radial_nps(avg_nps_2d, freq_x, freq_y, pixel_spacing)
        except Exception as e: return self._format_error(f"Error calculando NPS 1D Radial: {e}")
        nnps_1d_radial = nps_1d_radial / (mean_linear_signal_overall**2) if mean_linear_signal_overall > 1e-9 else np.full_like(nps_1d_radial, np.nan)

        # --- 7. Ensamblar y devolver resultados ---
        results = {
            "status": "OK", "rqa_type": rqa_type, "num_images": num_rois_averaged,
            "pixel_spacing": pixel_spacing, "roi_shape": roi_shape,
            "subregion_size": self.subregion_size, "overlap": self.overlap,
            "detrending_method": self.detrending_method,
            "mean_linear_signal": mean_linear_signal_overall,
            # --- VALORES NUEVOS / CAMBIADOS ---
            "mean_vmp_roi": mean_vmp_roi,           # VMP promedio de ROIs crudas
            "slope_prime": slope_prime,             # Pendiente VMP vs q/area usada
            # "air_kerma_nps": air_kerma_nps,      # ELIMINADO de aquí
            # --- RESTO DE RESULTADOS ---
            "nps_2d": avg_nps_2d, "nnps_2d": avg_nnps_2d,
            "freq_x": freq_x, "freq_y": freq_y,
            "nps_1d_h": nps_1d_h, "nps_1d_v": nps_1d_v,
            "nnps_1d_h": nnps_1d_h, "nnps_1d_v": nnps_1d_v,
            "freq_1d_h": freq_1d_h, "freq_1d_v": freq_1d_v,
            "nps_1d_radial": nps_1d_radial, "nnps_1d_radial": nnps_1d_radial,
            "freq_1d_radial": freq_1d_radial,
            "num_average_rows_1d_total": self.num_average_rows_1d_total,
            "smoothing_applied_1d_hv": smoothing_applied,
            "num_radial_bins": self.num_radial_bins if self.num_radial_bins else 'Default (df)',
            "processed_files": processed_files[:num_rois_averaged],
            "error_details": None
        }
        if self.verbose: print(f"=== Análisis NPS/NNPS para RQA: {rqa_type} completado ({results['status']}) ===")
        return results
    # --------------------------------------------------------------------------
    # Métodos Internos Adaptados
    # --------------------------------------------------------------------------

    def _load_and_validate_images(self, flat_field_paths: List[str]) -> Tuple[str, Dict]:
        # ... (Sin cambios respecto a la versión anterior) ...
        if self.verbose: print(f"  [Paso 1] Cargando {len(flat_field_paths)} imágenes DICOM...")
        all_pixel_arrays_raw = []
        pixel_spacing = None
        ref_shape = None
        processed_files = []
        for fpath in flat_field_paths:
            try:
                ds = pydicom.dcmread(fpath)
                img_raw = ds.pixel_array.astype(np.float32)
                current_spacing_tuple = ds.get("PixelSpacing", ds.get("ImagerPixelSpacing"))
                if current_spacing_tuple is None: raise ValueError("PixelSpacing no encontrado.")
                current_spacing = float(current_spacing_tuple[0])
                if len(current_spacing_tuple) > 1 and not np.isclose(current_spacing, float(current_spacing_tuple[1])):
                     warnings.warn(f"Píxeles anisotrópicos detectados ({current_spacing_tuple}) en {os.path.basename(fpath)}. Usando el primero ({current_spacing:.4f}).")

                if pixel_spacing is None:
                    pixel_spacing = current_spacing
                    ref_shape = img_raw.shape
                    if self.verbose: print(f"    Ref: {os.path.basename(fpath)} -> Shape={ref_shape}, PixelSpacing={pixel_spacing:.4f} mm")
                elif not np.isclose(pixel_spacing, current_spacing, rtol=1e-4):
                     raise ValueError(f"Inconsistencia PixelSpacing: {current_spacing:.4f} vs {pixel_spacing:.4f}")
                elif img_raw.shape != ref_shape:
                     raise ValueError(f"Inconsistencia Shape: {img_raw.shape} vs {ref_shape}")

                all_pixel_arrays_raw.append(img_raw)
                processed_files.append(fpath)
            except Exception as e:
                print(f"    Error cargando/validando {os.path.basename(fpath)}: {e}. Se omite.")
                continue

        if len(all_pixel_arrays_raw) < 1:
             return "Error", {"error_details": f"No se cargaron imágenes válidas."}
        if self.verbose: print(f"  {len(all_pixel_arrays_raw)} imágenes cargadas y validadas OK.")
        return "OK", {"pixel_arrays": all_pixel_arrays_raw, "pixel_spacing": pixel_spacing, "ref_shape": ref_shape, "processed_files": processed_files}


    def _extract_central_rois(self, all_pixel_arrays_raw: List[np.ndarray], ref_shape: Tuple[int, int], roi_extract_fraction: float) -> Tuple[str, Dict]:
        # ... (Sin cambios respecto a la versión anterior) ...
        if self.verbose: print(f"  [Paso 2] Extrayendo ROI central ({roi_extract_fraction*100:.0f}%)...")
        rois_raw = []
        try:
            h, w = ref_shape
            roi_h = int(h * roi_extract_fraction); roi_w = int(w * roi_extract_fraction)
            # Asegurar que la ROI sea al menos tan grande como la subregión
            if roi_h < self.subregion_size or roi_w < self.subregion_size:
                 raise ValueError(f"La fracción de ROI ({roi_extract_fraction}) resulta en una ROI ({roi_h}x{roi_w}) más pequeña que la subregión ({self.subregion_size}). Aumenta la fracción.")
            roi_h = roi_h - (roi_h % 2); roi_w = roi_w - (roi_w % 2) # Hacer par
            y_start = (h - roi_h) // 2; x_start = (w - roi_w) // 2
            y_end = y_start + roi_h; x_end = x_start + roi_w
            roi_shape = (roi_h, roi_w)
            if self.verbose: print(f"    ROI coords: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}] -> Size={roi_h}x{roi_w}")
            for img_raw in all_pixel_arrays_raw:
                rois_raw.append(img_raw[y_start:y_end, x_start:x_end].copy()) # Usar .copy()
            return "OK", {"rois": rois_raw, "roi_shape": roi_shape}
        except Exception as e:
             return "Error", {"error_details": f"Error extrayendo ROI central: {e}"}

    def _linearize_rois(self, rois_raw: List[np.ndarray], rqa_type: str, processed_files: List[str]) -> Tuple[str, Dict]:
        # ... (Sin cambios respecto a la versión anterior) ...
        if self.verbose: print(f"  [Paso 3] Linealizando {len(rois_raw)} ROIs a quanta/área...")
        rois_linearized = []
        all_mean_linear_signals = []
        try:
            for i, roi_raw in enumerate(rois_raw):
                linearized_roi = linearize_preprocessed_image_from_df(
                    preprocessed_image=roi_raw, calibration_df=self.calibration_df,
                    rqa_type=rqa_type, rqa_factors_dict=self.rqa_factors_dict
                )
                if linearized_roi is None: raise ValueError(f"Fallo linealización ROI #{i+1} de {os.path.basename(processed_files[i])}")
                rois_linearized.append(linearized_roi)
                all_mean_linear_signals.append(np.mean(linearized_roi))
            mean_linear_signal_overall = np.mean(all_mean_linear_signals)
            if self.verbose: print(f"  Mean Linear Signal (overall avg): {mean_linear_signal_overall:.3e} quanta/area")
            return "OK", {"rois_linearized": rois_linearized, "mean_linear_signal": mean_linear_signal_overall}
        except Exception as e:
             return "Error", {"error_details": f"Error durante la linealización: {e}"}

    def _fit_global_poly(self, image_roi_linear: np.ndarray) -> np.ndarray:
        """Ajusta un polinomio 2D a la ROI y devuelve la superficie ajustada."""
        if self.verbose: print(f"    Ajustando polinomio 2D (grado {self.poly_degree}) a ROI {image_roi_linear.shape}...")
        h, w = image_roi_linear.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        # Aplanar coordenadas y valores para el ajuste
        X_coords = np.vstack((xx.ravel(), yy.ravel())).T
        Z_values = image_roi_linear.ravel()

        try:
            poly = PolynomialFeatures(degree=self.poly_degree)
            X_poly = poly.fit_transform(X_coords)
            reg = LinearRegression(fit_intercept=True) # Intercepto está en los features polinómicos
            reg.fit(X_poly, Z_values)
            # Predecir sobre la misma malla para obtener la superficie
            Z_pred = reg.predict(X_poly)
            trend_surface = Z_pred.reshape(h, w)
            if self.verbose: print("    Ajuste polinómico completado.")
            return trend_surface
        except Exception as e:
            warnings.warn(f"Fallo en el ajuste polinómico global: {e}. Se procederá sin él para esta ROI.")
            # Devolver una superficie de ceros para que la resta no haga nada
            return np.zeros_like(image_roi_linear)


    def _calculate_and_average_nps2d(self, rois_linearized: List[np.ndarray], pixel_spacing: float) -> Tuple[str, Dict]:
        """ --- MODIFICADO ---
        Calcula el NPS 2D (W_out) para cada ROI y los promedia, aplicando detrending.
        """
        if self.verbose: print(f"  [Paso 4] Calculando y promediando NPS 2D (W_out)... Detrending: {self.detrending_method}")
        all_nps_2d = []
        freq_x, freq_y = None, None
        pixel_area = pixel_spacing**2 # Asumiendo píxeles cuadrados

        for i, roi_lin in enumerate(rois_linearized):
            if self.verbose: print(f"    Procesando ROI #{i+1}/{len(rois_linearized)}...")
            try:
                # --- Aplicar Detrending ---
                global_trend_surface = None
                if self.detrending_method == 'global_poly':
                    # Ajustar polinomio a esta ROI grande
                    global_trend_surface = self._fit_global_poly(roi_lin)

                # Calcular NPS crudo (promedio de |FFT|^2) para esta ROI
                # Pasa la superficie de tendencia si existe
                nps_2d_raw_avg, current_freq_x, current_freq_y = self._calculate_raw_nps_roi(
                    roi_lin, pixel_spacing, global_trend_surface
                )

                # Verificar consistencia de frecuencias
                if freq_x is None: freq_x, freq_y = current_freq_x, current_freq_y
                elif not np.allclose(freq_x, current_freq_x) or not np.allclose(freq_y, current_freq_y):
                     print(f"    Advertencia: Inconsistencia freqs ROI #{i+1}. Se omite."); continue

                # Calcular W_out según IEC Eq. 4
                N = self.subregion_size
                w_out = nps_2d_raw_avg * pixel_area / (N * N)
                all_nps_2d.append(w_out)

            except Exception as e:
                 print(f"    Error calculando NPS 2D para ROI #{i+1}: {e}. Se omite."); continue

        if len(all_nps_2d) < 1: return "Error", {"error_details": "No se pudo calcular ningún NPS 2D válido."}

        avg_nps_2d = np.mean(np.array(all_nps_2d), axis=0) # Promedio de W_out
        num_rois_averaged = len(all_nps_2d)
        if self.verbose: print(f"  NPS 2D (W_out) promedio calculado a partir de {num_rois_averaged} ROIs.")
        return "OK", {"avg_nps_2d": avg_nps_2d, "freq_x": freq_x, "freq_y": freq_y, "num_rois_averaged": num_rois_averaged}


    def _calculate_raw_nps_roi(self, image_roi_linear: np.ndarray, pixel_spacing: float, global_trend_surface: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ --- MODIFICADO ---
        Calcula el promedio de |FFT|^2 de subregiones (NPS crudo) para UNA ROI dada.
        Aplica detrending local o usa la tendencia global proporcionada.
        """
        roi_h, roi_w = image_roi_linear.shape
        ss = self.subregion_size
        ov = self.overlap
        if roi_h < ss or roi_w < ss: raise ValueError(f"ROI ({roi_h}x{roi_w}) < subregion_size ({ss}x{ss})")
        step = max(1, int(ss * (1 - ov)))
        num_subregions_y = (roi_h - ss) // step + 1
        num_subregions_x = (roi_w - ss) // step + 1
        if num_subregions_x <= 0 or num_subregions_y <= 0: raise ValueError(f"No se pueden extraer subregiones {ss}x{ss} overlap {ov} de ROI {roi_h}x{roi_w}")

        nps_2d_sum = np.zeros((ss, ss), dtype=np.float64)
        count = 0
        for i in range(num_subregions_y):
            for j in range(num_subregions_x):
                y_start = i * step; x_start = j * step
                subregion = image_roi_linear[y_start : y_start + ss, x_start : x_start + ss]

                # --- Detrending Específico ---
                if global_trend_surface is not None:
                     # Restar la sección correspondiente de la superficie de tendencia global
                     subregion_trend = global_trend_surface[y_start : y_start + ss, x_start : x_start + ss]
                     subregion_detrended = subregion - subregion_trend
                elif self.detrending_method == 'local_mean':
                     # Restar la media local (método anterior)
                     subregion_detrended = subregion - np.mean(subregion)
                else:
                     # Si es 'global_poly' pero falló el ajuste, no hacer nada o restar media local?
                     # Por seguridad, restemos media local si no hay superficie
                     subregion_detrended = subregion - np.mean(subregion)

                fft_2d = fft2(subregion_detrended)
                nps_2d_subregion = np.abs(fftshift(fft_2d))**2 # |FFT|^2
                nps_2d_sum += nps_2d_subregion
                count += 1
        if count == 0: raise RuntimeError("No se procesaron subregiones.")

        nps_2d_raw_avg = nps_2d_sum / count # Promedio de |FFT|^2
        freq_x = fftshift(np.fft.fftfreq(ss, d=pixel_spacing))
        freq_y = fftshift(np.fft.fftfreq(ss, d=pixel_spacing))
        return nps_2d_raw_avg, freq_x, freq_y

    def _extract_nps_1d_hv(self, nps_2d: np.ndarray, freq_x: np.ndarray, freq_y: np.ndarray, pixel_spacing: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """ --- MODIFICADO ---
        Extrae perfiles NPS 1D H/V desde NPS 2D según IEC 62220-1 (promedia N filas/cols a cada lado, excluye eje).
        Aplica suavizado/binning final si self.apply_1d_smoothing es True.
        Devuelve NPS 1D (W_out), ejes de frecuencia y flag de suavizado.
        """
        if nps_2d.shape[0] != len(freq_y) or nps_2d.shape[1] != len(freq_x): raise ValueError("Dimensiones NPS 2D / freqs no coinciden.")

        # --- CORRECCIÓN AQUÍ ---
        N_side = self.num_average_rows_1d_side # Acceder al atributo guardado
        # --- FIN CORRECCIÓN ---

        if (N_side * 2) > nps_2d.shape[0] or (N_side * 2) > nps_2d.shape[1]: raise ValueError("num_average_rows_1d_side demasiado grande.")

        center_y = nps_2d.shape[0] // 2
        center_x = nps_2d.shape[1] // 2

        # Índices de filas/columnas a promediar (excluyendo el centro)
        # Usar N_side calculado arriba
        rows_indices = list(range(center_y - N_side, center_y)) + list(range(center_y + 1, center_y + N_side + 1))
        cols_indices = list(range(center_x - N_side, center_x)) + list(range(center_x + 1, center_x + N_side + 1))

        # ... (resto del método _extract_nps_1d_hv como antes) ...
        nps_1d_h_raw = np.mean(nps_2d[rows_indices, :], axis=0)
        freq_1d_h = freq_x
        nps_1d_v_raw = np.mean(nps_2d[:, cols_indices], axis=1)
        freq_1d_v = freq_y
        # ... (lógica de suavizado y retorno) ...
        smoothing_applied = False
        if self.apply_1d_smoothing:
            # ... (código de suavizado) ...
            # ... (actualizar nps_1d_h, freq_1d_h, nps_1d_v, freq_1d_v) ...
             if self.verbose: print("    Aplicando suavizado/binning final IEC a NPS 1D H/V...")
             f_int = 0.01 / pixel_spacing
             nyquist_x = np.max(np.abs(freq_1d_h))
             nyquist_y = np.max(np.abs(freq_1d_v))
             target_freq_step = 0.05 # O ajustar según necesidad
             target_freqs_h = np.arange(0, nyquist_x + target_freq_step/2, target_freq_step)
             target_freqs_v = np.arange(0, nyquist_y + target_freq_step/2, target_freq_step)
             nps_1d_h_smoothed = self._smooth_1d_profile(nps_1d_h_raw, freq_1d_h, target_freqs_h, f_int)
             nps_1d_v_smoothed = self._smooth_1d_profile(nps_1d_v_raw, freq_1d_v, target_freqs_v, f_int)
             nps_1d_h = nps_1d_h_smoothed
             freq_1d_h = target_freqs_h
             nps_1d_v = nps_1d_v_smoothed
             freq_1d_v = target_freqs_v
             smoothing_applied = True
        else:
            # ... (código para devolver perfiles raw sin suavizar) ...
            mask_h_pos = freq_1d_h >= 0
            nps_1d_h = nps_1d_h_raw[mask_h_pos]
            freq_1d_h = freq_1d_h[mask_h_pos]
            mask_v_pos = freq_1d_v >= 0
            nps_1d_v = nps_1d_v_raw[mask_v_pos]
            freq_1d_v = freq_1d_v[mask_v_pos]

        return nps_1d_h, nps_1d_v, freq_1d_h, freq_1d_v, smoothing_applied

    def _smooth_1d_profile(self, nps_1d_raw: np.ndarray, freq_raw: np.ndarray, target_freqs: np.ndarray, f_int: float) -> np.ndarray:
        """Aplica el suavizado/binning final IEC a un perfil 1D."""
        nps_smoothed = np.zeros_like(target_freqs) * np.nan
        for i, f_target in enumerate(target_freqs):
            # Definir intervalo [f - fint, f + fint]
            f_lower = f_target - f_int
            f_upper = f_target + f_int
            # Encontrar índices de frecuencias raw que caen en el intervalo
            # Usar frecuencia absoluta para el binning
            indices_in_bin = np.where((np.abs(freq_raw) >= f_lower) & (np.abs(freq_raw) <= f_upper))[0]
            if len(indices_in_bin) > 0:
                # Promediar los valores NPS correspondientes
                nps_smoothed[i] = np.mean(nps_1d_raw[indices_in_bin])
            # else: dejar NaN

        # Interpolar NaNs si existen (excepto quizás el primero/último)
        mask_nan = np.isnan(nps_smoothed)
        if np.any(mask_nan) and np.any(~mask_nan):
             nps_smoothed[mask_nan] = np.interp(target_freqs[mask_nan], target_freqs[~mask_nan], nps_smoothed[~mask_nan])

        return nps_smoothed

    def _calculate_radial_nps(self, nps_2d: np.ndarray, freq_x: np.ndarray, freq_y: np.ndarray, pixel_spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        """ --- MODIFICADO ---
        Calcula el perfil NPS 1D radial (W_out radial).
        """
        # ... (Lógica interna sin cambios respecto a la versión anterior, pero opera sobre nps_2d=W_out) ...
        fx_grid, fy_grid = np.meshgrid(freq_x, freq_y)
        fr_grid = np.sqrt(fx_grid**2 + fy_grid**2)
        df = freq_x[1] - freq_x[0]
        f_max_radial = max(np.abs(freq_x).max(), np.abs(freq_y).max())

        if self.num_radial_bins is None:
            num_bins = int(np.ceil(f_max_radial / df))
            bin_edges = np.linspace(0, num_bins * df, num_bins + 1)
            #if self.verbose: print(f"    Radial binning: Using df={df:.4f} -> {num_bins} bins up to {bin_edges[-1]:.2f}")
        else:
            num_bins = self.num_radial_bins
            bin_edges = np.linspace(0, f_max_radial, num_bins + 1)
            #if self.verbose: print(f"    Radial binning: Using {num_bins} fixed bins up to {f_max_radial:.2f}")

        nps_flat = nps_2d.ravel()
        fr_flat = fr_grid.ravel()
        valid_mask = ~np.isnan(nps_flat)

        nps_radial_mean, _, _ = binned_statistic(
            fr_flat[valid_mask], nps_flat[valid_mask], statistic='mean', bins=bin_edges
        )
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0

        # Interpolar NaNs
        mask_nan = np.isnan(nps_radial_mean)
        if np.any(mask_nan) and np.any(~mask_nan):
            #if self.verbose: print(f"    Advertencia: {np.sum(mask_nan)} bins radiales vacíos (NaN). Interpolando...")
            nps_radial_mean[mask_nan] = np.interp(bin_centers[mask_nan], bin_centers[~mask_nan], nps_radial_mean[~mask_nan])

        return nps_radial_mean, bin_centers

    # --------------------------------------------------------------------------
    # Métodos de Utilidad y Plotting Adaptados
    # --------------------------------------------------------------------------
    def _format_error(self, message: str) -> Dict[str, Any]:
        # ... (sin cambios) ...
        if self.verbose: print(f"  ERROR: {message}")
        return {"status": "Error", "error_details": message}

    def plot_nps_2d(self, results: Dict[str, Any], normalize: bool = True, log_scale: bool = True, colormap: str = 'viridis'):
        """ --- MODIFICADO ---
        Grafica el NPS 2D (W_out) o el NNPS 2D.
        """
        if results.get("status") != "OK" or results.get("nps_2d") is None:
            print("No hay datos válidos de NPS 2D para graficar."); return

        if normalize:
            data_2d = results.get("nnps_2d")
            plot_title = "NNPS 2D"
            cbar_label = "NNPS [mm^2]"
            if data_2d is None: print("Datos NNPS 2D no disponibles para graficar."); return
        else:
            data_2d = results.get("nps_2d") # W_out
            plot_title = "NPS 2D (W_out)"
            # Asumiendo unidades de q/area para señal linealizada
            cbar_label = "NPS [(q/area)^2 * mm^2]"
            if data_2d is None: print("Datos NPS 2D (W_out) no disponibles para graficar."); return

        freq_x = results["freq_x"]; freq_y = results["freq_y"]
        rqa_type = results.get("rqa_type", "N/A"); num_img = results.get("num_images", "N/A")

        plt.figure(figsize=(8, 7));
        plt.suptitle(f"{plot_title} ({rqa_type}, {num_img} imágenes)", fontsize=14)
        extent = [freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()]

        if log_scale:
            data_to_plot = np.log10(np.maximum(data_2d, 1e-15)) # Ajustar límite inferior
            cbar_label = f"log10({cbar_label.split('[')[0].strip()}) [{cbar_label.split('[')[1]}" if '[' in cbar_label else f"log10({cbar_label})"
        else:
            data_to_plot = data_2d

        im = plt.imshow(data_to_plot, cmap=colormap, extent=extent, origin='lower', interpolation='nearest', aspect='equal')
        plt.colorbar(im, label=cbar_label); plt.xlabel("Fx [c/mm]"); plt.ylabel("Fy [c/mm]")
        plt.title(f"Subreg={results['subregion_size']}, Overlap={results['overlap']:.1f}, Detrend={results['detrending_method']}")
        plt.grid(True, alpha=0.3, ls=':'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


    def plot_nps_1d(self, results: Dict[str, Any], normalize: bool = True, log_scale: bool = True, plot_radial: bool = True):
        """ --- MODIFICADO ---
        Grafica los perfiles NPS 1D (W_out) o NNPS 1D (H/V y opcionalmente Radial).
        """
        status_ok = results.get("status") == "OK"
        # Comprobar si existen los datos base (NPS o NNPS)
        if normalize:
            prefix = "nnps"
            unit = "NNPS [mm^2]"
            has_hv = results.get("nnps_1d_h") is not None and results.get("nnps_1d_v") is not None
            has_radial = results.get("nnps_1d_radial") is not None
        else:
            prefix = "nps"
            unit = "NPS [(q/area)^2 * mm^2]"
            has_hv = results.get("nps_1d_h") is not None and results.get("nps_1d_v") is not None
            has_radial = results.get("nps_1d_radial") is not None

        if not status_ok or not has_hv or (plot_radial and not has_radial):
            print(f"No hay datos {prefix.upper()} 1D válidos para graficar (H/V" + (" y Radial" if plot_radial else "") + ")."); return

        nps_h = results[f"{prefix}_1d_h"]; freq_h = results["freq_1d_h"]
        nps_v = results[f"{prefix}_1d_v"]; freq_v = results["freq_1d_v"]
        if plot_radial: nps_r = results[f"{prefix}_1d_radial"]; freq_r = results["freq_1d_radial"]

        rqa_type = results.get("rqa_type", "N/A"); num_img = results.get("num_images", "N/A")
        smoothing_applied = results.get("smoothing_applied_1d_hv", False)

        plt.figure(figsize=(10, 6));
        title = f"Perfiles {prefix.upper()} 1D ({rqa_type}, {num_img} imágenes)"
        if prefix == "nps" and smoothing_applied: title += " (IEC Smoothed H/V)"
        plt.title(title)

        # Plot H y V
        plt.plot(freq_h, nps_h, 'b-', lw=1.5, label=f'Horizontal ({self.num_average_rows_1d_total} filas prom.)')
        plt.plot(freq_v, nps_v, 'r--', lw=1.5, label=f'Vertical ({self.num_average_rows_1d_total} cols prom.)')

        # Plot Radial
        if plot_radial: plt.plot(freq_r, nps_r, 'ko-', ms=4, lw=1.5, alpha=0.8, label='Radial')

        if log_scale: plt.yscale('log'); plt.ylabel(f"{unit} (log scale)")
        else: plt.ylabel(unit)

        plt.xlabel("Frecuencia espacial [c/mm]"); plt.legend(); plt.grid(True, which='both', ls=':')
        nyquist = 0.5 / results["pixel_spacing"]
        plt.xlim(0, nyquist)
        if not log_scale: # Ajustar Y lim para escala lineal
            max_val_list = [np.nanmax(arr[freq > 0]) for arr, freq in [(nps_h, freq_h), (nps_v, freq_v)] if np.any(freq > 0)]
            if plot_radial and np.any(freq_r > 0): max_val_list.append(np.nanmax(nps_r[freq_r > 0]))
            if max_val_list: plt.ylim(bottom=0, top=np.nanmax(max_val_list) * 1.15)
            else: plt.ylim(bottom=0)
        plt.tight_layout(); plt.show()


# ==============================================================================
# Ejemplo de Uso (Adaptado)
# ==============================================================================
if __name__ == "__main__":

    print("--- Ejemplo de Uso NnpsAnalyzer (Adaptado a IEC 62220-1) ---")

    # --- Configuración Obligatoria ---
    # 1. Ruta al archivo CSV de calibración (VMP vs K_uGy)
    CALIBRATION_CSV_PATH = "src/data/linearizacion.csv" # !#!#! ACTUALIZA RUTA !#!#!

    # 2. Diccionario de Factores RQA (SNR_in^2 / 1000) - Ejemplo
    #    Unidades: 1 / (mm^2 * uGy)
    RQA_FACTORS = { # !#!#! COMPLETA CON TUS VALORES REALES !#!#!
        'RQA5': 30174.0, # Valor de IEC 62220-1 Tabla 2
        'RQA9': 31077.0, # Valor de IEC 62220-1 Tabla 2
        # ... otros RQA que uses ...
    }

    # 3. Lista de archivos DICOM de campo plano para un RQA específico
    FLAT_FIELD_FILES_RQA5 = [
        "src/test/Img35__SN1510CG__KVP=70__mAs=3.0__IE=192.dcm",
        "src/test/Img37__SN1510CG__KVP=70__mAs=3.0__IE=192.dcm"]   # !#!#! ACTUALIZA !#!#!
        # ... añade más si tienes ...
    CURRENT_RQA = 'RQA5' # El RQA correspondiente a los archivos de arriba

    # 4. Air Kerma (Ka) medido para las imágenes NPS anteriores [uGy]
    #    ¡¡¡ ESTE VALOR ES CRUCIAL Y DEBE SER EL CORRECTO PARA TUS IMÁGENES !!!
    AIR_KERMA_FOR_NPS_IMAGES = 2.5 # [uGy] <--- !#!#! ACTUALIZA ESTE VALOR !#!#!

    # 5. Parámetros NNPS (Ajustados a IEC por defecto)
    NNPS_PARAMS = {
        'subregion_size': 256,           # IEC
        'overlap': 0.5,                  # IEC
        'num_average_rows_1d_side': 7,   # IEC
        'apply_1d_smoothing': True,      # Aplicar suavizado IEC 1D H/V?
        'detrending_method': 'global_poly', # 'global_poly' (IEC) o 'local_mean'
        'poly_degree': 2,                # IEC (sugerido)
        'num_radial_bins': None,         # Opcional: número fijo de bins radiales
        'verbose': True
    }

    # --- Ejecución ---
    print(f"Cargando calibración desde: {CALIBRATION_CSV_PATH}")
    calibration_data = obtener_datos_calibracion_vmp_k(CALIBRATION_CSV_PATH)

    if calibration_data is not None:
        print("Datos de calibración cargados OK.")
        try:
            # Crear instancia del analizador
            nnps_analyzer = NnpsAnalyzer(
                calibration_df=calibration_data,
                rqa_factors_dict=RQA_FACTORS,
                **NNPS_PARAMS
            )
            print("Instancia NnpsAnalyzer creada.")

            # Verificar que existan los archivos antes de llamar
            valid_ff_files = [f for f in FLAT_FIELD_FILES_RQA5 if os.path.exists(f)]
            if not valid_ff_files:
                 print(f"Error: No se encontraron archivos DICOM en las rutas especificadas para {CURRENT_RQA}.")
            elif len(valid_ff_files) < len(FLAT_FIELD_FILES_RQA5):
                 print(f"Advertencia: No se encontraron todos los archivos DICOM especificados. Usando {len(valid_ff_files)} archivos.")

            if valid_ff_files:
                 # Ejecutar el análisis para el conjunto de imágenes, pasando el Air Kerma
                 nps_results = nnps_analyzer.analyze_flat_field_set(
                     flat_field_paths=valid_ff_files, # Usar solo los válidos
                     rqa_type=CURRENT_RQA,
                     air_kerma_nps=AIR_KERMA_FOR_NPS_IMAGES # Pasar el valor de Ka
                 )

                 # Mostrar resultados y graficar si OK
                 if nps_results and nps_results.get("status") == "OK":
                     print("\n--- Resultados del Análisis NPS/NNPS ---")
                     # Imprimir un resumen de los resultados clave
                     print(f"Status: {nps_results['status']}")
                     print(f"RQA Type: {nps_results['rqa_type']}")
                     print(f"Imágenes Usadas: {nps_results['num_images']}")
                     print(f"Air Kerma (Ka): {nps_results['air_kerma_nps']:.3f} µGy")
                     print(f"Pixel Spacing: {nps_results['pixel_spacing']:.4f} mm")
                     print(f"ROI Shape Analizada: {nps_results['roi_shape']}")
                     print(f"Mean Linear Signal: {nps_results['mean_linear_signal']:.3e} q/area")
                     print(f"Detrending: {nps_results['detrending_method']}")
                     print(f"Smoothing 1D H/V Aplicado: {nps_results['smoothing_applied_1d_hv']}")
                     print(f"NPS 2D (W_out) shape: {nps_results['nps_2d'].shape}")
                     print(f"NNPS 2D shape: {nps_results['nnps_2d'].shape}")
                     print(f"NNPS 1D Radial len: {len(nps_results['nnps_1d_radial'])}")

                     # Graficar los resultados de NNPS (normalizados)
                     print("\nGenerando gráficos NNPS...")
                     # Graficar NNPS 2D (log scale)
                     nnps_analyzer.plot_nps_2d(nps_results, normalize=True, log_scale=True)
                     # Graficar NNPS 1D H, V, Radial (log scale)
                     nnps_analyzer.plot_nps_1d(nps_results, normalize=True, log_scale=True)

                 else:
                     print("\n--- Análisis NPS/NNPS fallido ---")
                     print(f"Status: {nps_results.get('status', 'Desconocido')}")
                     print(f"Detalles: {nps_results.get('error_details', 'No disponibles')}")

        except (TypeError, ValueError, ImportError, FileNotFoundError) as e:
            print(f"\nError durante la inicialización o ejecución: {e}")
            import traceback
            traceback.print_exc() # Imprime más detalles del error
        except Exception as e_inesperado:
             print(f"\nError inesperado: {e_inesperado}")
             import traceback
             traceback.print_exc()

    else:
        print("Error fatal: No se pudieron cargar los datos de calibración.")

    print("\n--- Fin del Ejemplo ---")
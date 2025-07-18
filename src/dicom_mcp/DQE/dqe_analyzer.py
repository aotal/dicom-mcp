# dqe_analyzer.py (Nuevo archivo)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any, Tuple
import warnings

class DqeAnalyzer:
    """
    Clase para calcular la Eficiencia Cuántica de Detección (DQE)
    a partir de los resultados de MTF y NPS (W_out) según IEC 62220-1:2003.
    """

    # Tabla 2 de IEC 62220-1:2003 - SNR_in^2 por Air Kerma [1/(mm^2 * uGy)]
    RQA_SNR_FACTORS_TABLE = {
        'RQA3': 21759.0,
        'RQA5': 30174.0,
        'RQA7': 32362.0,
        'RQA9': 31077.0,
        # Añadir otros RQA si se usan y se conocen los factores
    }

    def __init__(self,
                 rqa_snr_factors: Optional[Dict[str, float]] = None,
                 verbose: bool = True):
        """
        Inicializa el analizador DQE.

        Args:
            rqa_snr_factors (dict, optional): Diccionario mapeando RQA type a SNR_in^2 [1/(mm^2*uGy)].
                                             Si es None, usa la tabla por defecto de IEC 62220-1.
            verbose (bool): Si imprimir mensajes detallados.
        """
        if rqa_snr_factors is None:
            self.rqa_snr_factors = self.RQA_SNR_FACTORS_TABLE
            if verbose: print("Usando tabla SNR_in^2 por defecto de IEC 62220-1:2003.")
        elif isinstance(rqa_snr_factors, dict):
            self.rqa_snr_factors = rqa_snr_factors
        else:
            raise TypeError("rqa_snr_factors debe ser un diccionario o None.")

        self.verbose = verbose
        if self.verbose:
            print("DqeAnalyzer inicializado.")
            print(f"  Factores SNR_in^2 conocidos: {list(self.rqa_snr_factors.keys())}")

    def calculate_dqe(self,
                      mtf_results: Dict[str, Any],
                      nps_results: Dict[str, Any],
                      direction: str) -> Dict[str, Any]:

        if self.verbose: print(f"\n=== Calculando DQE para dirección: {direction} ===")

        # Inicializar variables clave
        mtf_freq, mtf_val, nps_freq, nps_val = None, None, None, None
        rqa_type, air_kerma_nps = None, None
        common_freq = None # <--- Inicializar aquí
        mtf_interp = None # <--- Inicializar aquí

        # --- 1. Validación de Entradas ---
        # ... (como antes) ...

        # --- 2. Extraer Datos ---
        try:
            # ... (Extracción como antes -> definir mtf_freq, mtf_val, nps_freq, nps_val, rqa_type, air_kerma_nps) ...
            required_mtf = ['frequencies', 'mtf']
            if direction == 'horizontal': key_nps, key_freq = 'nps_1d_h', 'freq_1d_h'
            elif direction == 'vertical': key_nps, key_freq = 'nps_1d_v', 'freq_1d_v'
            elif direction == 'radial': key_nps, key_freq = 'nps_1d_radial', 'freq_1d_radial'
            else: return self._format_error(f"Dirección '{direction}' no reconocida.")
            required_nps = ['rqa_type', 'air_kerma_nps', key_nps, key_freq]

            if not all(k in mtf_results for k in required_mtf): return self._format_error(f"Faltan claves MTF: {required_mtf}")
            if not all(k in nps_results for k in required_nps): return self._format_error(f"Faltan claves NPS para '{direction}': {required_nps}")

            mtf_freq = mtf_results['frequencies']; mtf_val = mtf_results['mtf']
            nps_freq = nps_results[key_freq]; nps_val = nps_results[key_nps]
            rqa_type = nps_results['rqa_type']; air_kerma_nps = nps_results['air_kerma_nps']

            if mtf_val is None or mtf_freq is None: return self._format_error("Datos MTF son None.")
            if nps_val is None or nps_freq is None: return self._format_error(f"Datos NPS para '{direction}' son None.")
            if len(mtf_freq) < 2 or len(nps_freq) < 2: return self._format_error("Arrays de frecuencia < 2 puntos.")
            if len(mtf_freq) != len(mtf_val) or len(nps_freq) != len(nps_val): return self._format_error("Discrepancia longitud freqs/valores.")

        except KeyError as e: return self._format_error(f"Error extrayendo datos: Clave {e}")
        except Exception as e: return self._format_error(f"Error inesperado extrayendo datos: {e}")

        # --- 3. Calcular W_in ---
        # ... (como antes) ...
        if rqa_type not in self.rqa_snr_factors: return self._format_error(f"Factor SNR_in^2 no encontrado para RQA '{rqa_type}'.")
        snr_in_sq = self.rqa_snr_factors[rqa_type]
        w_in = air_kerma_nps * snr_in_sq
        if self.verbose: print(f"  W_in = Ka * SNR_in^2 = {air_kerma_nps:.3f} uGy * {snr_in_sq:.1f} 1/(mm^2*uGy) = {w_in:.3e} 1/mm^2")


        # --- 4. Interpolar MTF al eje de frecuencias del NPS ---
        common_freq = nps_freq # Asignar el eje común
        if self.verbose: print(f"  Interpolando MTF ({len(mtf_freq)} pts) al eje NPS ({len(common_freq)} pts)...")
        try:
            # ... (Interpolación como antes -> mtf_interp) ...
            sort_idx_mtf = np.argsort(mtf_freq)
            mtf_freq_sorted = mtf_freq[sort_idx_mtf]
            mtf_val_sorted = mtf_val[sort_idx_mtf]
            mtf_interp = np.interp(common_freq, mtf_freq_sorted, mtf_val_sorted, left=1.0, right=0.0)
            mtf_interp = np.clip(mtf_interp, 0.0, 1.0)
            if self.verbose: print("  Interpolación MTF completada.")
        except Exception as e:
            return self._format_error(f"Error durante la interpolación de MTF: {e}")

        # --- Añadir Debugging de Magnitudes (con comprobación extra) ---
        if self.verbose and common_freq is not None and mtf_interp is not None and len(common_freq) > 1: # <--- Comprobar todo
            print(f"    DEBUG: Freq range: [{common_freq.min():.2f}, {common_freq.max():.2f}] c/mm")
            print(f"    DEBUG: W_in value: {w_in:.3e} [1/mm^2]")
            # ... (resto de prints de DEBUG como antes) ...
            idx_low_f = np.argmin(np.abs(common_freq - 0.1))
            idx_mid_f = np.argmin(np.abs(common_freq - 1.0))
            idx_high_f = np.argmin(np.abs(common_freq - 2.0))

            print(f"    DEBUG @ ~{common_freq[idx_low_f]:.2f} c/mm:")
            print(f"      MTF^2: {mtf_interp[idx_low_f]**2:.3e}")
            print(f"      NPS (W_out): {nps_val[idx_low_f]:.3e} [(q/area)^2*mm^2]")
            print(f"      MTF^2 * W_in: {(mtf_interp[idx_low_f]**2 * w_in):.3e} [1/mm^2]")
            if nps_val[idx_low_f] > 1e-20:
                 print(f"      ==> Est. DQE ~ {(mtf_interp[idx_low_f]**2 * w_in / nps_val[idx_low_f]):.3e}")

            print(f"    DEBUG @ ~{common_freq[idx_mid_f]:.2f} c/mm:")
            print(f"      MTF^2: {mtf_interp[idx_mid_f]**2:.3e}")
            print(f"      NPS (W_out): {nps_val[idx_mid_f]:.3e}")
            print(f"      MTF^2 * W_in: {(mtf_interp[idx_mid_f]**2 * w_in):.3e}")
            if nps_val[idx_mid_f] > 1e-20:
                 print(f"      ==> Est. DQE ~ {(mtf_interp[idx_mid_f]**2 * w_in / nps_val[idx_mid_f]):.3e}")

            if idx_high_f != idx_mid_f and idx_high_f < len(common_freq): # Añadir chequeo idx_high_f
                print(f"    DEBUG @ ~{common_freq[idx_high_f]:.2f} c/mm:")
                print(f"      MTF^2: {mtf_interp[idx_high_f]**2:.3e}")
                print(f"      NPS (W_out): {nps_val[idx_high_f]:.3e}")
                print(f"      MTF^2 * W_in: {(mtf_interp[idx_high_f]**2 * w_in):.3e}")
                if nps_val[idx_high_f] > 1e-20:
                     print(f"      ==> Est. DQE ~ {(mtf_interp[idx_high_f]**2 * w_in / nps_val[idx_high_f]):.3e}")
        elif self.verbose:
             print("    DEBUG: No se pueden imprimir magnitudes detalladas (faltan datos o longitud < 2).")
        # --- Fin Debugging ---

        # --- 5. Calcular DQE ---
        if common_freq is None or mtf_interp is None or nps_val is None: # Comprobación final
             return self._format_error("Faltan datos necesarios (freq, mtf_interp, nps) para calcular DQE.")

        if self.verbose: print("  Calculando DQE usando DQE = MTF^2 * W_in / W_out...")
        # ... (Resto del cálculo como antes) ...
        try:
            nps_threshold = 1e-20
            dqe = np.zeros_like(common_freq) * np.nan
            valid_nps_mask = nps_val > nps_threshold
            dqe[valid_nps_mask] = (mtf_interp[valid_nps_mask]**2) * w_in / nps_val[valid_nps_mask]
            dqe = np.clip(dqe, 0.0, 1.1)
            num_invalid = np.sum(~valid_nps_mask)
            if num_invalid > 0 and self.verbose:
                 print(f"    Advertencia: {num_invalid} puntos con NPS <= {nps_threshold:.1e}. DQE puesto a NaN/0 allí.")
        except Exception as e:
            return self._format_error(f"Error inesperado al calcular DQE: {e}")


        # --- 6. Ensamblar y devolver resultados ---
        # ... (como antes) ...
        results = {
            "status": "OK", "direction": direction, "frequency": common_freq,
            "dqe": dqe, "mtf_interpolated": mtf_interp, "nps": nps_val,
            "w_in": w_in, "air_kerma_nps": air_kerma_nps, "rqa_type": rqa_type,
            "snr_in_sq": snr_in_sq, "error_details": None }
        if self.verbose: print(f"=== Cálculo DQE para {direction} completado ({results['status']}) ===")
        return results

    # --------------------------------------------------------------------------
    # Métodos de Utilidad y Plotting
    # --------------------------------------------------------------------------
    def _format_error(self, message: str) -> Dict[str, Any]:
        """Formatea un diccionario de error estándar."""
        if self.verbose: print(f"  ERROR: {message}")
        return {"status": "Error", "error_details": message}

    def plot_dqe(self, dqe_results: Dict[str, Any], plot_mtf: bool = True):
        """
        Grafica la DQE calculada.

        Args:
            dqe_results (dict): Diccionario resultado de calculate_dqe.
            plot_mtf (bool): Si incluir la MTF^2 interpolada en el gráfico.
        """
        if dqe_results.get("status") != "OK" or dqe_results.get("dqe") is None:
            print("No hay datos DQE válidos para graficar.")
            return

        freq = dqe_results["frequency"]
        dqe = dqe_results["dqe"]
        direction = dqe_results["direction"]
        rqa = dqe_results["rqa_type"]
        ka = dqe_results["air_kerma_nps"]

        plt.figure(figsize=(9, 6))
        plt.plot(freq, dqe, 'o-', ms=4, lw=1.5, label=f'DQE ({direction})')

        if plot_mtf and dqe_results.get("mtf_interpolated") is not None:
            mtf_sq = dqe_results["mtf_interpolated"]**2
            plt.plot(freq, mtf_sq, 'r--', lw=1.0, alpha=0.7, label='$MTF^2$ (interpolada)')

        plt.xlabel("Frecuencia espacial [c/mm]")
        plt.ylabel("DQE")
        plt.title(f"Detective Quantum Efficiency ({direction})\nRQA={rqa}, Ka={ka:.2f} µGy")
        plt.grid(True, which='both', ls=':')
        plt.xlim(left=0, right=freq.max() if freq.max() > 0 else 1.0)
        plt.ylim(bottom=-0.05, top=max(1.1, np.nanmax(dqe)*1.1 if np.any(~np.isnan(dqe)) else 1.1) ) # Ajustar límite superior
        plt.legend()
        plt.tight_layout()
        plt.show()

# ==============================================================================
# Ejemplo de Uso (Necesitarás resultados de MtfAnalyzer y NnpsAnalyzer)
# ==============================================================================
if __name__ == "__main__":

    print("--- Ejemplo de Uso DqeAnalyzer ---")

    # --- Asumimos que ya hemos ejecutado MtfAnalyzer y NnpsAnalyzer ---
    # --- y tenemos sus resultados en diccionarios ---

    # Ejemplo de resultados MTF (PROMEDIO para una dirección) - ¡DEBES SUSTITUIR ESTO!
    # Estos vendrían de MtfAnalyzer.calculate_average_mtf o similar
    mtf_results_example_h = {
        'frequencies': np.linspace(0, 3.5, 50),
        'mtf': np.exp(-0.8 * np.linspace(0, 3.5, 50)) # MTF de ejemplo (exponencial decreciente)
        # ... otros campos que pudiera devolver MtfAnalyzer ...
    }
    mtf_results_example_v = {
         'frequencies': np.linspace(0, 3.5, 50),
         'mtf': np.exp(-0.75 * np.linspace(0, 3.5, 50)) # Ligeramente diferente para vertical
    }


    # Ejemplo de resultados NPS (de NnpsAnalyzer) - ¡DEBES SUSTITUIR ESTO!
    # Estos vendrían de NnpsAnalyzer.analyze_flat_field_set
    pixel_pitch_ej = 0.15 # mm
    nyquist_ej = 0.5 / pixel_pitch_ej
    freq_1d_ej = np.fft.fftshift(np.fft.fftfreq(256, d=pixel_pitch_ej)) # Eje de freqs ejemplo
    mask_pos_ej = freq_1d_ej >= 0
    freq_1d_pos_ej = freq_1d_ej[mask_pos_ej]

    nps_results_example = {
        'status': 'OK',
        'rqa_type': 'RQA5',
        'air_kerma_nps': 2.5, # Ka en uGy - ¡DEBES PONER EL VALOR REAL!
        'pixel_spacing': pixel_pitch_ej,
        'mean_linear_signal': 3e9, # Ejemplo q/area
         # Ejemplo W_out(fx, 0) - Plano + ruido blanco leve
        'nps_1d_h': 1e-9 * np.ones_like(freq_1d_pos_ej) + np.random.rand(len(freq_1d_pos_ej))*1e-11,
        'freq_1d_h': freq_1d_pos_ej,
         # Ejemplo W_out(0, fy) - Similar
        'nps_1d_v': 1.05e-9 * np.ones_like(freq_1d_pos_ej) + np.random.rand(len(freq_1d_pos_ej))*1e-11,
        'freq_1d_v': freq_1d_pos_ej,
         # Ejemplo W_out(fr) - Promedio radial
        'nps_1d_radial': 1.02e-9 * np.ones_like(freq_1d_pos_ej) + np.random.rand(len(freq_1d_pos_ej))*1e-11,
        'freq_1d_radial': freq_1d_pos_ej,
        # ... otros campos que devuelve NnpsAnalyzer ...
    }
    # Simular que el primer punto (DC) es más grande o NaN si se desea
    if len(nps_results_example['nps_1d_h']) > 0: nps_results_example['nps_1d_h'][0] = np.nan
    if len(nps_results_example['nps_1d_v']) > 0: nps_results_example['nps_1d_v'][0] = np.nan
    if len(nps_results_example['nps_1d_radial']) > 0: nps_results_example['nps_1d_radial'][0] = np.nan

    # --- Crear instancia y Calcular DQE ---
    try:
        dqe_analyzer = DqeAnalyzer(verbose=True)

        # Calcular DQE Horizontal
        print("\nCalculando DQE Horizontal...")
        dqe_results_h = dqe_analyzer.calculate_dqe(
            mtf_results=mtf_results_example_h,
            nps_results=nps_results_example,
            direction='horizontal'
        )

        if dqe_results_h["status"] == "OK":
            print("DQE Horizontal calculada OK.")
            dqe_analyzer.plot_dqe(dqe_results_h, plot_mtf=True)
            # Puedes guardar estos resultados numéricos también
            # df_dqe_h = pd.DataFrame({'freq': dqe_results_h['frequency'], 'dqe': dqe_results_h['dqe']})
            # df_dqe_h.to_csv("dqe_horizontal_RQA5_2.5uGy.csv", index=False)
        else:
            print(f"Error calculando DQE Horizontal: {dqe_results_h['error_details']}")

        # Calcular DQE Vertical
        print("\nCalculando DQE Vertical...")
        dqe_results_v = dqe_analyzer.calculate_dqe(
            mtf_results=mtf_results_example_v,
            nps_results=nps_results_example,
            direction='vertical'
        )
        if dqe_results_v["status"] == "OK":
            print("DQE Vertical calculada OK.")
            dqe_analyzer.plot_dqe(dqe_results_v, plot_mtf=True)
        else:
             print(f"Error calculando DQE Vertical: {dqe_results_v['error_details']}")

        # Calcular DQE Radial (usando una MTF promedio o radial si la tienes)
        print("\nCalculando DQE Radial...")
        # Para radial, necesitarías una MTF representativa (quizás promedio H/V?)
        mtf_results_avg = {
            'frequencies': mtf_results_example_h['frequencies'],
            'mtf': (mtf_results_example_h['mtf'] + mtf_results_example_v['mtf']) / 2.0
        }
        dqe_results_r = dqe_analyzer.calculate_dqe(
             mtf_results=mtf_results_avg, # Usando MTF promedio como ejemplo
             nps_results=nps_results_example,
             direction='radial'
        )
        if dqe_results_r["status"] == "OK":
             print("DQE Radial calculada OK.")
             dqe_analyzer.plot_dqe(dqe_results_r, plot_mtf=True)
        else:
              print(f"Error calculando DQE Radial: {dqe_results_r['error_details']}")


    except Exception as e:
        print(f"\nError durante la ejecución del ejemplo DQE: {e}")

    print("\n--- Fin del Ejemplo DQE ---")
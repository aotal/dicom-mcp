# utils.py
import os
import logging
import shutil
import pydicom
import numpy as np 
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple # Asegurar que Optional, Dict, Tuple están importados
import warnings # Para calculate_linearization_slope
import math # Para convert_to_json_serializable


def apply_dicom_linearity(ds: pydicom.Dataset) -> np.ndarray:
    """Aplica la transformación de linealidad (Rescale Slope/Intercept) a los datos de píxeles."""
    pixel_array = ds.pixel_array
    
    # Obtener pendiente e intercepto, con valores por defecto si no existen
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))

    # Aplicar la transformación solo si es necesario
    if slope != 1.0 or intercept != 0.0:
        return pixel_array.astype(np.float64) * slope + intercept
    else:
        return pixel_array


def escribir_base64(ruta_archivo, cadena_base64):
    """Escribe una cadena Base64 en un archivo de texto.

    Args:
        ruta_archivo (str): La ruta completa al archivo de salida (incluyendo .txt).
        cadena_base64 (str): La cadena Base64 a escribir.
    """
    try:
        # Asegurarse de que el directorio exista
        directorio = os.path.dirname(ruta_archivo)
        os.makedirs(directorio, exist_ok=True)

        with open(ruta_archivo, "w") as f:
            f.write(cadena_base64)
    except Exception as e:
        logging.exception(f"Error al escribir Base64 en {ruta_archivo}: {e}")


def obtener_ruta_salida(ruta_original, carpeta_destino, extension=".txt"):
    """
    Genera la ruta de salida correcta para los archivos base64.

    Args:
        ruta_original (str): La ruta original del archivo DICOM limpio.
        carpeta_destino (str): La carpeta de destino (ej: "data/processed/base64").
        extension (str):  La extensión deseada (ej: ".txt").

    Returns:
        str: La ruta de salida completa.
    """
    nombre_archivo = os.path.basename(ruta_original)  # Obtiene "ImgX__etc.dcm"
    nombre_base, _ = os.path.splitext(nombre_archivo)  # Separa "ImgX__etc" de ".dcm"
    ruta_destino = os.path.join(carpeta_destino, nombre_base + extension)
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    return ruta_destino



def copiar_dicom(ruta_origen, ruta_destino):
    """Copia un archivo DICOM.

    Args:
        ruta_origen (str): Ruta completa al archivo de origen.
        ruta_destino (str): Ruta completa al archivo de destino.
    """
    try:
        # Asegurarse de que el directorio de destino exista
        directorio_destino = os.path.dirname(ruta_destino)
        os.makedirs(directorio_destino, exist_ok=True)
        shutil.copy2(ruta_origen, ruta_destino)
    except Exception as e:
        logging.exception(
            f"Error al copiar DICOM desde {ruta_origen} a {ruta_destino}: {e}"
        )


def configurar_logging(log_file):
    """Configura el logging para escribir en un archivo y en la consola."""
    # Crear un directorio para los logs si no existe
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Escribe en el archivo de log
            logging.StreamHandler()  # Escribe en la consola
        ]
    )

def leer_dicom(ruta_archivo):
    """Lee un archivo DICOM y devuelve el dataset, manejando errores."""
    try:
        ds = pydicom.dcmread(ruta_archivo)
        return ds
    except Exception as e:
        logging.error(f"Error al leer el archivo DICOM {ruta_archivo}: {e}")
        return None
    
def calcular_vmp(imagen: np.ndarray, halfroi: int) -> tuple[float, float]:
    """
    Calcula el Valor Medio de Píxel (VMP) y la desviación estándar
    dentro de una región de interés (ROI) cuadrada centrada en una imagen.

    Args:
        imagen (np.ndarray): La imagen como un array NumPy.
        halfroi (int): La mitad del tamaño del lado del ROI cuadrado.

    Returns:
        tuple[float, float]: Una tupla con el VMP y la desviación estándar,
                             o (None, None) si hay un error.
    """
    try:
        centro_x = imagen.shape[1] // 2
        centro_y = imagen.shape[0] // 2

        roi = imagen[
            centro_y - halfroi : centro_y + halfroi,
            centro_x - halfroi : centro_x + halfroi,
        ]
        vmp = np.mean(roi)
        std = np.std(roi)
        return vmp, std
    except Exception as e:
        logging.exception(f"Error al calcular VMP: {e}")
        return None, None

def obtener_factores_linealizacion(
    ruta_archivo_csv: str,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    """
    Carga un archivo CSV que contiene los factores de linealización (a, b)
    para diferentes valores de kV y filtro.

    Args:
        ruta_archivo_csv: Ruta al archivo CSV.

    Returns:
        Un dataframe de pandas
    """
    try:
        df = pd.read_csv(ruta_archivo_csv)
        return df
    except FileNotFoundError:
        logging.error(f"Archivo CSV no encontrado: {ruta_archivo_csv}")
        return None
    except Exception as e:
        logging.exception(f"Error al leer factores de linealización: {e}")
        return None


# Configuración básica de logging (si no está ya en tu utils.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def obtener_datos_calibracion_vmp_k( # Renombrada para claridad
    ruta_archivo_csv: str,
) -> pd.DataFrame | None: # Corregido el tipo de retorno
    """
    Carga un archivo CSV que contiene los datos de calibración VMP vs Kerma
    obtenidos del detector para una configuración específica (ej: RQA).

    Se espera que el CSV tenga columnas como 'K_uGy' (Kerma) y 'VMP'
    (Valor Medio del Píxel preprocesado).

    Args:
        ruta_archivo_csv: Ruta al archivo CSV con los datos de calibración.

    Returns:
        Un DataFrame de pandas con los datos cargados, o None si ocurre un error.
    """
    try:
        # pd.read_csv maneja correctamente el formato CSV generado
        df = pd.read_csv(ruta_archivo_csv)
        logging.info(f"Datos de calibración VMP vs K cargados desde: {ruta_archivo_csv}")
        # Opcional: Validar que las columnas esperadas existen
        required_columns = ['K_uGy', 'VMP'] # Ajusta si tus nombres son diferentes
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"Columnas requeridas {missing} no encontradas en {ruta_archivo_csv}")
            return None
        return df
    except FileNotFoundError:
        logging.error(f"Archivo CSV de calibración no encontrado: {ruta_archivo_csv}")
        return None
    except Exception as e:
        # Usamos logging.exception para incluir el traceback
        logging.exception(f"Error al leer datos de calibración desde {ruta_archivo_csv}: {e}")
        return None

# --- Función de extensión (sin cambios) ---
def Extension(x):
    """Obtiene la extensión de un nombre de archivo."""
    return os.path.splitext(os.path.basename(x))[1]


# --- Función de linealización (la versión que usa la tabla/CSV) ---
# Esta función AHORA usaría el DataFrame devuelto por
# obtener_datos_calibracion_vmp_k en lugar del parámetro 'table_data'
# que era una lista de diccionarios.

def linearize_pixel_array(
    pixel_array: np.ndarray,
    calibration_df: pd.DataFrame,
    rqa_type: str,
    rqa_factors_dict: dict,
    epsilon=1e-9
) -> np.ndarray | None:
    """
    Calcula la pendiente de linealización y la aplica a un array de píxeles.

    Args:
        pixel_array (np.ndarray): El array de píxeles a linealizar.
        calibration_df (pd.DataFrame): DataFrame con columnas 'K_uGy' y 'VMP'.
        rqa_type (str): Calidad de radiación (ej: 'RQA5').
        rqa_factors_dict (dict): Diccionario con factores de SNR_in^2 / 1000.
        epsilon (float): Pequeño valor para evitar división por cero.

    Returns:
        np.ndarray: El array de píxeles linealizado.
    """
    slope = calculate_linearization_slope(calibration_df, rqa_type, rqa_factors_dict)
    if slope is None:
        warnings.warn(f"No se pudo obtener la pendiente para {rqa_type}, la linealización ha fallado.")
        return None
    
    if abs(slope) < epsilon:
        warnings.warn(f"La pendiente de linealización para {rqa_type} es cercana a cero ({slope:.2e}).")
        return None

    return pixel_array.astype(np.float64) / slope

def linearize_preprocessed_image_from_df( # Renombrada para claridad
    preprocessed_image: np.ndarray,
    calibration_df: pd.DataFrame, # Ahora recibe el DataFrame
    rqa_type: str,
    rqa_factors_dict: dict,
    epsilon=1e-9
) -> np.ndarray | None:
    """
    Linealiza una imagen DR PREPROCESADA usando datos de calibración desde un DataFrame.

    Args:
        preprocessed_image (np.ndarray): Imagen ya corregida por offset y ganancia.
        calibration_df (pd.DataFrame): DataFrame cargado por
                                       obtener_datos_calibracion_vmp_k, con columnas
                                       'K_uGy' y 'VMP'.
        rqa_type (str): Cadena que identifica la calidad de radiación usada para
                        obtener los datos de calibración (ej: 'RQA5').
        rqa_factors_dict (dict): Diccionario RQA_FACTORS (SNR_in^2 / 1000).
        epsilon (float): Valor pequeño para evitar divisiones por cero.

    Returns:
        np.ndarray: Imagen linealizada en unidades de (quanta/área), o None si error.
    """
    # --- Validación de Entradas ---
    if not isinstance(preprocessed_image, np.ndarray):
        raise TypeError("preprocessed_image debe ser un numpy array.")
    
    # La validación de los otros argumentos ahora se hace en linearize_pixel_array
    return linearize_pixel_array(
        pixel_array=preprocessed_image,
        calibration_df=calibration_df,
        rqa_type=rqa_type,
        rqa_factors_dict=rqa_factors_dict,
        epsilon=epsilon
    )
    

def calculate_simple_vmp(imagen: np.ndarray, fraccion_roi=0.5) -> float:
    """Calcula el VMP en una ROI central cuadrada de la imagen."""
    if not isinstance(imagen, np.ndarray) or imagen.ndim != 2:
        warnings.warn("calculate_simple_vmp recibió una entrada no válida.")
        return np.nan
    h, w = imagen.shape
    if h == 0 or w == 0: return np.nan # Imagen vacía

    roi_h = int(h * fraccion_roi); roi_w = int(w * fraccion_roi)
    # Asegurar que el tamaño de la ROI no sea cero
    roi_h = max(1, roi_h); roi_w = max(1, roi_w)

    y0=(h-roi_h)//2; x0=(w-roi_w)//2
    # Asegurarse que los indices no se salgan
    y1 = min(h, y0 + roi_h); x1 = min(w, x0 + roi_w)
    y0 = max(0, y0); x0 = max(0, x0)

    # Volver a comprobar tamaño por si acaso el cálculo de índices da vacío
    if y1 <= y0 or x1 <= x0:
         warnings.warn(f"ROI calculada para VMP tiene tamaño cero o negativo ({y0}:{y1}, {x0}:{x1}).")
         return np.nan

    roi = imagen[y0:y1, x0:x1]
    if roi.size == 0: return np.nan # Doble chequeo por si acaso
    return float(np.mean(roi)) # Devolver float estándar

def calculate_linearization_slope(calibration_df: pd.DataFrame, rqa_type: str, rqa_factors_dict: Dict[str, float]) -> Optional[float]:
    """Calcula la pendiente (VMP vs quanta/area) para un RQA dado."""
    try:
        if not isinstance(calibration_df, pd.DataFrame): raise TypeError("calibration_df debe ser DataFrame.")
        if not isinstance(rqa_factors_dict, dict): raise TypeError("rqa_factors_dict debe ser dict.")
        if rqa_type not in rqa_factors_dict: raise ValueError(f"RQA type '{rqa_type}' no en rqa_factors_dict.")
        if not all(col in calibration_df.columns for col in ['K_uGy', 'VMP']): raise ValueError("calibration_df necesita 'K_uGy', 'VMP'.")

        factor_lin = rqa_factors_dict.get(rqa_type)
        if factor_lin is None: raise ValueError(f"Factor lin. no encontrado para {rqa_type}") # Ya cubierto por check anterior, pero OK
        snr_in_squared_factor = factor_lin * 1000.0; epsilon = 1e-9
        valid_cal_data = calibration_df[calibration_df['K_uGy'] > epsilon].copy()
        if valid_cal_data.empty: raise ValueError("No hay puntos cal. válidos (K>0).")

        valid_cal_data['quanta_per_area'] = valid_cal_data['K_uGy'] * snr_in_squared_factor
        x_values = valid_cal_data['quanta_per_area'].values; y_values = valid_cal_data['VMP'].values
        valid_x_mask = np.abs(x_values) > epsilon
        if not np.any(valid_x_mask): raise ValueError("No hay valores q/area > epsilon.")

        y_values_masked = y_values[valid_x_mask]; x_values_masked = x_values[valid_x_mask]
        if len(y_values_masked) != len(x_values_masked): raise ValueError("Discrepancia longitud X/Y pendiente.")
        if len(x_values_masked) == 0: raise ValueError("No quedan puntos válidos para pendiente.")

        slopes_prime = y_values_masked / x_values_masked
        slope_prime = np.mean(slopes_prime)
        if abs(slope_prime) < epsilon: raise ValueError(f"Pendiente {slope_prime:.2e} cercana a cero.")
        return float(slope_prime) # Devolver float estándar
    except Exception as e:
        warnings.warn(f"No se pudo calcular pendiente para {rqa_type}: {e}")
        return None

def convert_to_json_serializable(item):
    """Convierte tipos NumPy y NaN/inf a tipos nativos compatibles con JSON."""
    if isinstance(item, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [convert_to_json_serializable(elem) for elem in item]
    elif isinstance(item, np.ndarray):
        if np.issubdtype(item.dtype, np.floating):
            temp_item = item.astype(object)
            temp_item[np.isnan(item)] = None
            temp_item[np.isinf(item)] = None
            return temp_item.tolist()
        else: # Integer, bool, etc.
            return item.tolist()
    elif isinstance(item, (np.bool_)): return bool(item.item())
    elif isinstance(item, np.integer): return int(item.item())
    elif isinstance(item, np.floating):
        scalar_item = item.item()
        return None if math.isnan(scalar_item) or math.isinf(scalar_item) else scalar_item
    elif isinstance(item, float): return None if math.isnan(item) or math.isinf(item) else item
    elif isinstance(item, (str, int, bool)) or item is None: return item
    else: warnings.warn(f"Tipo no reconocido: {type(item)}. Convirtiendo a string."); return str(item)

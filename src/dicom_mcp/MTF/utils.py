# utils.py
import os
import logging
import shutil
import pydicom
import numpy as np 
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
import warnings
import math

def apply_dicom_linearity(ds: pydicom.Dataset) -> np.ndarray:
    """Applies the linearity transformation (Rescale Slope/Intercept) to the pixel data."""
    pixel_array = ds.pixel_array
    
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))

    if slope != 1.0 or intercept != 0.0:
        return pixel_array.astype(np.float64) * slope + intercept
    else:
        return pixel_array

def write_base64(file_path, base64_string):
    """Writes a Base64 string to a text file.

    Args:
        file_path (str): The full path to the output file (including .txt).
        base64_string (str): The Base64 string to write.
    """
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(base64_string)
    except Exception as e:
        logging.exception(f"Error writing Base64 to {file_path}: {e}")

def get_output_path(original_path, destination_folder, extension=".txt"):
    """Generates the correct output path for base64 files.

    Args:
        original_path (str): The original path of the clean DICOM file.
        destination_folder (str): The destination folder (e.g., "data/processed/base64").
        extension (str): The desired extension (e.g., ".txt").

    Returns:
        str: The full output path.
    """
    file_name = os.path.basename(original_path)
    base_name, _ = os.path.splitext(file_name)
    destination_path = os.path.join(destination_folder, base_name + extension)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    return destination_path

def copy_dicom(source_path, destination_path):
    """Copies a DICOM file.

    Args:
        source_path (str): Full path to the source file.
        destination_path (str): Full path to the destination file.
    """
    try:
        destination_directory = os.path.dirname(destination_path)
        os.makedirs(destination_directory, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    except Exception as e:
        logging.exception(
            f"Error copying DICOM from {source_path} to {destination_path}: {e}"
        )

def configure_logging(log_file):
    """Configures logging to write to a file and the console."""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def read_dicom(file_path):
    """Reads a DICOM file and returns the dataset, handling errors."""
    try:
        ds = pydicom.dcmread(file_path)
        return ds
    except Exception as e:
        logging.error(f"Error reading DICOM file {file_path}: {e}")
        return None
    
def calculate_vmp(image: np.ndarray, halfroi: int) -> tuple[float, float]:
    """Calculates the Mean Pixel Value (VMP) and standard deviation within a centered square ROI.

    Args:
        image (np.ndarray): The image as a NumPy array.
        halfroi (int): Half the side length of the square ROI.

    Returns:
        tuple[float, float]: A tuple with the VMP and standard deviation, or (None, None) if an error occurs.
    """
    try:
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2

        roi = image[
            center_y - halfroi : center_y + halfroi,
            center_x - halfroi : center_x + halfroi,
        ]
        vmp = np.mean(roi)
        std = np.std(roi)
        return vmp, std
    except Exception as e:
        logging.exception(f"Error calculating VMP: {e}")
        return None, None

def get_linearization_factors(
    csv_file_path: str,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    """Loads a CSV file containing linearization factors (a, b) for different kV and filter values.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame.
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
        return None
    except Exception as e:
        logging.exception(f"Error reading linearization factors: {e}")
        return None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_vmp_k_calibration_data(
    csv_file_path: str,
) -> pd.DataFrame | None:
    """Loads a CSV file containing VMP vs. Kerma calibration data from the detector for a specific configuration (e.g., RQA).

    The CSV is expected to have columns like 'K_uGy' (Kerma) and 'VMP' (Mean Pixel Value).

    Args:
        csv_file_path: Path to the CSV file with calibration data.

    Returns:
        A pandas DataFrame with the loaded data, or None if an error occurs.
    """
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"VMP vs K calibration data loaded from: {csv_file_path}")
        required_columns = ['K_uGy', 'VMP']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"Required columns {missing} not found in {csv_file_path}")
            return None
        return df
    except FileNotFoundError:
        logging.error(f"Calibration CSV file not found: {csv_file_path}")
        return None
    except Exception as e:
        logging.exception(f"Error reading calibration data from {csv_file_path}: {e}")
        return None

def Extension(x):
    """Gets the extension of a filename."""
    return os.path.splitext(os.path.basename(x))[1]

def linearize_pixel_array(
    pixel_array: np.ndarray,
    calibration_df: pd.DataFrame,
    rqa_type: str,
    rqa_factors_dict: dict,
    epsilon=1e-9
) -> np.ndarray | None:
    """Calculates the linearization slope and applies it to a pixel array.

    Args:
        pixel_array (np.ndarray): The pixel array to linearize.
        calibration_df (pd.DataFrame): DataFrame with 'K_uGy' and 'VMP' columns.
        rqa_type (str): Radiation quality (e.g., 'RQA5').
        rqa_factors_dict (dict): Dictionary with SNR_in^2 / 1000 factors.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        np.ndarray: The linearized pixel array.
    """
    slope = calculate_linearization_slope(calibration_df, rqa_type, rqa_factors_dict)
    if slope is None:
        warnings.warn(f"Could not get slope for {rqa_type}, linearization failed.")
        return None
    
    if abs(slope) < epsilon:
        warnings.warn(f"Linearization slope for {rqa_type} is close to zero ({slope:.2e}).")
        return None

    return pixel_array.astype(np.float64) / slope

def linearize_preprocessed_image_from_df(
    preprocessed_image: np.ndarray,
    calibration_df: pd.DataFrame,
    rqa_type: str,
    rqa_factors_dict: dict,
    epsilon=1e-9
) -> np.ndarray | None:
    """Linearizes a preprocessed DR image using calibration data from a DataFrame.

    Args:
        preprocessed_image (np.ndarray): The image already corrected for offset and gain.
        calibration_df (pd.DataFrame): DataFrame loaded by get_vmp_k_calibration_data.
        rqa_type (str): String identifying the radiation quality used for calibration (e.g., 'RQA5').
        rqa_factors_dict (dict): RQA_FACTORS dictionary (SNR_in^2 / 1000).
        epsilon (float): A small value to avoid division by zero.

    Returns:
        np.ndarray: The linearized image in (quanta/area) units, or None if an error occurs.
    """
    if not isinstance(preprocessed_image, np.ndarray):
        raise TypeError("preprocessed_image must be a numpy array.")
    
    return linearize_pixel_array(
        pixel_array=preprocessed_image,
        calibration_df=calibration_df,
        rqa_type=rqa_type,
        rqa_factors_dict=rqa_factors_dict,
        epsilon=epsilon
    )
    

def calculate_simple_vmp(image: np.ndarray, roi_fraction=0.5) -> float:
    """Calculates the VMP in a central square ROI of the image."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        warnings.warn("calculate_simple_vmp received invalid input.")
        return np.nan
    h, w = image.shape
    if h == 0 or w == 0: return np.nan

    roi_h = int(h * roi_fraction); roi_w = int(w * roi_fraction)
    roi_h = max(1, roi_h); roi_w = max(1, roi_w)

    y0=(h-roi_h)//2; x0=(w-roi_w)//2
    y1 = min(h, y0 + roi_h); x1 = min(w, x0 + roi_w)
    y0 = max(0, y0); x0 = max(0, x0)

    if y1 <= y0 or x1 <= x0:
         warnings.warn(f"Calculated ROI for VMP has zero or negative size ({y0}:{y1}, {x0}:{x1}).")
         return np.nan

    roi = image[y0:y1, x0:x1]
    if roi.size == 0: return np.nan
    return float(np.mean(roi))

def calculate_linearization_slope(calibration_df: pd.DataFrame, rqa_type: str, rqa_factors_dict: Dict[str, float]) -> Optional[float]:
    """Calculates the slope (VMP vs. quanta/area) for a given RQA."""
    try:
        if not isinstance(calibration_df, pd.DataFrame): raise TypeError("calibration_df must be a DataFrame.")
        if not isinstance(rqa_factors_dict, dict): raise TypeError("rqa_factors_dict must be a dict.")
        if rqa_type not in rqa_factors_dict: raise ValueError(f"RQA type '{rqa_type}' not in rqa_factors_dict.")
        if not all(col in calibration_df.columns for col in ['K_uGy', 'VMP']): raise ValueError("calibration_df needs 'K_uGy', 'VMP'.")

        factor_lin = rqa_factors_dict.get(rqa_type)
        if factor_lin is None: raise ValueError(f"Linearization factor not found for {rqa_type}")
        snr_in_squared_factor = factor_lin * 1000.0; epsilon = 1e-9
        valid_cal_data = calibration_df[calibration_df['K_uGy'] > epsilon].copy()
        if valid_cal_data.empty: raise ValueError("No valid calibration points (K>0).")

        valid_cal_data['quanta_per_area'] = valid_cal_data['K_uGy'] * snr_in_squared_factor
        x_values = valid_cal_data['quanta_per_area'].values; y_values = valid_cal_data['VMP'].values
        valid_x_mask = np.abs(x_values) > epsilon
        if not np.any(valid_x_mask): raise ValueError("No q/area values > epsilon.")

        y_values_masked = y_values[valid_x_mask]; x_values_masked = x_values[valid_x_mask]
        if len(y_values_masked) != len(x_values_masked): raise ValueError("X/Y length mismatch for slope calculation.")
        if len(x_values_masked) == 0: raise ValueError("No valid points left for slope calculation.")

        slopes_prime = y_values_masked / x_values_masked
        slope_prime = np.mean(slopes_prime)
        if abs(slope_prime) < epsilon: raise ValueError(f"Slope {slope_prime:.2e} is close to zero.")
        return float(slope_prime)
    except Exception as e:
        warnings.warn(f"Could not calculate slope for {rqa_type}: {e}")
        return None

def convert_to_json_serializable(item):
    """Converts NumPy types and NaN/inf to native types compatible with JSON."""
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
        else:
            return item.tolist()
    elif isinstance(item, (np.bool_)): return bool(item.item())
    elif isinstance(item, np.integer): return int(item.item())
    elif isinstance(item, np.floating):
        scalar_item = item.item()
        return None if math.isnan(scalar_item) or math.isinf(scalar_item) else scalar_item
    elif isinstance(item, float): return None if math.isnan(item) or math.isinf(item) else item
    elif isinstance(item, (str, int, bool)) or item is None: return item
    else: warnings.warn(f"Unrecognized type: {type(item)}. Converting to string."); return str(item)
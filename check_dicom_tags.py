

import pydicom
import numpy as np

file_path = "dcm/mtf.dcm"

try:
    ds = pydicom.dcmread(file_path)

    rescale_intercept = ds.get("RescaleIntercept", "N/A")
    rescale_slope = ds.get("RescaleSlope", "N/A")
    
    print(f"--- Metadatos de Linealización para: {file_path} ---")
    print(f"Rescale Intercept (0028,1052): {rescale_intercept}")
    print(f"Rescale Slope (0028,1053): {rescale_slope}")

    # Accessing pixel_array applies the transformations
    pixel_array = ds.pixel_array
    print(f"\nTipo de datos de ds.pixel_array (después de aplicar transformaciones): {pixel_array.dtype}")
    
    # Check if the slope and intercept have been applied by checking the data type. 
    # If they are floats, they have likely been applied.
    if np.issubdtype(pixel_array.dtype, np.floating):
        print("La pendiente/intercepto de reescalado probablemente se ha aplicado (el tipo de datos es flotante).")
    else:
        print("La pendiente/intercepto de reescalado NO se ha aplicado (el tipo de datos no es flotante).")

except FileNotFoundError:
    print(f"Error: El archivo no se encontró en la ruta: {file_path}")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo DICOM: {e}")


# --- En el archivo getroi.py ---
import pydicom
import numpy as np
import cv2
import json
import math
import os
from skimage.filters import threshold_otsu
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    plt = None
    Rectangle = None

class RoiExtractor:
    """
    Clase para encontrar un objeto central en un array de píxeles
    y extraer ROIs específicas para análisis MTF.
    Opera sobre un array de píxeles ya cargado y linealizado.
    """
    def __init__(self, pixel_array: np.ndarray, pixel_spacing_mm: float, verbose=True):
        self.pixel_array = pixel_array
        self.pixel_spacing_mm = pixel_spacing_mm
        self.verbose = verbose
        
        self.binary_image = None
        self.ref_contour = None
        self.ref_centroid = None
        self.ref_angle = None
        self.img_shape_yx = self.pixel_array.shape

        if self.pixel_array is None or self.pixel_array.size == 0:
            raise ValueError("El array de píxeles de entrada no puede estar vacío.")
        
        self._analyze_geometry()

    def _analyze_geometry(self):
        """Binarización y análisis de contorno para encontrar el centroide del objeto."""
        # --- Binarizar ---
        # Otsu funciona mejor con enteros. Normalizamos el array de flotantes a un rango de 16 bits
        # para replicar el comportamiento original que operaba sobre `image_16bit`.
        try:
            pa_min, pa_max = np.min(self.pixel_array), np.max(self.pixel_array)
            if pa_max > pa_min:
                image_for_otsu = ((self.pixel_array - pa_min) / (pa_max - pa_min) * 65535).astype(np.uint16)
            else:
                image_for_otsu = self.pixel_array.astype(np.uint16) # Array plano

            threshold_value = threshold_otsu(image_for_otsu)
            if self.verbose: print(f"Umbral Otsu (sobre array normalizado): {threshold_value}")
            # Objeto=0 (negro), Fondo=255 (blanco)
            self.binary_image = (image_for_otsu >= threshold_value).astype(np.uint8) * 255
        except Exception as e:
            raise RuntimeError(f"Error calculando umbral Otsu o binarizando: {e}")

        # --- Analizar Imagen Binaria ---
        try:
            # Asumiendo objeto = 0 (negro)
            img_inv = cv2.bitwise_not(self.binary_image)
            contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: raise ValueError("No se encontraron contornos.")

            self.ref_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(self.ref_contour)
            if M["m00"] == 0: raise ValueError("Contorno con área cero.")

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            self.ref_centroid = (cx, cy)

            # Calcular ángulo
            mu20 = M['mu20'] / M['m00']; mu02 = M['mu02'] / M['m00']; mu11 = M['mu11'] / M['m00']
            self.ref_angle = math.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
            if self.verbose: print(f"Objeto encontrado: Centroide=({cx:.1f}, {cy:.1f}), Ángulo={self.ref_angle:.1f}°")

        except Exception as e:
            raise RuntimeError(f"Error analizando imagen binaria: {e}")

    def extract_mtf_rois(self, roi1_offset_mm, roi1_shape_yx, roi2_offset_mm, roi2_shape_yx):
        """
        Extrae ROIs para MTF relativo al centroide detectado.

        Args:
            roi1_offset_mm (tuple): Offset (dx, dy) en mm para ROI 1.
            roi1_shape_yx (tuple): Forma (alto, ancho) en píxeles para ROI 1.
            roi2_offset_mm (tuple): Offset (dx, dy) en mm para ROI 2.
            roi2_shape_yx (tuple): Forma (alto, ancho) en píxeles para ROI 2.

        Returns:
            list: [roi1_array, roi2_array] como arrays float64, o None si falla.
        """
        if self.ref_centroid is None or self.pixel_spacing_mm is None or self.pixel_array is None:
            print("Error: El objeto no se procesó correctamente antes de extraer ROIs.")
            return None

        rois_extracted = []
        roi_params = [
            {"offset": roi1_offset_mm, "shape": roi1_shape_yx, "id": 1},
            {"offset": roi2_offset_mm, "shape": roi2_shape_yx, "id": 2}
        ]

        img_h, img_w = self.img_shape_yx
        obj_cx, obj_cy = self.ref_centroid

        for params in roi_params:
            roi_id = params["id"]
            offset_x_mm, offset_y_mm = params["offset"]
            roi_h, roi_w = params["shape"] # Alto, Ancho

            # Calcular centro de la ROI en píxeles
            roi_center_x = obj_cx + (offset_x_mm / self.pixel_spacing_mm)
            roi_center_y = obj_cy + (offset_y_mm / self.pixel_spacing_mm)

            # Calcular coordenadas de inicio/fin (top-left based)
            x_start = max(0, int(round(roi_center_x - roi_w / 2)))
            x_end = min(img_w, int(round(roi_center_x + roi_w / 2)))
            y_start = max(0, int(round(roi_center_y - roi_h / 2)))
            y_end = min(img_h, int(round(roi_center_y + roi_h / 2)))

            # Extraer ROI del array de píxeles ya linealizado (float64)
            cropped_roi = self.pixel_array[y_start:y_end, x_start:x_end]

            if self.verbose:
                print(f"ROI {roi_id}: Centro=({roi_center_x:.1f}, {roi_center_y:.1f})px <- Offset={params['offset']}mm")
                print(f"       Extracción: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}], Forma={cropped_roi.shape}")

            if cropped_roi.size == 0:
                print(f"Advertencia: ROI {roi_id} extraída está vacía!")
                rois_extracted.append(np.array([], dtype=np.float64)) # Añadir array vacío
            else:
                 rois_extracted.append(cropped_roi)

        return rois_extracted

    # --- Otros métodos/propiedades útiles ---
    @property
    def pixel_spacing(self):
        return self.pixel_spacing_mm

    @property
    def centroid(self):
        return self.ref_centroid

# --- Ejemplo de uso (opcional, mover a otro script) ---
if __name__ == "__main__":
     # Este ejemplo de uso ya no funcionará directamente porque necesita un pixel_array.
     # Se deja como referencia de la lógica de llamada.
     pass
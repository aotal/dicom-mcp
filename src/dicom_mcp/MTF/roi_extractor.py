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
    Clase para cargar una imagen DICOM, encontrar un objeto central
    y extraer ROIs específicas para análisis MTF.
    """
    def __init__(self, dicom_filepath, verbose=True):
        self.dicom_filepath = dicom_filepath
        self.verbose = verbose
        self.ds = None
        self.image_16bit = None
        self.pixel_spacing_mm = None
        self.binary_image = None
        self.ref_contour = None
        self.ref_centroid = None
        self.ref_angle = None
        self.img_shape_yx = None

        if not os.path.exists(dicom_filepath):
            raise FileNotFoundError(f"Archivo DICOM no encontrado: {dicom_filepath}")

        self._load_and_process_dicom()

    def _load_and_process_dicom(self):
        """Carga interna, binarización y análisis de contorno."""
        # --- Leer DICOM y Pixel Spacing (adaptado de binarize_dicom) ---
        try:
            self.ds = pydicom.dcmread(self.dicom_filepath)
            self.image_16bit = self.ds.pixel_array.astype(np.int16) # Usar int16 o uint16 según sea necesario
            self.img_shape_yx = self.image_16bit.shape
            if self.image_16bit is None or self.image_16bit.size == 0:
                raise ValueError("No se encontraron datos de píxeles.")
            if self.verbose: print(f"DICOM '{os.path.basename(self.dicom_filepath)}' leído ({self.img_shape_yx}, {self.image_16bit.dtype}).")

            spacing = self.ds.get("PixelSpacing", self.ds.get("ImagerPixelSpacing", [1.0, 1.0]))
            self.pixel_spacing_mm = float(spacing[0])
            if len(spacing) > 1 and abs(float(spacing[0]) - float(spacing[1])) > 1e-6:
                if self.verbose: print(f"Advertencia: Píxeles anisotrópicos ({spacing}). Usando primer valor ({self.pixel_spacing_mm:.4f} mm).")
            if self.verbose: print(f"Tamaño de Píxel: {self.pixel_spacing_mm:.4f} mm/píxel")

        except Exception as e:
            raise RuntimeError(f"Error al leer DICOM o pixel spacing: {e}")

        # --- Binarizar (adaptado de binarize_dicom) ---
        try:
            threshold_value = threshold_otsu(self.image_16bit)
            if self.verbose: print(f"Umbral Otsu: {threshold_value}")
            # Objeto=0 (negro), Fondo=255 (blanco)
            self.binary_image = (self.image_16bit >= threshold_value).astype(np.uint8) * 255
        except Exception as e:
            raise RuntimeError(f"Error calculando umbral Otsu o binarizando: {e}")

        # --- Analizar Imagen Binaria (adaptado de analyze_binary_image) ---
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

            # Calcular ángulo (opcional, podría no ser necesario para extracción por offset)
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
            list: [roi1_array, roi2_array] como arrays float32, o None si falla.
        """
        if self.ref_centroid is None or self.pixel_spacing_mm is None or self.image_16bit is None:
            print("Error: El objeto DICOM no se procesó correctamente antes de extraer ROIs.")
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

            # Extraer ROI (asegurarse de que sea float32)
            cropped_roi = self.image_16bit[y_start:y_end, x_start:x_end].astype(np.float32)

            if self.verbose:
                print(f"ROI {roi_id}: Centro=({roi_center_x:.1f}, {roi_center_y:.1f})px <- Offset={params['offset']}mm")
                print(f"       Extracción: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}], Forma={cropped_roi.shape}")

            if cropped_roi.size == 0:
                print(f"Advertencia: ROI {roi_id} extraída está vacía!")
                # Podrías decidir devolver None o un array vacío
                # return None # Abortar si una ROI falla? O continuar?
                rois_extracted.append(np.array([], dtype=np.float32)) # Añadir array vacío
            else:
                 rois_extracted.append(cropped_roi)

        # Visualización opcional (adaptada de extraer_rois_mtf)
        # ... (se podría añadir un método show_rois() si se desea) ...

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
     INPUT_DICOM_FILE = "src/MTF/Img6__SN2024H1__KVP=70__mAs=13.0__IE=729.dcm"
     ROI1_OFFSET_MM = (13, 0); ROI1_SHAPE_YX = (100, 200) # Alto, Ancho
     ROI2_OFFSET_MM = (0, -14); ROI2_SHAPE_YX = (200, 100) # Alto, Ancho

     try:
         extractor = RoiExtractor(INPUT_DICOM_FILE)
         print(f"Pixel spacing from extractor: {extractor.pixel_spacing}")
         print(f"Centroid from extractor: {extractor.centroid}")

         rois = extractor.extract_mtf_rois(ROI1_OFFSET_MM, ROI1_SHAPE_YX,
                                           ROI2_OFFSET_MM, ROI2_SHAPE_YX)

         if rois and rois[0].size > 0 and rois[1].size > 0:
             print("Ambas ROIs extraídas con éxito.")
             # np.save("roi1_clase.npy", rois[0])
             # np.save("roi2_clase.npy", rois[1])
         else:
             print("Fallo al extraer una o ambas ROIs.")

     except (FileNotFoundError, RuntimeError, ValueError) as e:
         print(f"Error procesando DICOM: {e}")
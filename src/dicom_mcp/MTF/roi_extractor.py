# roi_extractor.py
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
    """A class to find a central object in a pixel array and extract specific ROIs for MTF analysis.

    This class operates on an already loaded and linearized pixel array.
    """
    def __init__(self, pixel_array: np.ndarray, pixel_spacing_mm: float, verbose=True):
        """Initializes the RoiExtractor.

        Args:
            pixel_array (np.ndarray): The input pixel array.
            pixel_spacing_mm (float): The pixel spacing in millimeters.
            verbose (bool): If True, prints detailed messages.
        """
        self.pixel_array = pixel_array
        self.pixel_spacing_mm = pixel_spacing_mm
        self.verbose = verbose
        
        self.binary_image = None
        self.ref_contour = None
        self.ref_centroid = None
        self.ref_angle = None
        self.img_shape_yx = self.pixel_array.shape

        if self.pixel_array is None or self.pixel_array.size == 0:
            raise ValueError("The input pixel array cannot be empty.")
        
        self._analyze_geometry()

    def _analyze_geometry(self):
        """Performs binarization and contour analysis to find the object's centroid."""
        try:
            pa_min, pa_max = np.min(self.pixel_array), np.max(self.pixel_array)
            if pa_max > pa_min:
                image_for_otsu = ((self.pixel_array - pa_min) / (pa_max - pa_min) * 65535).astype(np.uint16)
            else:
                image_for_otsu = self.pixel_array.astype(np.uint16)

            threshold_value = threshold_otsu(image_for_otsu)
            if self.verbose: print(f"Otsu threshold (on normalized array): {threshold_value}")
            self.binary_image = (image_for_otsu >= threshold_value).astype(np.uint8) * 255
        except Exception as e:
            raise RuntimeError(f"Error calculating Otsu threshold or binarizing: {e}")

        try:
            img_inv = cv2.bitwise_not(self.binary_image)
            contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: raise ValueError("No contours found.")

            self.ref_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(self.ref_contour)
            if M["m00"] == 0: raise ValueError("Contour with zero area.")

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            self.ref_centroid = (cx, cy)

            mu20 = M['mu20'] / M['m00']; mu02 = M['mu02'] / M['m00']; mu11 = M['mu11'] / M['m00']
            self.ref_angle = math.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
            if self.verbose: print(f"Object found: Centroid=({cx:.1f}, {cy:.1f}), Angle={self.ref_angle:.1f}°")

        except Exception as e:
            raise RuntimeError(f"Error analyzing binary image: {e}")

    def extract_mtf_rois(self, roi1_offset_mm, roi1_shape_yx, roi2_offset_mm, roi2_shape_yx):
        """Extracts ROIs for MTF analysis relative to the detected centroid.

        Args:
            roi1_offset_mm (tuple): The offset (dx, dy) in mm for ROI 1.
            roi1_shape_yx (tuple): The shape (height, width) in pixels for ROI 1.
            roi2_offset_mm (tuple): The offset (dx, dy) in mm for ROI 2.
            roi2_shape_yx (tuple): The shape (height, width) in pixels for ROI 2.

        Returns:
            list: A list containing [roi1_array, roi2_array] as float64 arrays, or None if it fails.
        """
        if self.ref_centroid is None or self.pixel_spacing_mm is None or self.pixel_array is None:
            print("Error: The object was not processed correctly before extracting ROIs.")
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
            roi_h, roi_w = params["shape"]

            roi_center_x = obj_cx + (offset_x_mm / self.pixel_spacing_mm)
            roi_center_y = obj_cy + (offset_y_mm / self.pixel_spacing_mm)

            x_start = max(0, int(round(roi_center_x - roi_w / 2)))
            x_end = min(img_w, int(round(roi_center_x + roi_w / 2)))
            y_start = max(0, int(round(roi_center_y - roi_h / 2)))
            y_end = min(img_h, int(round(roi_center_y + roi_h / 2)))

            cropped_roi = self.pixel_array[y_start:y_end, x_start:x_end]

            if self.verbose:
                print(f"ROI {roi_id}: Center=({roi_center_x:.1f}, {roi_center_y:.1f})px <- Offset={params['offset']}mm")
                print(f"       Extraction: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}], Shape={cropped_roi.shape}")

            if cropped_roi.size == 0:
                print(f"Warning: Extracted ROI {roi_id} is empty!")
                rois_extracted.append(np.array([], dtype=np.float64))
            else:
                 rois_extracted.append(cropped_roi)

        return rois_extracted

    @property
    def pixel_spacing(self):
        """Returns the pixel spacing in millimeters."""
        return self.pixel_spacing_mm

    @property
    def centroid(self):
        """Returns the detected centroid of the object."""
        return self.ref_centroid

if __name__ == "__main__":
     pass

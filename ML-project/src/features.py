import cv2
import numpy as np
from typing import Dict, Optional
from PIL import Image


def _to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR format."""
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def calculate_leaf_features(image: Image.Image) -> Optional[Dict[str, float]]:
    """
    Extract leaf features:
    - gcv (green color value %)
    - area (leaf contour area)
    - aspect_ratio (width / height of bounding box)
    - roundness (4π * area / perimeter²)
    """
    img_bgr = _to_bgr(image)

    # Convert to HSV and create mask for green
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # Extract only leaf pixels
    leaf_pixels = img_bgr[mask == 255]
    if leaf_pixels.size == 0:
        return None  # No green area detected

    # Average color (BGR → RGB)
    avg_bgr = np.mean(leaf_pixels, axis=0)
    avg_rgb = avg_bgr[::-1].astype(np.uint8)

    # --- CIEDE2000 color difference ---
    def ciede2000(lab1, lab2):
        L1, a1, b1 = map(float, lab1)
        L2, a2, b2 = map(float, lab2)
        avg_L = (L1 + L2) / 2
        C1 = (a1**2 + b1**2) ** 0.5
        C2 = (a2**2 + b2**2) ** 0.5
        avg_C = (C1 + C2) / 2
        G = 0.5 * (1 - ((avg_C**7) / (avg_C**7 + 25**7)) ** 0.5)
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = (a1p**2 + b1**2) ** 0.5
        C2p = (a2p**2 + b2**2) ** 0.5
        avg_Cp = (C1p + C2p) / 2
        h1p = np.degrees(np.arctan2(b1, a1p)) % 360
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360
        if abs(h1p - h2p) > 180:
            avg_hp = ((h1p + h2p + 360) if (h1p + h2p) < 360 else (h1p + h2p - 360)) / 2
        else:
            avg_hp = (h1p + h2p) / 2
        T = (
            1
            - 0.17 * np.cos(np.radians(avg_hp - 30))
            + 0.24 * np.cos(np.radians(2 * avg_hp))
            + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
            - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
        )
        delta_hp = h2p - h1p
        if abs(delta_hp) > 180:
            delta_hp -= 360 if delta_hp > 0 else -360
        delta_Lp = L2 - L1
        delta_Cp = C2p - C1p
        delta_Hp = 2 * (C1p * C2p) ** 0.5 * np.sin(np.radians(delta_hp) / 2)
        Sl = 1 + ((0.015 * (avg_L - 50) ** 2) / ((20 + (avg_L - 50) ** 2) ** 0.5))
        Sc = 1 + 0.045 * avg_Cp
        Sh = 1 + 0.015 * avg_Cp * T
        delta_theta = 30 * np.exp(-(((avg_hp - 275) / 25) ** 2))
        Rc = 2 * ((avg_Cp**7) / (avg_Cp**7 + 25**7)) ** 0.5
        Rt = -np.sin(np.radians(2 * delta_theta)) * Rc
        return (
            (delta_Lp / Sl) ** 2
            + (delta_Cp / Sc) ** 2
            + (delta_Hp / Sh) ** 2
            + Rt * (delta_Cp / Sc) * (delta_Hp / Sh)
        ) ** 0.5

    # LAB values
    avg_lab = cv2.cvtColor(np.uint8([[avg_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    green_lab = cv2.cvtColor(np.uint8([[[0, 128, 0]]]), cv2.COLOR_RGB2LAB)[0][0]
    yellow_lab = cv2.cvtColor(np.uint8([[[200, 200, 50]]]), cv2.COLOR_RGB2LAB)[0][0]

    d1 = ciede2000(avg_lab, green_lab)
    d2 = ciede2000(avg_lab, yellow_lab)

    gcv = 100 * (1 - d1 / (d1 + d2)) if (d1 + d2) != 0 else 100.0

    # --- Shape features ---
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main))

    x, y, w, h = cv2.boundingRect(main)
    aspect_ratio = float(w) / h if h != 0 else 0.0

    peri = cv2.arcLength(main, True)
    roundness = (4 * np.pi * area) / (peri**2) if peri != 0 else 0.0

    return {
        "gcv": float(gcv),
        "area": area,
        "aspect_ratio": float(aspect_ratio),
        "roundness": float(roundness),
    }

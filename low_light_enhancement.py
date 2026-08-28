"""
Low-Light Enhancement
Detects dark/underexposed images and enhances visibility using
CLAHE (Contrast Limited Adaptive Histogram Equalization) + gamma correction,
so downstream segmentation gets a clearer image to work with.
"""

import cv2
import numpy as np


def is_dark_image(image: np.ndarray, threshold: float = 90.0) -> dict:
    """
    Detects whether an image is under-lit based on average brightness (V channel of HSV).
    Returns whether it's dark, plus the measured brightness.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    return {
        "is_dark": brightness < threshold,
        "brightness": round(brightness, 2),
        "threshold": threshold,
    }


def _apply_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE on the L channel in LAB color space to boost local contrast
    without blowing out colors, then converts back to BGR.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _apply_gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Brightens midtones/shadows using gamma correction.
    gamma > 1 brightens a dark image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


def enhance_low_light(image: np.ndarray, brightness_threshold: float = 90.0) -> dict:
    """
    Main entry point. Detects if an image is dark, and if so, enhances it.
    Always returns an image (enhanced if needed, original otherwise) so it
    can be fed straight into the segmentation step.
    """
    dark_check = is_dark_image(image, threshold=brightness_threshold)

    if not dark_check["is_dark"]:
        return {
            "enhanced": False,
            "brightness_before": dark_check["brightness"],
            "brightness_after": dark_check["brightness"],
            "image": image,
        }

    # Scale gamma based on how dark the image is - darker images get more boost
    darkness_ratio = dark_check["brightness"] / brightness_threshold
    gamma = 1.2 + (1.0 - darkness_ratio) * 1.3  # roughly 1.2 - 2.5 range

    gamma_corrected = _apply_gamma_correction(image, gamma=gamma)
    final_image = _apply_clahe(gamma_corrected)

    after_brightness = float(np.mean(cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)[:, :, 2]))

    return {
        "enhanced": True,
        "brightness_before": dark_check["brightness"],
        "brightness_after": round(after_brightness, 2),
        "gamma_used": round(gamma, 2),
        "image": final_image,
    }


def enhance_low_light_from_path(image_path: str, output_path: str = None, brightness_threshold: float = 90.0) -> dict:
    """
    Convenience wrapper: reads an image from disk, enhances it if needed,
    optionally writes the result to output_path, and returns the report
    (without the raw image array, so it's easy to log/serialize).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    result = enhance_low_light(image, brightness_threshold=brightness_threshold)

    if output_path:
        cv2.imwrite(output_path, result["image"])

    report = {k: v for k, v in result.items() if k != "image"}
    report["output_path"] = output_path
    return report
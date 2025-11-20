# src/utils/image_ops.py
"""
image_ops.py
-------------

Utility functions for reading, resizing, normalizing, and converting images.

Responsibilities:
- Read image from disk (PIL/OpenCV)
- Convert to RGB
- Resize and pad
- Normalize for model input
- Convert to tensor-ready formats

Tools:
- Pillow
- numpy
- cv2 (optional)

TODO:
- Add batch image loader
"""

import cv2


def save_image(path, img):
    cv2.imwrite(str(path), img)

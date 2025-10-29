import cv2
import numpy as np

def translate_image(image_path, shift_x, shift_y, target_size=(256, 256)):
    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape

    # Create white canvas same size as original image
    canvas = np.ones((h, w), dtype=np.uint8) * 255

    # Calculate roi coordinates with boundary checks
    x_start = max(0, shift_x)
    y_start = max(0, shift_y)
    x_end = min(w, w + shift_x)
    y_end = min(h, h + shift_y)

    img_x_start = max(0, -shift_x)
    img_y_start = max(0, -shift_y)
    img_x_end = img_x_start + (x_end - x_start)
    img_y_end = img_y_start + (y_end - y_start)

    # Paste shifted image onto canvas
    canvas[y_start:y_end, x_start:x_end] = img[img_y_start:img_y_end, img_x_start:img_x_end]

    # Resize to target size
    resized = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)

    return resized



from pathlib import Path

img = Path.cwd() / "img_testing" / 'Ltype1.png'
img256 = translate_image(img, shift_x=0, shift_y=-50, target_size=(256, 256))
cv2.imwrite(str(Path.cwd() / "img_testing" / "translated_img.png"), img256)
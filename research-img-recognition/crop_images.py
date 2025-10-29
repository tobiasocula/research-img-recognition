import cv2
import numpy as np

def crop_text_and_standardize(image_path, target_size=(256, 256)):
    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to binary (text is black)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the text regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return resized original
    if not contours:
        return cv2.resize(img, target_size)

    # Get bounding box around all contours
    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

    cropped = img[y_min:y_max, x_min:x_max]

    # Compute padding to reach target size while keeping aspect ratio
    h, w = cropped.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas and paste resized text image centered
    canvas = np.ones(target_size, dtype=np.uint8) * 255
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas



from pathlib import Path

this = Path.cwd()
from_dir = this / "input_images_train"
to_dir = this / "input_images_train_cropped"

for dir in from_dir.iterdir():
    cropped = crop_text_and_standardize(dir)
    cv2.imwrite(to_dir / dir.name, cropped)
    print('done with one, moving to', to_dir / dir)


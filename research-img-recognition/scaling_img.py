import cv2
import numpy as np

def scale_text_image(image_path, scale_factor, target_size=(256, 256)):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape

    # Resize text area by scale factor
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas of target size
    canvas = np.ones(target_size, dtype=np.uint8) * 255

    # Compute top-left corner for centering scaled image on canvas
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2

    # Paste scaled image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled_img

    return canvas

from pathlib import Path
import numpy as np

# for images
t = "research-img-recognition"
target = Path.cwd() / t / t / "input_img_cropped_modified"
img_path = Path.cwd() / t / t / "input_images_train_cropped"
for dir in img_path.iterdir():
    c = 0
    img = cv2.imread(dir)
    for sc in np.linspace(0.4, 0.9, 4):
        translated = scale_text_image(img_path / dir.name, sc)
        k = dir.name.split('.')
        cv2.imwrite(target / f"{k[0]}_scaled_{c}.png", translated)
        c += 1
"""
# for bars
t = "research-img-recognition"
target = Path.cwd() / t / t / "bars_img_cropped_modified"
img_path = Path.cwd() / t / t / "bars_targets_train_cropped"
for dir in img_path.iterdir():
    c = 0
    img = cv2.imread(dir)
    for sc in np.linspace(0.4, 0.9, 4):
        translated = scale_text_image(img_path / dir.name, sc)
        k = dir.name.split('.')
        cv2.imwrite(target / f"{k[0]}_scaled_{c}.png", translated)
        c += 1
"""
import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
from pathlib import Path
from loss_and_score_funcs import *

import tensorflow as tf




# Load the trained model
t = "research-img-recognition"
model_path = Path.cwd() / t / t / "models" / "unet_vertical_bars" / "unet_vertical_bars_0.keras"
model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects={
        'bce_dice_loss': bce_dice_loss,
        'resize_with_tf': resize_with_tf
    })

# Function to preprocess the input image
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype("float32") / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Load and preprocess your input image
# input_image_path = Path.cwd() / "research-img-recognition" / "trial_images" / "sample_text.png"
input_image_path = Path.cwd() / t / t / "trial_images" / "sample_text_2.png"
print('input image path:', input_image_path)
target_size = (256, 256)  # Use the same size as during training
preprocessed_image = preprocess_image(input_image_path, target_size)

# Make prediction
pred_mask = model.predict(preprocessed_image)

pred_mask_img = np.squeeze(pred_mask)  # remove batch and channel dims -> shape (H, W)
pred_mask_img = (pred_mask_img * 255).astype(np.uint8)  # scale to [0, 255] uint8

out_to = Path.cwd() / t / t / "output_from_model"
out_to.mkdir(parents=True, exist_ok=True)

print("Mask min/max:", pred_mask.min(), pred_mask.max())

# If needed, process or visualize pred_mask here
print('saving to:', out_to / "trial_image.png")
cv2.imwrite(str(out_to / "trial_image.png"), pred_mask_img)
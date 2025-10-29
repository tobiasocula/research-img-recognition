import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
from pathlib import Path

from loss_and_score_funcs import *

# Load the trained model
model_path = Path.cwd() / "research-img-recognition" / "models" / "unet_vertical_bars" / "unet_vertical_bars_0.keras"
model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects={
        'bce_dice_loss': bce_dice_loss
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
input_image_path = Path.cwd() / "research-img-recognition" / "trial_images" / "sample_text.png"
print('input image path:', input_image_path)
target_size = (256, 256)  # Use the same size as during training
preprocessed_image = preprocess_image(input_image_path, target_size)

# Make prediction
pred_mask = model.predict(preprocessed_image)

out_to = Path.cwd() / "output_from_model"

# If needed, process or visualize pred_mask here
cv2.imwrite(out_to / "trial_image.png", pred_mask)
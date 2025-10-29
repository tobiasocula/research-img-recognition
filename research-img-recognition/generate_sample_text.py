from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def make_text_image(text, width=256, height=256, font_size=48, font_path=None, margin=5):
    image = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(image)
    
    # Load default or fallback font
    font_file = font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    # Dynamically adjust font size to fit text
    current_size = font_size
    while current_size > 1:
        try:
            font = ImageFont.truetype(font_file, current_size)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width + 2 * margin <= width and text_height + 2 * margin <= height:
            break
        current_size -= 1

    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, font=font, fill=0)
    return np.array(image)




save_to = Path.cwd() / "research-img-recognition" / "trial_images"

# Example usage
out = make_text_image("This is some text", width=256, height=256, font_size=48)
Image.fromarray(out).save(str(save_to / "sample_text.png"))

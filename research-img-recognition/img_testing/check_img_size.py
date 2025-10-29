import cv2
from pathlib import Path

path = Path.cwd() / "img_testing" / 'translated_img.png'
img = cv2.imread(path)
h, w, c = img.shape
print(h, w, c)

import base64
import io
from PIL import Image


def decode_base64_image(data):
    header, encoded = data.split(',', 1) if ',' in data else ('', data)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    return img


def save_image(img, path):
    img.save(path, format="PNG")


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

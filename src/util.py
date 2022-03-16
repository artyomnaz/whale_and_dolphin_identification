from PIL import Image
import numpy as np


def pad_and_resize_image(image_path, image_size=128):
    """
    Pad image and resize it to image_size.
    """
    # open image
    image = Image.open(image_path).convert('RGB')
    
    # image shape
    w, h = image.size
    
    # resize with keeping aspect ratio
    if w > h:
        k = h / w
        image = image.resize((image_size, int(image_size * k)))
    else:
        k = w / h
        image = image.resize((int(image_size * k), image_size))
    
    # new image shape
    w, h = image.size
    
    # padded image variable
    image_padded = np.zeros((image_size, image_size, 3))
    
    # pad image
    if w < h:
        start = (image_size - w) // 2
        image_padded[:, start:(start + w), ...] = image
    else:
        start = (image_size - h) // 2
        image_padded[start:(start + h), ...] = image
    
    return image_padded

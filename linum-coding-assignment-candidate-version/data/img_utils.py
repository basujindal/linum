"""
File: img_utils.py
Description: Image utility functions
"""
import numpy as np
from PIL import Image

import torch


def normalize_rgb_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize the values in the img_arr to be between -1 and 1

    Output image is num_channels (3) x h x w or b x (3) x h x w (if input is 4 dimensional)
    """
    norm_img = (img / 127.5 - 1).astype(np.float32)
    if img.ndim == 3:
        norm_img = np.transpose(norm_img, (2, 0, 1))
    elif img.ndim == 4:
        norm_img = np.transpose(norm_img, (0, 3, 1, 2))
    else:
        raise Exception("Input array must be three or four dimensional")
    return norm_img


def read_rgba_img(img_path: str) -> np.ndarray:
    """
        Read RGBA image and convert into an RGB Image as numpy array.

        Output is h x w x num_channels (3)
    """
    rgba_img = (Image.open(img_path)).convert("RGBA")

    # Convert the PIL Image to a NumPy array
    rgba_array = np.array(rgba_img)

    # Separate the RGBA channels
    r = rgba_array[..., 0]
    g = rgba_array[..., 1]
    b = rgba_array[..., 2]
    a = rgba_array[..., 3]

    # Normalize RGB channels with alpha and discard alpha channel
    rgb_array = np.stack([r * (a / 255), g * (a / 255), b * (a / 255)],
                         axis=-1)
    return rgb_array


def convert_img_tensor_to_pil_img(norm_img: torch.Tensor) -> Image:
    """
    Convert an image tensor (c x h x w) with values
    scaled between -1 and 1 to an Image object.
    """
    if norm_img.requires_grad:
        norm_img = norm_img.detach()
    norm_img_arr = norm_img.cpu().numpy()
    return convert_img_ndarray_to_pil_img(norm_img_arr)


def convert_mask_tensor_to_pil_img(mask: torch.Tensor) -> Image:
    """
    Convert binary mask tensor (h x w) into Image (0,255)
    """
    if mask.requires_grad:
        mask = mask.detach()
    mask = mask.cpu().numpy()
    return convert_mask_ndarray_to_pil_img(mask)


def convert_img_ndarray_to_pil_img(norm_img: np.ndarray) -> Image:
    """
    Convert an image numpy array (c x h x w) with values
    scaled between -1 and 1 to an Image object with RGB values
    between 0 and 255.
    """
    unnormalized_img = ((norm_img + 1) * 127.5).astype(np.uint8)
    unnormalized_img = unnormalized_img.transpose(1, 2, 0)
    return Image.fromarray(unnormalized_img)


def convert_mask_ndarray_to_pil_img(mask: np.ndarray) -> Image:
    """
    Convert binary mask numpy array (h x w) into Image (0,255)
    """
    mask = mask.astype(np.uint8)
    return Image.fromarray(mask * 255)

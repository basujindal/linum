"""
File: image_restoration_model.py
Description: Image Restoration Model (U-NET with 1 Encoder and 2 Decoders)
"""
from typing import Tuple
import torch
import torch.nn as nn


class ImageRestorationModel(nn.Module):
    def forward(
            self, corrupted_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Image Restoration Model.

        Given a `corrupted_image` with shape (B, C, H, W) where B = batch size, C = # channels,
        H = image height, W = image width and normalized values between -1 and 1,
        run the Image Restoration Model forward and return a tuple of two tensors:
        (`predicted_image`, `predicted_binary_mask`).

        The `predicted_image` should be the output of the Image Decoder (B, C, H, W). In the
        assignment this is referred to as x^{hat}. This is NOT the `reconstructed_image`,
        referred to as `x_{reconstructed}` in the assignment handout.

        The `predicted_binary_mask` should be the output of the Binary Mask Decoder (B, 1, H, W). This
        is `m^{hat}` in the assignment handout.
        """
        raise NotImplementedError("Please implement forward method")

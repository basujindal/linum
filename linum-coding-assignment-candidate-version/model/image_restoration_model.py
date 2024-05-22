"""
File: image_restoration_model.py
Description: Image Restoration Model (U-NET with 1 Encoder and 2 Decoders)
"""
from typing import Tuple
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, out_c,padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.conv2 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.conv3 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.norm = nn.BatchNorm2d(out_c)
        self.silu = nn.SiLU()

    def forward(self, x):

        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.norm(x) + x
        x = self.silu(x)

        return x

class Encoder(nn.Module):
    def __init__(self, in_c, out_c,  padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c,out_c,3,1,padding)
        self.conv2 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.conv3 = nn.Conv2d(out_c,out_c,3,1,padding)

        self.resBlocks = nn.ModuleList([ResnetBlock(out_c, padding) for i in range(3)])

        self.silu = nn.SiLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.silu(x)

        for block in self.resBlocks:
            x = block(x)

        return self.pool(x), x


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, img_sizes, padding=1):
        super().__init__()

        self.upsample = nn.Upsample(size=img_sizes)


        self.conv1 = nn.Conv2d(in_c,in_c,3,1, padding)
        self.conv2 = nn.Conv2d(in_c,in_c,3,1, padding)
        self.conv3 = nn.Conv2d(in_c,in_c//2,3,1, padding)

        self.conv4 = nn.Conv2d(in_c,out_c,3,1, padding)
        self.conv5 = nn.Conv2d(out_c,out_c,3,1, padding)
        self.conv6 = nn.Conv2d(out_c,out_c,3,1, padding)

        self.resBlocks = nn.ModuleList([ResnetBlock(out_c, padding) for i in range(3)])

        self.silu = nn.SiLU()

    def forward(self, x, skip):

        x = self.upsample(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat((x, skip), dim = 1)
        x = self.conv6(self.conv5(self.conv4(x)))
        x = self.silu(x)

        for block in self.resBlocks:

            x = block(x)

        return x

class ImageRestorationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([Encoder(*i) for i in [(3,32), (32,64), (64, 128), (128, 256), (256, 512)]])

        self.imageDecoders = nn.ModuleList([Decoder(*i) for i in [(512,256, (21,37)), (256,128,(42,74)), (128, 64,(84,149)), (64, 32,(168,298))]])

        self.maskDecoders = nn.ModuleList([Decoder(*i) for i in [(512,256, (21,37)), (256,128,(42,74)), (128, 64,(84,149)), (64, 32,(168,298))]])

        self.imgDecoder = nn.Conv2d(32,3,1,1)
        self.tanh = nn.Tanh()

        self.maskDecoder = nn.Conv2d(32,1,1,1)
        self.sigmoid = nn.Sigmoid()

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
        skips = []
        x = corrupted_image
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = skips[-1]

        for idx, dec in enumerate(self.imageDecoders):
            x = dec(x, skips[3-idx])

        image = self.tanh(self.imgDecoder(x))

        x = skips[-1]

        for idx, dec in enumerate(self.maskDecoders):
            x = dec(x, skips[3-idx])
            
        mask = self.sigmoid(self.maskDecoder(x))
        return image, mask

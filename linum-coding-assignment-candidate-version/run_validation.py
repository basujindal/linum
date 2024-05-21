"""
File: run_validation.py
Description: Load a pretrained model from a PyTorch checkpoint,
generate predictions on the validation set, and save the predicted masks, predicted images,
and reconstructed images to a directory.
"""
import argparse
import glob
import os
import tqdm

import numpy as np
import torch

from data.img_utils import (convert_img_ndarray_to_pil_img,
                            convert_mask_ndarray_to_pil_img, normalize_rgb_img,
                            read_rgba_img)
from model.image_restoration_model import ImageRestorationModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint',
                        type=str,
                        required=True,
                        help='Path to PyTorch model checkpoint')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/validation',
                        help='Data directory')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help=
        'Directory to save predicted masks, predicted images, and fixed images'
    )
    parser.add_argument(
        '--binary-mask-threshold',
        type=float,
        default=0.1,
        help='Threshold for converting the predicted mask to a binary mask. \
        Predicted values greater than or equal to this threshold will get rounded to 1.0 \
        and marked as a corrupted pixel when creating the reconstructed image.'
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------------
    # Load the PyTorch Model
    # -------------------------------------------------------------------------------
    model_checkpoint_state_dict = torch.load(args.model_checkpoint)

    # NOTE: If you have parameters needed to initialize your `ImageRestorationModel`
    # please place them in the kwargs, so that the model can be properly initialized.
    kwargs = {}

    image_restoration_model = ImageRestorationModel(**kwargs)
    image_restoration_model.load_state_dict(
        model_checkpoint_state_dict)s
    image_restoration_model.eval()
    image_restoration_model.to('cuda')

    # -------------------------------------------------------------------------------
    # Generate predictions for validation set
    # -------------------------------------------------------------------------------
    corrupted_img_paths = list(glob.glob(f'{args.data_dir}/corrupted_imgs/*.png'))

    predicted_imgs_out_dir = f'{args.output_dir}/predicted_imgs'
    reconstructed_imgs_out_dir = f'{args.output_dir}/reconstructed_imgs'
    binary_masks_out_dir = f'{args.output_dir}/binary_masks'

    os.makedirs(predicted_imgs_out_dir, exist_ok=True)
    os.makedirs(reconstructed_imgs_out_dir, exist_ok=True)
    os.makedirs(binary_masks_out_dir, exist_ok=True)

    with torch.inference_mode():
        for corrupted_img_path in tqdm.tqdm(corrupted_img_paths):
            # Read the image into memory and normalize values between -1 and 1,
            corrupted_img = read_rgba_img(img_path=corrupted_img_path)
            norm_corrupted_img = torch.from_numpy(
                normalize_rgb_img(img=corrupted_img))
            norm_corrupted_img = norm_corrupted_img.to('cuda')
            norm_corrupted_img = norm_corrupted_img.unsqueeze(0) # The model expects a batch dimension

            # Generate a prediction
            predicted_img, predicted_mask = image_restoration_model(
                norm_corrupted_img)

            # Move to CPU and numpy
            norm_corrupted_img = norm_corrupted_img.detach().cpu().numpy(
            )  # 1 x c x h x w
            predicted_img = predicted_img.detach().cpu().numpy()  # 1 x c x h x w
            predicted_mask = predicted_mask.detach().cpu().numpy()  # 1 x 1 x h x w

            # Convert the predicted mask to a binary mask, based on the threshold
            binary_mask = (predicted_mask >=
                           args.binary_mask_threshold).astype(int)

            # Within the binary mask, the 1s indicate corrupted pixels and 0s indicate
            # uncorrupted pixels. For all the pixels with value 1, copy the "fixed" value
            # from the `predicted_img`. For all the pixels with value 0, copy the "original"
            # from the `norm_corrupted_img`.
            # Match dims of predicted_img (1 x 1 x h x w -> 1 x 3 x h x w)
            expanded_binary_mask = np.repeat(binary_mask, repeats=3, axis=1) 
            reconstructed_img = predicted_img * expanded_binary_mask + (
                1 - expanded_binary_mask) * norm_corrupted_img

            # Let's un-normalize the images and masks
            reconstructed_pil_img = convert_img_ndarray_to_pil_img(
                norm_img=np.squeeze(reconstructed_img, axis=0))
            predicted_pil_img = convert_img_ndarray_to_pil_img(
                norm_img=np.squeeze(predicted_img, axis=0))
            binary_mask_pil_img = convert_mask_ndarray_to_pil_img(
                mask=np.squeeze(binary_mask, axis=(0, 1)))

            # Let's save the images and masks
            filename = os.path.basename(corrupted_img_path)
            reconstructed_pil_img.save(
                os.path.join(reconstructed_imgs_out_dir, filename))
            predicted_pil_img.save(
                os.path.join(predicted_imgs_out_dir, filename))
            binary_mask_pil_img.save(
                os.path.join(binary_masks_out_dir, filename))
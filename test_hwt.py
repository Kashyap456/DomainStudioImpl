from PIL import Image
import torch
import numpy as np
from model.loss import HaarWaveletTransform


def apply_haar_wavelet_transform(image_path):
    # Load the image
    image = Image.open(image_path)
    # Convert the image to a tensor
    image_tensor = torch.tensor(np.array(image)).permute(
        2, 0, 1).unsqueeze(0).float()
    # Initialize the HaarWaveletTransform
    hwt = HaarWaveletTransform()
    # Apply the transform
    transformed_image_tensor = hwt(image_tensor)
    print(transformed_image_tensor.shape)
    # Convert the tensor back to an image
    transformed_image = Image.fromarray(
        transformed_image_tensor.squeeze(0).permute(1, 2, 0).byte().numpy())
    # Show the transformed image
    transformed_image.show()


# Example usage
image_path = './domain_set/55.jpeg'
apply_haar_wavelet_transform(image_path)

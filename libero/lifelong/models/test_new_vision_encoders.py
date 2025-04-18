import numpy as np
import torch
# from r3m_encoder import R3MEncoder  # Assuming you save the class in r3m_encoder.py
from modules.rgb_modules import *

def test_r3m_encoder():
    # Dummy image input: batch of 2 RGB images of size 224x224
    dummy_images = np.random.rand(2, 3, 224, 224).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_images).to("cuda")

    # Initialize encoder
    encoder = R3MEncoder(input_shape=(3, 224, 224), output_size=128).to("cuda")

    # Forward pass
    output = encoder(dummy_tensor)

    print("Input shape:", dummy_tensor.shape)
    print("Output shape:", output.shape)
    print("Sample output:", output[0])


if __name__ == "__main__":
    test_r3m_encoder()

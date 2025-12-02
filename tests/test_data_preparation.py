import numpy as np
import pytest
import os
import sys

# Add the src directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.data_preparation import apply_realesrgan_degradation, get_image_sharpness

@pytest.fixture
def sample_image() -> np.ndarray:
    """
    Creates a simple 100x100 black image for testing purposes.
    
    Returns:
        np.ndarray: A black image of size 100x100x3 with dtype uint8.
    """
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_apply_realesrgan_degradation_smoke(sample_image):
    """
    Smoke test for the apply_realesrgan_degradation function.
    
    This test checks if the function runs without raising an error and returns
    an image of the expected data type. It does not validate the content
    of the degraded image.
    
    Args:
        sample_image (np.ndarray): A pytest fixture providing a sample image.
    """
    print("Running smoke test for apply_realesrgan_degradation...")
    degraded_image = apply_realesrgan_degradation(sample_image, scale=2)
    
    # Check if the output is a numpy array
    assert isinstance(degraded_image, np.ndarray), "Function should return a numpy array."
    
    # Check if the output has the correct data type
    assert degraded_image.dtype == np.uint8, "Returned image should have dtype uint8."
    
    # Check if the output is a 3-channel image
    assert len(degraded_image.shape) == 3 and degraded_image.shape[2] == 3, "Returned image should be a 3-channel image."
    print("Smoke test passed.")

def test_get_image_sharpness_smoke(sample_image):
    """
    Smoke test for the get_image_sharpness function.
    
    This test verifies that the function runs without error and returns a float value,
    which is the expected type for the sharpness score.
    
    Args:
        sample_image (np.ndarray): A pytest fixture providing a sample image.
    """
    print("Running smoke test for get_image_sharpness...")
    sharpness = get_image_sharpness(sample_image)
    
    # Check if the returned value is a float
    assert isinstance(sharpness, float), "Sharpness score should be a float."
    
    # For a black image, sharpness should be zero
    assert sharpness == 0.0, "Sharpness for a completely black image should be 0."
    print("Smoke test passed.")

if __name__ == '__main__':
    """
    Allows running tests directly for debugging.
    
    Example:
    python tests/test_data_preparation.py
    """
    pytest.main([__file__])

import argparse
import os
import sys
import cv2
import shutil
import yaml
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.data_preparation import apply_realesrgan_degradation, get_image_sharpness

def main():
    """
    Main function to prepare the image dataset.
    This script reads configuration from a YAML file to process high-resolution images,
    creating a dataset with paired low-resolution (degraded) and high-resolution images.
    """
    parser = argparse.ArgumentParser(description="Prepare dataset for image enhancement training using a config file.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    hr_dir = os.path.abspath(data_config['hr_dir'])
    dataset_dir = os.path.abspath(data_config['dataset_dir'])
    sharpness_threshold = data_config['sharpness_threshold']
    
    # Define output directories
    train_dir = os.path.join(dataset_dir, "train", "1_")
    reg_dir = os.path.join(dataset_dir, "reg", "1_")
    blur_dir = os.path.join(hr_dir, "blur") # Store blurry images separately

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(blur_dir, exist_ok=True)

    # Get list of PNG files
    file_list = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(hr_dir, f))]
    
    if not file_list:
        print(f"No images found in {hr_dir}. Aborting.")
        return

    print(f"Starting dataset preparation for {len(file_list)} images in {hr_dir}...")

    for filename in tqdm(file_list, desc="Preparing Dataset"):
        hr_image_path = os.path.join(hr_dir, filename)
        
        hr_image = cv2.imread(hr_image_path)
        if hr_image is None:
            print(f"\nWarning: Could not read image {hr_image_path}. Skipping.")
            continue

        # Calculate image sharpness
        sharpness = get_image_sharpness(hr_image)
        
        # If the image is sharp enough, create a degraded version and move both to the dataset folder.
        # Otherwise, move the blurry image to a separate folder.
        if sharpness > sharpness_threshold:
            lr_image = apply_realesrgan_degradation(hr_image)
            cv2.imwrite(os.path.join(reg_dir, filename), lr_image)
            shutil.move(hr_image_path, os.path.join(train_dir, filename))
        else:
            shutil.move(hr_image_path, os.path.join(blur_dir, filename))

    print("\nDataset preparation complete.")
    print(f" - Training HR images moved to: {train_dir}")
    print(f" - Training LR images created in: {reg_dir}")
    print(f" - Blurry images moved to: {blur_dir}")


if __name__ == "__main__":
    main()

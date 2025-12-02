import argparse
import os
import sys
import yaml
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.inference import run_img2img_repair

def main():
    """
    Main function to run the image-to-image repair process on a directory of images.
    This script reads all its configuration from a YAML file.
    """
    parser = argparse.ArgumentParser(description="Run batch img2img repair on a directory using a config file.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    infer_config = config['inference']

    # Get parameters from config
    base_model_path = infer_config['base_model_path']
    lora_path = infer_config['lora_path']
    input_dir = infer_config['input_dir']
    output_dir = infer_config['output_dir']
    prompt = infer_config['prompt']
    strength = infer_config['strength']
    cfg_scale = infer_config['cfg_scale']
    steps = infer_config['steps']
    seed = infer_config['seed']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images to process
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_list:
        print(f"No images found in {input_dir}. Aborting.")
        return

    print(f"Starting batch inference for {len(image_list)} images...")
    
    # Process each image in the directory
    for filename in tqdm(image_list, desc="Running Inference"):
        input_image_path = os.path.join(input_dir, filename)
        
        run_img2img_repair(
            base_model_path=base_model_path,
            lora_path=lora_path,
            input_image_path=input_image_path,
            output_dir=output_dir,
            prompt=prompt,
            strength=strength,
            cfg_scale=cfg_scale,
            steps=steps,
            seed=seed
        )

    print("\nBatch inference complete.")
    print(f"Repaired images saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.val import run_validation

def main():
    """
    Command-line interface for running the full validation pipeline.
    This script handles inference on a validation set and subsequent evaluation,
    reading all parameters from a YAML config file.
    """
    parser = argparse.ArgumentParser(
        description="Run a full validation pipeline (inference + evaluation) using a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    val_config = config['validation_pipeline']

    # Create output directories if they don't exist
    os.makedirs(val_config['generate_folder'], exist_ok=True)
    os.makedirs(val_config['output_dir'], exist_ok=True)
    
    print("Starting validation script...")
    try:
        run_validation(
            base_model_path=val_config['base_model_path'],
            lora_path=val_config['lora_path'],
            hr_folder=val_config['hr_folder'],
            lr_folder=val_config['lr_folder'],
            generate_folder=val_config['generate_folder'],
            output_dir=val_config['output_dir'],
            prompt=val_config['prompt'],
            strength=val_config['strength'],
            cfg_scale=val_config['cfg_scale'],
            steps=val_config['steps'],
            seed=val_config['seed']
        )
    except KeyError as e:
        print(f"Error: Missing required key in config file's 'validation_pipeline' section: {e}")
        sys.exit(1)

    print("Validation script finished.")

if __name__ == "__main__":
    main()

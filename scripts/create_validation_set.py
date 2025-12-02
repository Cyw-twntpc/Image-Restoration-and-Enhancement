import argparse
import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.val_dataset import create_validation_set

def main():
    """
    Command-line interface for creating a validation set from the training data.
    Reads configuration from a YAML file.
    """
    parser = argparse.ArgumentParser(
        description="Create a validation set by sampling from the training data, based on a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    val_config = config['validation_set']

    try:
        create_validation_set(
            train_dir=val_config['train_dir'],
            reg_dir=val_config['reg_dir'],
            hr_dir=val_config['hr_dir'],
            lr_dir=val_config['lr_dir'],
            num_val_files=val_config['num_val_files']
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required key in config file's 'validation_set' section: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import os
import random
import shutil
from tqdm import tqdm

def create_validation_set(train_dir: str, reg_dir: str, hr_dir: str, lr_dir: str, num_val_files: int = 1000):
    """
    Creates a validation set by randomly sampling and moving files from the training set directories.

    Args:
        train_dir (str): Directory with the high-resolution training images (e.g., './dataset/train/1_').
        reg_dir (str): Directory with the corresponding low-resolution training images (e.g., './dataset/reg/1_').
        hr_dir (str): Destination directory for the high-resolution validation images (e.g., './val/hr').
        lr_dir (str): Destination directory for the low-resolution validation images (e.g., './val/lr').
        num_val_files (int): The number of files to include in the validation set.
    """
    # Ensure all paths are absolute
    train_dir = os.path.abspath(train_dir)
    reg_dir = os.path.abspath(reg_dir)
    hr_dir = os.path.abspath(hr_dir)
    lr_dir = os.path.abspath(lr_dir)

    # Create destination directories if they don't exist
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # Check if source directories exist
    if not os.path.isdir(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return
    if not os.path.isdir(reg_dir):
        print(f"Error: Registration directory not found at {reg_dir}")
        return

    train_files = os.listdir(train_dir)
    
    # Ensure there are enough files to create the validation set
    if len(train_files) < num_val_files:
        print(f"Warning: Requested {num_val_files} validation files, but only {len(train_files)} are available in {train_dir}. Using all available files.")
        num_val_files = len(train_files)

    # Randomly select files for the validation set
    val_files = random.sample(train_files, num_val_files)

    print(f"Moving {len(val_files)} files from training to validation directories...")
    
    # Move the selected files
    for file in tqdm(val_files, desc="Creating Validation Set"):
        # Source paths
        train_hr_path = os.path.join(train_dir, file)
        train_lr_path = os.path.join(reg_dir, file)
        
        # Destination paths
        val_hr_path = os.path.join(hr_dir, file)
        val_lr_path = os.path.join(lr_dir, file)

        # Move HR file (from train to val)
        if os.path.exists(train_hr_path):
            shutil.move(train_hr_path, val_hr_path)
        else:
            print(f"Warning: HR file not found: {train_hr_path}")

        # Move LR file (from reg to val)
        if os.path.exists(train_lr_path):
            shutil.move(train_lr_path, val_lr_path)
        else:
            print(f"Warning: LR file not found: {train_lr_path}")

    print("\nValidation set created successfully.")
    print(f" - HR images moved to: {hr_dir}")
    print(f" - LR images moved to: {lr_dir}")

# Project: AI-Powered Image Restoration and Enhancement Pipeline

A comprehensive system for training, evaluating, and deploying custom AI models to restore low-quality images to high-resolution, realistic counterparts. This project is structured as a professional MLOps pipeline, emphasizing modularity, configurability, and reproducibility.

## 1. Tech Stack

*   **AI Framework:** PyTorch
*   **Core Libraries:** Hugging Face (Diffusers, Transformers, Accelerate) for Stable Diffusion models.
*   **Training & Fine-tuning:** Kohya's SS scripts for LoRA (Low-Rank Adaptation) training.
*   **Image Processing:** OpenCV, Pillow, Scikit-image for data augmentation and processing.
*   **Evaluation Metrics:** LPIPS, Torch-Fidelity (for FID), Scikit-image (for PSNR, SSIM).
*   **Configuration:** PyYAML for centralized pipeline configuration.

## 2. Project Structure

The project follows a standardized structure to separate concerns and improve maintainability.

```
photo_enhancer/
├── configs/
│   └── config.yaml         # Central configuration for all scripts
├── data/
│   ├── hr/                 # Source high-resolution images
│   └── val/                # Validation set (HR and LR pairs)
├── models/
│   ├── base/               # Base Stable Diffusion models
│   └── lora/               # Trained LoRA models
├── outputs/
│   ├── evaluation_results/ # Evaluation reports and charts
│   ├── inference_results/  # Images repaired by run_inference.py
│   └── val/                # Outputs from the validation pipeline
├── scripts/                # Standalone scripts to run pipeline stages
│   ├── prepare_dataset.py
│   ├── create_validation_set.py
│   ├── run_inference.py
│   ├── run_evaluation.py
│   └── run_validation.py
├── src/
│   └── photo_enhancer/     # Core Python source code as an installable package
│       ├── __init__.py
│       ├── data_preparation.py
│       ├── evaluation.py
│       └── inference.py
├── tools/
│   └── kohya_ss/           # External training toolchain
├── .gitignore
├── README.md
└── requirements.txt
```

## 3. Configuration

All pipeline parameters are managed in a central configuration file: `configs/config.yaml`. This approach allows you to define and track the settings for data processing, model paths, inference parameters, and evaluation targets without modifying the source code.

Before running any scripts, please review and update `configs/config.yaml` to match your local paths and desired parameters.

## 4. How to Run the Pipeline

This pipeline is designed to be run sequentially using the scripts in the `scripts/` directory. Each script reads its parameters from `configs/config.yaml`.

### Step 1: Setup
Install the required Python packages.
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data
Place your high-resolution source images in the directory specified by `data.hr_dir` in your `config.yaml` (default: `./data/hr`).

### Step 3: Generate Training Dataset
Run the data preparation script. It will apply a realistic degradation pipeline to your HR images, creating the low-resolution (LR) and corresponding high-resolution (HR) pairs required for training.
```bash
python scripts/prepare_dataset.py
```
The output will be saved in the `data.dataset_dir` (default: `./data/dataset`).

### Step 4: Train the LoRA Model
This project uses **Kohya's SS**, located in the `tools/kohya_ss` directory, for training.

1.  Launch the Gradio Web UI.
    ```bash
    # Inside the tools/kohya_ss directory
    ./gui.bat # On Windows
    ./gui.sh  # On Linux
    ```
2.  In the Kohya GUI, configure the training run. Point the `Image folder` to the directory created in Step 3 (e.g., `D:\python_project\photo_enhancer\data\dataset`).
3.  Set your model paths, training parameters, and start the training.
4.  Once complete, move your trained LoRA model (`.safetensors` file) to the `models.lora` directory.

### Step 5: Run Inference
To repair a set of images, update the `inference` section in your `config.yaml` with the correct model and data paths. Then, run the inference script:
```bash
python scripts/run_inference.py
```
The restored images will be saved in the `inference.output_dir`.

### Step 6: Evaluate Performance
To quantitatively measure the performance of your model, update the `evaluation` section in the config and run the evaluation script. This compares the restored images against the originals.
```bash
python scripts/run_evaluation.py
```
A summary report and individual metric charts will be saved in `evaluation.output_dir`.

### (Optional) Full Validation Pipeline
To run inference and evaluation in a single step, configure the `validation_pipeline` section and execute:
```bash
python scripts/run_validation.py
```

### Step 7: Testing
This project includes a test suite to ensure code quality and stability. The tests are located in the `/tests` directory and use the `pytest` framework.

To run the tests, execute the following command from the project root directory:
```bash
python -m pytest
```

## 5. Technical Highlights

### Realistic Training Data
The core technical challenge is creating training data that mirrors real-world image degradation. The `apply_realesrgan_degradation` function in **`src/photo_enhancer/data_preparation.py`** implements a sophisticated, multi-stage degradation pipeline (simulating blur, noise, JPEG artifacts) inspired by the Real-ESRGAN paper. This ensures the model learns robust and generalized restoration capabilities.

### Integrated Evaluation Loop
The project features a tight training and evaluation loop. Scripts like **`scripts/run_evaluation.py`** and **`scripts/run_validation.py`** provide immediate, quantitative feedback (PSNR, SSIM, LPIPS, FID) on model performance, enabling rapid and efficient iteration.

## 6. Future Work

*   **Hyperparameter Optimization:** Automate the process of tuning the degradation pipeline and model training parameters using techniques like Optuna or Ray Tune.
*   **Web Interface:** Develop a user-friendly Gradio or FastAPI web application to allow users to easily upload an image and receive the restored version.
*   **Model Expansion:** Train models to handle a wider variety of degradations, such as text watermarks, scratches, or color fading.
*   **Performance Optimization:** Investigate model quantization (e.g., 8-bit) and optimization engines like ONNX Runtime or TensorRT to accelerate inference speed.
*   **CI/CD Automation:** Implement a GitHub Actions workflow to automate linting, testing, and even trigger validation runs on a small dataset.

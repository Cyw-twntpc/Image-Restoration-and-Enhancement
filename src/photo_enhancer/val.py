import os
import shutil
import tqdm
from .evaluation import ImageEvaluator
from .inference import run_img2img_repair

def run_validation(
    base_model_path: str,
    lora_path: str,
    hr_folder: str,
    lr_folder: str,
    generate_folder: str,
    output_dir: str,
    prompt: str = "",
    strength: float = 0.2,
    cfg_scale: float = 7.0,
    steps: int = 28,
    seed: int = -1
):
    """
    Runs a full validation pipeline:
    1. Generates repaired images from a low-resolution folder using a specified model and LoRA.
    2. Evaluates the generated images against a high-resolution ground truth folder.

    Args:
        base_model_path (str): Path to the base Stable Diffusion model.
        lora_path (str): Path to the trained LoRA model.
        hr_folder (str): Path to the directory with high-resolution ground truth images.
        lr_folder (str): Path to the directory with low-resolution images to be processed.
        generate_folder (str): Directory to save the generated/repaired images.
        output_dir (str): Directory to save the evaluation results and plots.
        prompt (str): Text prompt for the img2img process.
        strength (float): Denoising strength for inference.
        cfg_scale (float): CFG scale for inference.
        steps (int): Number of inference steps.
        seed (int): Random seed for inference.
    """
    # Absolutize paths
    hr_folder = os.path.abspath(hr_folder)
    lr_folder = os.path.abspath(lr_folder)
    generate_folder = os.path.abspath(generate_folder)
    base_model_path = os.path.abspath(base_model_path)
    lora_path = os.path.abspath(lora_path)

    # Clean and create the generation folder
    if os.path.exists(generate_folder):
        shutil.rmtree(generate_folder)
    os.makedirs(generate_folder)
    print(f"Cleaned and created folder: {generate_folder}")

    # Run inference on all images in the low-resolution folder
    img_list = [f for f in os.listdir(lr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in tqdm.tqdm(img_list, desc="Running Inference", unit="image"):
        img_path = os.path.join(lr_folder, img_name)
        run_img2img_repair(
            base_model_path=base_model_path,
            lora_path=lora_path,
            input_image_path=img_path,
            output_dir=generate_folder,
            prompt=prompt,
            strength=strength,
            cfg_scale=cfg_scale,
            steps=steps,
            seed=seed
        )

    # Run evaluation
    print("\nStarting evaluation of generated images...")
    evaluator = ImageEvaluator(
        ground_truth_dir=hr_folder, 
        generated_dir=generate_folder, 
        output_dir=output_dir
    )
    psnr_score, _ = evaluator.evaluate_psnr()
    ssim_score, _ = evaluator.evaluate_ssim()
    lpips_score, _ = evaluator.evaluate_lpips()
    fid_score = evaluator.evaluate_fid()
    isc_mean, isc_std = evaluator.evaluate_isc()

    # Print and save summary
    summary = f"""
    =================================================
    =              Validation Summary               =
    =================================================
    Base Model: {os.path.basename(base_model_path)}
    LoRA Model: {os.path.basename(lora_path)}
    -------------------------------------------------
    PSNR: {psnr_score:.4f} dB
    SSIM: {ssim_score:.4f}
    LPIPS: {lpips_score:.4f}
    FID: {fid_score:.4f}
    Inception Score: {isc_mean:.4f} ± {isc_std:.4f}
    =================================================
    """
    print(summary)
    
    summary_path = os.path.join(output_dir, "validation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Validation summary saved to {summary_path}")

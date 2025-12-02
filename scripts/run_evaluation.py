import argparse
import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.evaluation import ImageEvaluator

def main():
    """
    Main function to run the evaluation of generated images against ground truth images.
    This script computes several standard image quality metrics based on a YAML config file.
    """
    parser = argparse.ArgumentParser(description="Evaluate the quality of repaired images using a config file.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    eval_config = config['evaluation']
    ground_truth_dir = eval_config['ground_truth_dir']
    generated_dir = eval_config['generated_dir']
    output_dir = eval_config['output_dir']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing evaluator...")
    # Create an instance of the evaluator
    evaluator = ImageEvaluator(
        ground_truth_dir=ground_truth_dir,
        generated_dir=generated_dir,
        output_dir=output_dir
    )

    if not evaluator.image_pairs:
        print("No matching image pairs found. Aborting evaluation.")
        return
        
    print("\nStarting full evaluation...")

    # --- Run individual evaluations ---
    avg_psnr, _ = evaluator.evaluate_psnr()
    avg_ssim, _ = evaluator.evaluate_ssim()
    avg_lpips, _ = evaluator.evaluate_lpips()
    fid_score = evaluator.evaluate_fid()
    isc_mean, isc_std = evaluator.evaluate_isc()

    # --- Print a final summary report ---
    summary = f"""
    =================================================
    =               Evaluation Summary              =
    =================================================
    Directories:
    - Ground Truth: {os.path.abspath(ground_truth_dir)}
    - Generated:    {os.path.abspath(generated_dir)}

    Metrics (Higher is better for PSNR, SSIM, IS; Lower for LPIPS, FID):
    -------------------------------------------------
    - Average PSNR:         {avg_psnr:.4f} dB
    - Average SSIM:         {avg_ssim:.4f}
    - Average LPIPS:        {avg_lpips:.4f}
    - FID Score:            {fid_score:.4f}
    - Inception Score:      {isc_mean:.4f} ± {isc_std:.4f}
    -------------------------------------------------
    
    Individual metric plots have been saved to:
    {os.path.abspath(output_dir)}
    =================================================
    """
    print(summary)
    
    # Save the summary to a text file
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary report saved to {summary_path}")

if __name__ == "__main__":
    main()

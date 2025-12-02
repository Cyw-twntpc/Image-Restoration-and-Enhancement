"""
This module provides a comprehensive `ImageEvaluator` class for assessing the
quality of image restoration models. It calculates a suite of standard metrics
to provide a quantitative analysis of model performance.
"""
import os
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
from torch_fidelity import calculate_metrics

class ImageEvaluator:
    """
    An integrated calculator for evaluating the quality of restored images.

    This class compares a directory of generated images against a directory of
    ground truth images, calculating the following metrics:
    - PSNR (Peak Signal-to-Noise Ratio): Measures pixel-wise reconstruction accuracy.
    - SSIM (Structural Similarity Index): Measures perceptual similarity.
    - LPIPS (Learned Perceptual Image Patch Similarity): Measures perceptual
      similarity using a trained deep learning model.
    - FID (Fréchet Inception Distance): Measures the perceptual distance between
      the distribution of generated images and real images.
    - IS (Inception Score): Measures the quality and diversity of generated images.

    Attributes:
        gt_dir (str): Directory for ground truth images.
        gen_dir (str): Directory for generated (restored) images.
        output_dir (str): Directory to save evaluation plots and reports.
        device (str): The torch device to use for calculations ('cuda' or 'cpu').
        image_pairs (List[str]): A list of common filenames found in both directories.
    """
    def __init__(self, ground_truth_dir: str, generated_dir: str, output_dir: str = "./evaluation_results"):
        """
        Initializes the ImageEvaluator.

        Args:
            ground_truth_dir (str): Path to the directory containing the original,
                high-resolution ground truth images.
            generated_dir (str): Path to the directory containing the AI-restored images.
            output_dir (str, optional): Path to the directory where evaluation results
                (plots, reports) will be saved. Defaults to "./evaluation_results".
        """
        print("Initializing ImageEvaluator...")
        self.gt_dir = ground_truth_dir
        self.gen_dir = generated_dir
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.image_pairs = self._get_image_pairs()
        
        self.results: Dict[str, Any] = {"per_image": {}}
        self._per_image_metrics_calculated = False
        print(f"Found {len(self.image_pairs)} matching image pairs for evaluation.")

    def _get_image_pairs(self) -> List[str]:
        """
        Finds common image files between the ground truth and generated directories.

        Returns:
            List[str]: A sorted list of filenames that exist in both directories.
        """
        try:
            gt_files = set(f for f in os.listdir(self.gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            gen_files = set(f for f in os.listdir(self.gen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            common_files = sorted(list(gt_files.intersection(gen_files)))
            if not common_files:
                print("Warning: No matching image files found between the two directories.")
            elif len(common_files) < len(gt_files):
                print("Warning: Some ground truth images do not have a corresponding generated image.")
            return common_files
        except FileNotFoundError as e:
            print(f"Error: Directory not found - {e}")
            return []

    def _calculate_all_per_image_metrics(self):
        """
        Internal method to calculate all one-to-one metrics (PSNR, SSIM, LPIPS)
        in a single pass to avoid redundant image loading. Results are cached.
        """
        if self._per_image_metrics_calculated or not self.image_pairs:
            return

        print("\n--- Calculating all per-image metrics (PSNR, SSIM, LPIPS) in one pass ---")
        for filename in tqdm(self.image_pairs, desc="Evaluating Per-Image Metrics"):
            gt_path = os.path.join(self.gt_dir, filename)
            gen_path = os.path.join(self.gen_dir, filename)
            
            try:
                img_gt_np = np.array(Image.open(gt_path).convert('RGB')) / 255.0
                img_gen_np = np.array(Image.open(gen_path).convert('RGB')) / 255.0

                # Ensure shapes match for comparison
                if img_gt_np.shape != img_gen_np.shape:
                    h, w, _ = img_gt_np.shape
                    img_gen_np = resize(img_gen_np, (h, w), anti_aliasing=True)

                # PSNR
                psnr_score = psnr(img_gt_np, img_gen_np, data_range=1.0)
                
                # SSIM (with dynamic window size for small images)
                min_dim = min(img_gt_np.shape[0], img_gt_np.shape[1])
                win_size = min(7, min_dim - (min_dim % 2 - 1)) # Ensure odd and <= min_dim
                ssim_score = ssim(img_gt_np, img_gen_np, data_range=1.0, channel_axis=-1, win_size=win_size)

                # LPIPS
                img_gt_t = torch.from_numpy(img_gt_np).permute(2, 0, 1).float().unsqueeze(0) * 2 - 1
                img_gen_t = torch.from_numpy(img_gen_np).permute(2, 0, 1).float().unsqueeze(0) * 2 - 1
                lpips_score = self.lpips_model(img_gt_t.to(self.device), img_gen_t.to(self.device)).item()

                self.results["per_image"][filename] = {"psnr": psnr_score, "ssim": ssim_score, "lpips": lpips_score}
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        self._per_image_metrics_calculated = True

    def _plot_and_get_path(self, metric_name: str, higher_is_better: bool) -> str:
        """
        Internal method to plot the distribution of a metric and save it to a file.

        Args:
            metric_name (str): The name of the metric to plot (e.g., "PSNR").
            higher_is_better (bool): Determines the color of the bar chart.

        Returns:
            str: The file path to the saved plot, or an empty string if no scores exist.
        """
        scores = {fname: self.results["per_image"][fname][metric_name.lower()] for fname in self.image_pairs if metric_name.lower() in self.results["per_image"].get(fname, {})}
        if not scores: return ""
        
        filenames, score_values = list(scores.keys()), list(scores.values())
        plt.figure(figsize=(max(15, len(filenames) * 0.3), 8))
        plt.bar(filenames, score_values, color='deepskyblue' if higher_is_better else 'salmon')
        plt.ylabel('Score')
        plt.title(f'{metric_name} Score Distribution')
        plt.xticks(rotation=90, fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f"{metric_name}_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"{metric_name} distribution plot saved to {plot_path}")
        return plot_path

    # --- Public Evaluation Methods ---

    def evaluate_psnr(self) -> Tuple[float, str]:
        """Calculates the average PSNR across all image pairs."""
        self._calculate_all_per_image_metrics()
        scores = [v['psnr'] for v in self.results["per_image"].values()]
        if not scores: return 0.0, ""
        average_score = float(np.mean(scores))
        chart_path = self._plot_and_get_path("PSNR", higher_is_better=True)
        print(f"Average PSNR: {average_score:.4f} dB")
        return average_score, chart_path

    def evaluate_ssim(self) -> Tuple[float, str]:
        """Calculates the average SSIM across all image pairs."""
        self._calculate_all_per_image_metrics()
        scores = [v['ssim'] for v in self.results["per_image"].values()]
        if not scores: return 0.0, ""
        average_score = float(np.mean(scores))
        chart_path = self._plot_and_get_path("SSIM", higher_is_better=True)
        print(f"Average SSIM: {average_score:.4f}")
        return average_score, chart_path

    def evaluate_lpips(self) -> Tuple[float, str]:
        """Calculates the average LPIPS across all image pairs."""
        self._calculate_all_per_image_metrics()
        scores = [v['lpips'] for v in self.results["per_image"].values()]
        if not scores: return 0.0, ""
        average_score = float(np.mean(scores))
        chart_path = self._plot_and_get_path("LPIPS", higher_is_better=False)
        print(f"Average LPIPS: {average_score:.4f}")
        return average_score, chart_path

    def evaluate_fid(self) -> float:
        """Calculates the Fréchet Inception Distance (FID) between the two image sets."""
        print("\n--- Starting FID Evaluation ---")
        try:
            metrics = calculate_metrics(self.gen_dir, self.gt_dir, cuda=(self.device=='cuda'), isc=False, fid=True, kid=False, verbose=False)
            fid_score = metrics['frechet_inception_distance']
            print(f"FID Score: {fid_score:.4f} (Lower is better)")
            return fid_score
        except Exception as e:
            print(f"An error occurred during FID calculation: {e}")
            return -1.0

    def evaluate_isc(self) -> Tuple[float, float]:
        """Calculates the Inception Score (IS) for the generated image set."""
        print("\n--- Starting Inception Score Evaluation ---")
        try:
            metrics = calculate_metrics(self.gen_dir, cuda=(self.device=='cuda'), isc=True, fid=False, kid=False, verbose=False)
            isc_mean, isc_std = metrics['inception_score_mean'], metrics['inception_score_std']
            print(f"Inception Score: {isc_mean:.4f} ± {isc_std:.4f} (Higher is better)")
            return isc_mean, isc_std
        except Exception as e:
            print(f"An error occurred during Inception Score calculation: {e}")
            return -1.0, -1.0
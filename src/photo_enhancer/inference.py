"""
This module contains the core logic for running image-to-image inference
using a pre-trained Stable Diffusion model and a fine-tuned LoRA.
"""
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from safetensors.torch import load_file
from typing import Optional

def run_img2img_repair(
    base_model_path: str,
    lora_path: str,
    input_image_path: str,
    output_dir: str = "./repaired_images",
    prompt: str = "",
    strength: float = 0.2,
    cfg_scale: float = 7.0,
    steps: int = 28,
    seed: int = -1
) -> Optional[str]:
    """
    Loads a base model and a LoRA to perform an image-to-image restoration task.

    This function initializes a Stable Diffusion Img2Img pipeline, loads the base
    model weights, and then injects the fine-tuned LoRA weights. A critical
    step is the manual re-mapping of the LoRA state dictionary keys, which is
    necessary to ensure compatibility between the format produced by the Kohya's SS
    training scripts (`lora_unet_*`) and the format expected by the Diffusers
    library (`unet.*`).

    Args:
        base_model_path (str): The file path to the base Stable Diffusion model
            in .safetensors format.
        lora_path (str): The file path to the fine-tuned LoRA model in
            .safetensors format.
        input_image_path (str): The path to the low-quality input image that
            needs to be repaired.
        output_dir (str, optional): The directory where the repaired image will
            be saved. Defaults to "./repaired_images".
        prompt (str, optional): The text prompt to guide the diffusion process.
            Defaults to "".
        strength (float, optional): The denoising strength. A lower value preserves
            more of the original image structure. Defaults to 0.2.
        cfg_scale (float, optional): Classifier-Free Guidance scale. Controls how
            closely the model follows the prompt. Defaults to 7.0.
        steps (int, optional): The number of inference steps. Defaults to 28.
        seed (int, optional): The random seed for reproducibility. A value of -1
            indicates a random seed will be used. Defaults to -1.

    Returns:
        Optional[str]: The full path to the saved output image if successful,
                       otherwise None.
    """
    print("\n--- Starting Image Repair Inference ---")
    try:
        # Determine device and data type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Using device: {device}, Data type: {torch_dtype}")

        # Load the base model pipeline
        print(f"Loading base model from: {base_model_path}")
        pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
            base_model_path,
            torch_dtype=torch_dtype,
            variant="fp16",
            use_safetensors=True
        ).to(device)

        # Load LoRA weights and manually fix state dict keys for Diffusers compatibility
        print(f"Loading LoRA weights from: {lora_path}")
        lora_state_dict = load_file(lora_path, device=device)
        
        # This conversion is crucial. Kohya's SS saves UNet LoRA weights with the
        # prefix "lora_unet_...". The diffusers library expects them to be part
        # of the UNet's state dict, so we remap them to "unet...".
        # We only load UNet weights as the text encoder's role is minimal in this
        # restoration task, ensuring stability.
        unet_state_dict = {}
        for k, v in lora_state_dict.items():
            if "lora_unet_" in k:
                unet_state_dict[k.replace("lora_unet_", "unet.")] = v
        
        pipeline.load_lora_weights(unet_state_dict)
        print("Successfully loaded and remapped LoRA UNet weights.")

        # Prepare input image
        print(f"Reading input image: {input_image_path}")
        init_image = Image.open(input_image_path).convert("RGB")

        # Set up generator for reproducibility
        generator = torch.Generator(device=device)
        if seed != -1:
            generator.manual_seed(seed)

        # Run the img2img pipeline
        print(f"Generating image... Strength: {strength}, Steps: {steps}, CFG: {cfg_scale}")
        output_image = pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator
        ).images[0]

        # Save the result
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.basename(input_image_path)
        output_path = os.path.join(output_dir, output_filename)
        
        output_image.save(output_path)
        print(f"Inference complete. Image saved to: {output_path}")

        return output_path

    except FileNotFoundError as e:
        print(f"Error: Model or image file not found. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during inference: {e}")
        return None
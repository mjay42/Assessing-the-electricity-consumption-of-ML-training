"""Inference from stable diffusion.

Example of use:
    python diffusion_pipeline.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-1" \
        --prompt_csv="/home/mjay/ai-energy-consumption-framework/stable-diffusion/prompts/processed_prompts.csv" \
        --prompt_line=10 \
        --seed=42 \
        --negative_prompt="red" \
        --num_steps=25 \
        --img_size=256
"""

from diffusers import DiffusionPipeline
import argparse
import time
import torch   
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt_line",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--nb_image",
        type=int,
        default=1,
    )
    return parser.parse_args()
    
def main():
    args = parse_args()
 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
    )
    pipeline.to("cuda")
    # reproducibility : If reproducibility is important, we recommend always passing a CPU generator. The performance loss is often neglectable, and you’ll generate much more similar values than if the pipeline had been run on a GPU.
    # https://huggingface.co/docs/diffusers/v0.19.3/en/using-diffusers/reproducibility
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    
    if not args.prompt is None:
        prompt=args.prompt
    elif not args.prompt_line is None and not args.prompt_csv is None :
        prompt_df = pd.read_csv(args.prompt_csv).drop("Unnamed: 0", axis=1)
        prompt = prompt_df.values[args.prompt_line][0]
        print("prompt: "+prompt)
    else:
        return "Arguments missing."

    # list of parameters available: https://huggingface.co/docs/diffusers/v0.19.3/en/api/pipelines/stable_diffusion/text2img
    start = time.time()
    print(f"Starting inference at {start}")
    pipe = pipeline(
        prompt = prompt,
        negative_prompt = args.negative_prompt,
        num_inference_steps=args.num_steps,
        generator=generator,
        height=args.img_size,
        width=args.img_size,
        # guidance_rescale=0,
        num_images_per_prompt=args.nb_image,
        )
    images = pipe.images
    end = time.time()
    print(f"Ending inference at {end}") 
    # images[0].save(f"/home/mjay/laion/pokemon/tests/{args.prompt}.png")
    
if __name__ == "__main__":
    main()
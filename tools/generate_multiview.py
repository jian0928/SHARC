import argparse
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler

from sharc.models.synthesizer import CrossDetailMultiPerspectiveSynthesizer
from sharc.sampling import DDIMSampler
from sharc.utils.io import load_image, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--vae_path', type=str, default=None)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder='vae').to(args.device)
    scheduler = DDIMScheduler.from_pretrained(args.vae_path, subfolder='scheduler')
    synthesizer = CrossDetailMultiPerspectiveSynthesizer()
    checkpoint = torch.load(args.ckpt)
    synthesizer.load_state_dict(checkpoint['model'])
    synthesizer.eval().to(args.device)
    synthesizer.conditioning.vae = vae

    image = load_image(args.image).unsqueeze(0).to(args.device)
    condition_global = synthesizer.conditioning.build_condition(image=image)
    latent_h = args.image_size // 8
    latent_w = args.image_size // 8
    sampler = DDIMSampler(branch=synthesizer.global_branch, scheduler=scheduler, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, device=args.device)
    latents = sampler.sample(shape=(1, 4, latent_h, latent_w), condition_latents=condition_global)
    image_out = synthesizer.conditioning.decode(latents)
    os.makedirs(args.out, exist_ok=True)
    save_image(os.path.join(args.out, 'generated.png'), image_out[0])


if __name__ == '__main__':
    main()

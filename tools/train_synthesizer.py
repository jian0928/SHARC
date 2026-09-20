import argparse
import importlib
import os

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from sharc.data import FaceCrop, MultiviewDataset
from sharc.models.synthesizer import CrossDetailMultiPerspectiveSynthesizer
from sharc.models.synthesizer.losses import JointDenoisingLoss
from sharc.pipelines import SynthesizerTrainer
from sharc.utils.seed import set_seed


def load_dataset(args):
    module_name, class_name = args.dataset_module.rsplit('.', 1)
    module = importlib.import_module(module_name)
    dataset_cls = getattr(module, class_name)
    base = dataset_cls(root=args.dataset_root, split=args.split)
    face_crop = FaceCrop(crop_size=args.face_crop_size) if args.face_crop_size is not None else None
    return MultiviewDataset(base_dataset=base, multiview_angles=args.angles, face_crop=face_crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_module', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--angles', type=float, nargs='+', required=True)
    parser.add_argument('--face_crop_size', type=int, default=None)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--grad_accum', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder='vae').to(args.device)
    model = CrossDetailMultiPerspectiveSynthesizer()
    model.conditioning.vae = vae
    model.to(args.device)
    loss_fn = JointDenoisingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = SynthesizerTrainer(
        model=model,
        vae=vae,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        iterations=args.iterations,
        gradient_accumulation_steps=args.grad_accum,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.out,
    )
    dataset = load_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    trainer.train(dataloader)


if __name__ == '__main__':
    main()

import argparse
import os

import torch

from sharc.models.body import PIXIEEstimator
from sharc.utils.io import load_image
from sharc.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    image = load_image(args.image).unsqueeze(0).to(args.device)
    estimator = PIXIEEstimator(device=args.device)
    param_dict = estimator.estimate(image)
    params = estimator.extract_params(param_dict)
    os.makedirs(args.out, exist_ok=True)
    torch.save(params, os.path.join(args.out, 'smplx_params.pt'))


if __name__ == '__main__':
    main()

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import Dual_unet
from sr3.data import CTNpzDataset
from sr3.diffusion import GaussianDiffusion, load_filter_operator


def build_parser():
    parser = argparse.ArgumentParser(description="Sample UHRCT slices with the lightweight SR3 model.")
    parser.add_argument("--input_dir", required=True, help="Directory with LR .npz files, or HR files for synthetic LR.")
    parser.add_argument("--output_dir", default="results/sr3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--filter_ckpt", default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def main():
    opt = build_parser().parse_args()
    device = torch.device(opt.device if opt.device == "cpu" or torch.cuda.is_available() else "cpu")
    os.makedirs(opt.output_dir, exist_ok=True)

    dataset = CTNpzDataset(
        hr_dir=opt.input_dir,
        lr_dir=None,
        crop_size=opt.crop_size,
        scale=opt.scale,
        random_crop=False,
        random_flip=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = Dual_unet(image_channels=3, structure_channels=1, base_channels=opt.base_channels, out_channels=1)
    structure_operator = load_filter_operator(opt.filter_ckpt, device=device, trainable=False)
    diffusion = GaussianDiffusion(
        model,
        steps=opt.steps,
        sample_steps=opt.sample_steps,
        structure_operator=structure_operator,
    ).to(device)

    state = torch.load(opt.checkpoint, map_location=device)
    diffusion.model.load_state_dict(state.get("ema_model", state["model"]), strict=False)
    diffusion.eval()

    with torch.no_grad():
        for index, (lr_img, _) in enumerate(loader):
            lr_img = lr_img.to(device)
            sr = diffusion.sample(lr_img).squeeze().cpu().numpy().astype(np.float32)
            output_path = os.path.join(opt.output_dir, "{:04d}.npz".format(index))
            np.savez_compressed(output_path, sr)
            print("Saved {}".format(output_path))


if __name__ == "__main__":
    main()

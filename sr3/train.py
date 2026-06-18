import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Dual_unet
from sr3.data import CTNpzDataset
from sr3.diffusion import GaussianDiffusion, load_filter_operator


def build_parser():
    parser = argparse.ArgumentParser(description="Lightweight SR3 training for UHRCT super-resolution.")
    parser.add_argument("--hr_dir", required=True, help="Directory with HR/UHRCT .npz files.")
    parser.add_argument("--lr_dir", default=None, help="Optional directory with matched LR .npz files.")
    parser.add_argument("--save_dir", default="checkpoints/sr3")
    parser.add_argument("--filter_ckpt", default=None, help="Optional pretrained Filter_UNet checkpoint.")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--structure_weight", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--resume", default=None)
    return parser


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def save_checkpoint(path, diffusion, ema_diffusion, optimizer, epoch, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": diffusion.model.state_dict(),
            "ema_model": ema_diffusion.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def main():
    opt = build_parser().parse_args()
    device = torch.device(opt.device if opt.device == "cpu" or torch.cuda.is_available() else "cpu")
    os.makedirs(opt.save_dir, exist_ok=True)

    dataset = CTNpzDataset(
        hr_dir=opt.hr_dir,
        lr_dir=opt.lr_dir,
        crop_size=opt.crop_size,
        scale=opt.scale,
        random_crop=True,
        random_flip=True,
    )
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads, pin_memory=True)

    denoise_model = Dual_unet(image_channels=3, structure_channels=1, base_channels=opt.base_channels, out_channels=1)
    ema_model = copy.deepcopy(denoise_model)
    structure_operator = load_filter_operator(opt.filter_ckpt, device=device, trainable=False)
    ema_structure_operator = copy.deepcopy(structure_operator)

    diffusion = GaussianDiffusion(
        denoise_model,
        steps=opt.steps,
        sample_steps=opt.sample_steps,
        structure_operator=structure_operator,
        structure_weight=opt.structure_weight,
    ).to(device)
    ema_diffusion = GaussianDiffusion(
        ema_model,
        steps=opt.steps,
        sample_steps=opt.sample_steps,
        structure_operator=ema_structure_operator,
        structure_weight=opt.structure_weight,
    ).to(device)

    optimizer = optim.Adam(diffusion.model.parameters(), lr=opt.lr)
    start_epoch = 1
    if opt.resume is not None:
        state = torch.load(opt.resume, map_location=device)
        diffusion.model.load_state_dict(state["model"], strict=False)
        ema_diffusion.model.load_state_dict(state.get("ema_model", state["model"]), strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, opt.epochs + 1):
        diffusion.train()
        running_loss = 0.0
        for iteration, (lr_img, hr_img) in enumerate(loader, start=1):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            loss = diffusion(hr_img, lr_img)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(diffusion.model, ema_diffusion.model)

            running_loss += loss.item()
            if iteration % 20 == 0:
                print("Epoch {} Iter {} Loss {:.6f}".format(epoch, iteration, running_loss / iteration))

        avg_loss = running_loss / max(len(loader), 1)
        ckpt_path = os.path.join(opt.save_dir, "last.pt")
        save_checkpoint(ckpt_path, diffusion, ema_diffusion, optimizer, epoch, avg_loss)
        print("Epoch {} finished. Avg loss {:.6f}. Saved {}".format(epoch, avg_loss, ckpt_path))


if __name__ == "__main__":
    main()

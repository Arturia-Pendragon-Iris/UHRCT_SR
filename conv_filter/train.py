import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from .model import Filter_UNet
    from .utils import TrainSetLoader
except ImportError:
    from model import Filter_UNet
    from utils import TrainSetLoader


def build_parser():
    parser = argparse.ArgumentParser(description="Pretrain OFc to approximate Frangi enhancement.")
    parser.add_argument("--data", required=True, help="Directory containing .npz CT slices or stacks.")
    parser.add_argument("--save_dir", default="checkpoints/filter", help="Directory to store filter checkpoints.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no_aug", action="store_true")
    return parser


def train_one_epoch(loader, model, optimizer, criterion, device, epoch):
    model.train()
    loss_epoch = 0.0
    for iteration, (raw, target) in enumerate(loader, start=1):
        raw = raw.to(device)
        target = target.to(device)
        prediction = model(raw)
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if iteration % 20 == 0:
            avg_loss = loss_epoch / iteration
            print("Epoch {} Iter {} Loss {:.6f}".format(epoch, iteration, avg_loss))
    return loss_epoch / max(len(loader), 1)


def main():
    opt = build_parser().parse_args()
    device = torch.device(opt.device if opt.device == "cpu" or torch.cuda.is_available() else "cpu")
    os.makedirs(opt.save_dir, exist_ok=True)

    dataset = TrainSetLoader(opt.data, device=None, transform=not opt.no_aug)
    loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.threads, shuffle=True)

    model = Filter_UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.L1Loss()

    for epoch in range(1, opt.epochs + 1):
        avg_loss = train_one_epoch(loader, model, optimizer, criterion, device, epoch)
        ckpt_path = os.path.join(opt.save_dir, "filter_unet_last.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg_loss}, ckpt_path)
        print("Epoch {} finished. Avg loss {:.6f}. Saved {}".format(epoch, avg_loss, ckpt_path))


if __name__ == "__main__":
    main()

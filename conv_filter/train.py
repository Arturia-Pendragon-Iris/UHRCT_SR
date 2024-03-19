import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import Filter_UNet
from .utils import TrainSetLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--gamma', type=float, default=0.8, help='Learning Rate decay')


def ln_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    loss_2 = nn.MSELoss()
    return loss_1(prediction_results, ground_truth) + 5 * loss_2(prediction_results, ground_truth)


def grad_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    return loss_1((prediction_results[:, :, 1:, 1:] - prediction_results[:, :, :-1, :-1]),
                  ground_truth[:, :, 1:, 1:] - ground_truth[:, :, :-1, :-1])



def train():
    opt = parser.parse_args()
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = Filter_UNet()
    model = model.cuda()

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        training_set = TrainSetLoader('/data/Train_and_Test/super_resolution/SR*2', device)
        training_loader = DataLoader(dataset=training_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
        trainor(training_loader, optimizer, model, epoch)
        scheduler.step()
        # seg_scheduler.step()


def trainor(training_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    weight = 10
    model.train()
    loss_epoch = 0
    for iteration, (raw, gt) in enumerate(training_loader):
        # print(raw.shape, gt.shape)
        pre = model(raw)
        # print(pre.shape)
        # pre_cpu = raw.cpu().detach().numpy()[0, 0]
        # noisy_cpu = gt.cpu().detach().numpy()[0, 0]
        # # clean_cpu = clean.cpu().detach().numpy()[0, 0]
        # plot_parallel(
        #     a=pre_cpu,
        #     # b=clean_cpu,
        #     c=noisy_cpu
        # )
        loss = ln_loss(pre, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 200 + 1)))

        if (iteration + 1) % 200 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/data/Model/slice_process/filter")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, "jerman.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


train()



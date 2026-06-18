import argparse
import os

import numpy as np

from conv_filter.filter import perform_filter


def psnr(prediction, target, eps=1e-10):
    mse = np.mean((prediction - target) ** 2)
    return -10.0 * np.log10(mse + eps)


def smse(prediction, target):
    prediction_structure = perform_filter(prediction, hu_min=0.0, hu_max=1.0)
    target_structure = perform_filter(target, hu_min=0.0, hu_max=1.0)
    return float(np.mean((prediction_structure - target_structure) ** 2))


def _list_npz(root):
    return sorted([os.path.join(root, name) for name in os.listdir(root) if name.endswith(".npz")])


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRCT results with PSNR and Frangi SMSE.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    opt = parser.parse_args()

    pred_files = _list_npz(opt.pred_dir)
    gt_files = _list_npz(opt.gt_dir)
    if len(pred_files) != len(gt_files):
        raise ValueError("pred_dir and gt_dir contain different numbers of .npz files.")

    psnr_values = []
    smse_values = []
    for pred_path, gt_path in zip(pred_files, gt_files):
        pred = np.load(pred_path)["arr_0"].astype(np.float32)
        gt = np.load(gt_path)["arr_0"].astype(np.float32)
        psnr_values.append(psnr(pred, gt))
        smse_values.append(smse(pred, gt))

    print("PSNR {:.4f}".format(float(np.mean(psnr_values))))
    print("SMSE {:.6f}".format(float(np.mean(smse_values))))


if __name__ == "__main__":
    main()

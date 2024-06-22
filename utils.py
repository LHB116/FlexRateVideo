import os
import numpy as np
import random
import datetime
import argparse
import math

import struct
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor


def get_args():
    parser = argparse.ArgumentParser(description='Deep Video Coding')
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--state", type=str, default="train", choices=["train", "test"])
    # model path
    # ./logs/WOSM_VB1_mbt2018/checkpoints/step_664000.pth
    parser.add_argument("--model_restore_path", type=str,
                        default='./logs/VB_loadMSE_MSSSIM_1.2_34.0/checkpoints/step_1516000.pth')
    parser.add_argument("--load_pretrained", type=bool, default=True)

    parser.add_argument("--log_root", type=str, default="./logs")

    parser.add_argument("--mode_type", type=str, default='PSNR')  # PSNR  MSSSIM  I_level
    parser.add_argument("--l_PSNR", type=int, default=2048, choices=[256, 512, 1024, 2048])
    parser.add_argument("--l_MSSSIM", type=int, default=32, choices=[8, 16, 32, 64])
    parser.add_argument("--batch_size", type=int, default=2)  # 8
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--image_size", type=list, default=[256, 256, 3])

    # Dataset preprocess parameters
    parser.add_argument("--dataset_root", type=str,
                        default='/tdx/LHB/data/vimeo_septuplet')
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--train_dataset_root", type=str, default='D:/DataSet/Flicker/train')
    parser.add_argument("--val_dataset_root", type=str, default='./data/image/kodim')

    # Optimizer parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--aux_lr', type=float, default=1e-3)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.9999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--regular_weight", type=float, default=1e-5)
    parser.add_argument("--clip_max_norm", default=0.5, type=float, help="gradient clipping max norm ")

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4, help='# num_workers')

    return parser.parse_args()


def read_image(filepath):
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def cal_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def fix_random_seed(seed_value=2021):
    os.environ['PYTHONPATHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return 0


def save_weights(name, model, optim, scheduler, root, iteration):
    path = os.path.join(root, "{}_weights.pth".format(name))
    state = dict()
    state["name"] = name
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None
    torch.save(state, path)


def save_model(root, name, model):
    path = os.path.join(root, "{}_all.pth".format(name))
    torch.save(model, path)


def load_state(path, cuda):
    if cuda:
        print("INFO: [*] Load Mode To GPU")
        state = torch.load(path)
    else:
        print("INFO: [*] Load To CPU")
        state = torch.load(path, map_location=lambda storage, loc: storage)
    return state


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rename_key(key):
    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_pretrained(state_dict):
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def Average(lst):
    return sum(lst) / len(lst)


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def show_image(img: Image.Image):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()

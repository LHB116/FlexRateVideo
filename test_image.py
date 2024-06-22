# -*- coding: utf-8 -*-
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
from pathlib import Path
import torch
import compressai
from compressai.zoo import image_models, models
from compressai.utils import plot
from image_model import ICIP2020ResBVB1, GainedCheng2020v1
from utils import load_pretrained, read_body, torch2img, show_image, load_image, img2torch, pad, write_body, filesize
from utils import compute_psnr, compute_msssim, compute_bpp, ms_ssim, read_image, cal_psnr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

def TestImageCodec():
    # fun 0, 1 write strings and save images; 2, 3 fast test
    compressai.set_entropy_coder("ans")
    quality = 0.01
    # ckpt = r'D:\Project\Pytorch\DeepVideoCoding\DCVC\checkpoints\cheng2020-anchor-3-e49be189.pth.tar'
    # ckpt = './ckpt/0612/MeanScaleHyperprior_PSNR0.01/checkpoint.pth'  # checkpoint
    # ckpt = './ckpt/0612/Cheng2020Anchor_PSNR0.01/checkpoint.pth'
    ckpt = r'E:\temp\0615\Cheng2020Anchor_PSNR0.01\checkpoint.pth'
    img_path = './data/image/kodim'
    string_path = f'./output/bin/{quality}'
    os.makedirs(string_path, exist_ok=True)
    rec_path = f'./output/rec/{quality}'
    os.makedirs(rec_path, exist_ok=True)

    codec = Cheng2020Anchor()
    state_dict = torch.load(ckpt, map_location='cuda:0')["state_dict"]
    state_dict = load_pretrained(state_dict)
    codec.load_state_dict(state_dict)
    codec.update(force=True)

    images = glob.glob(os.path.join(img_path, '*.png'))
    print(f'* Find {len(images)} Images')
    Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
    for path in images:
        name = path.split('\\')[-1].split('.')[0]
        image = read_image(path).unsqueeze(0)

        num_pixels = image.size(0) * image.size(2) * image.size(3)
        # print(image.shape, image.shape[2] % 64, image.shape[3] % 64)

        with torch.no_grad():
            start = time.time()
            out_enc = codec.compress(image)
            enc_time = time.time() - start
            encT.append(enc_time)

            start = time.time()
            out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
            dec_time = time.time() - start
            decT.append(dec_time)

            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            psnr = cal_psnr(image, out_dec["x_hat"])
            ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
            PSNR.append(psnr)
            Bpp.append(bpp)
            MS_SSIM.append(ms_ssim1)
        print(
            f"{name} | Quality {quality} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
            f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
    print(
        f'Quality {quality} | Average BPP {np.mean(Bpp):.4f} | PSRN {np.mean(PSNR):.4f} | MSSSIM {np.mean(MS_SSIM):.4f}'
        f' | Encode Time {np.mean(encT):.4f} | Decode Time {np.mean(decT):.4f}')

    results = {"psnr": np.mean(PSNR), "ms-ssim": np.mean(MS_SSIM), "bpp": np.mean(Bpp),
               "encoding_time": np.mean(encT), "decoding_time": np.mean(decT)}
    output = {
        "name": 'lhb-fixhyperior_mse',
        "description": "Inference (ans)",
        "results": results,
    }
    # print(json.dumps(output, indent=2))
    with open("./output/test.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    return 0


def TestGainedImageCodec():
    compressai.set_entropy_coder("ans")

    ckpt = './checkpoint/GainedCheng2020v2.pth'
    img_path = './data/image/kodim'
    codec = GainedCheng2020v2()
    state_dict = torch.load(ckpt, map_location='cpu')
    # print(state_dict.keys())
    # exit()
    state_dict = load_pretrained(state_dict["state_dict"])
    codec.load_state_dict(state_dict)
    codec.update(force=True)
    codec.cuda()

    levels_intervals = []
    # for level in range(0, codec.levels - 1):
    #     levels_intervals.append((level, 1.0))
    for level in range(0, codec.levels - 1):
        for interval in np.arange(1., 0., -0.1):
            levels_intervals.append((level, interval))

    Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
    for level, interval in levels_intervals:
        images = glob.glob(os.path.join(img_path, '*.png'))
        print(f'* Find {len(images)} Images')
        _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
        for path in images:
            name = path.split('\\')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()
            num_pixels = image.size(0) * image.size(2) * image.size(3)

            with torch.no_grad():
                psnr = float('nan')
                judge = math.isnan(psnr)
                while judge:
                    start = time.time()
                    out_enc = codec.compress(image, level, interval)
                    enc_time = time.time() - start

                    start = time.time()
                    out_dec = codec.decompress(out_enc["strings"], out_enc["shape"], level, interval)
                    dec_time = time.time() - start

                    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                    psnr = cal_psnr(image, out_dec["x_hat"])
                    ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                    judge = math.isnan(psnr) or psnr < 0.0 or math.isinf(psnr)

                _encT.append(enc_time)
                _decT.append(dec_time)
                _PSNR.append(psnr)
                _Bpp.append(bpp)
                _MS_SSIM.append(ms_ssim1)

            print(
                f"{name} | Quality {level, interval} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
        print(
            f'Quality {level, interval} | Average BPP {np.mean(_Bpp):.4f} | PSRN {np.mean(_PSNR):.4f} | MSSSIM {np.mean(_MS_SSIM):.4f}'
            f' | Encode Time {np.mean(_encT):.4f} | Decode Time {np.mean(_decT):.4f}')

        PSNR.append(np.mean(_PSNR))
        MS_SSIM.append(np.mean(_MS_SSIM))
        Bpp.append(np.mean(_Bpp))
        encT.append(np.mean(_encT))
        decT.append(np.mean(_decT))
    print(f'BPP: {Bpp}')
    print(f'PSNR : {PSNR}')
    print(f'MSSSIM : {MS_SSIM}')

    results = {"psnr": PSNR, "ms-ssim": MS_SSIM, "bpp": Bpp, "encoding_time": encT, "decoding_time": decT}
    output = {"name": 'cheng2020hyperior_gain_mse', "description": "Inference (ans)", "results": results}
    with open("./output/cheng2020hyperior_gain_mse.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    return 0


def parse_json_file(filepath, metric, db_ssim=False):
    # psnr  ms-ssim  bpp  encoding_time  decoding_time
    filepath = Path(filepath)
    name = filepath.name.split(".")[0]
    with filepath.open("r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{filepath}"')
            raise err

    if "results" in data:
        results = data["results"]
    else:
        results = data

    if metric not in results:
        raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(results.keys())}'
        )

    try:
        if metric == "ms-ssim" and db_ssim:
            # Convert to db
            values = np.array(results[metric])
            results[metric] = -10 * np.log10(1 - values)

        return {
            "name": data.get("name", name),
            "xs": results["bpp"],
            "ys": results[metric],
        }
    except KeyError:
        raise ValueError(f'Invalid file "{filepath}"')


def matplotlib_plt(scatters, title, ylabel, output_file, limits=None, show=False, figsize=None):
    linestyle = "-"
    hybrid_matches = ["HM", "VTM", "JPEG", "JPEG2000", "WebP", "BPG", "AV1"]
    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    for sc in scatters:
        if any(x in sc["name"] for x in hybrid_matches):
            linestyle = "--"
        ax.plot(
            sc["xs"],
            sc["ys"],
            marker=".",
            linestyle=linestyle,
            linewidth=0.7,
            label=sc["name"],
        )

    ax.set_xlabel("Bit-rate [bpp]")
    ax.set_ylabel(ylabel)
    ax.grid()
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc="lower right")

    if title:
        ax.title.set_text(title)

    if show:
        plt.show()

    if output_file:
        fig.savefig(output_file, dpi=300)


def plot_result():
    metric = 'psnr'
    scatters = []
    # UVG
    # results_file = glob.glob(os.path.join('./data/UVG', '*.json'))
    results_file = glob.glob(os.path.join('./data/kodak_result/tgt/mse', '*.json'))
    for f in results_file:
        rv = parse_json_file(f, metric)
        scatters.append(rv)

    ylabel = f"{metric} [dB]"

    matplotlib_plt(scatters, title='Demo', ylabel=ylabel, output_file='./output/test.png', show=True)
    return 0


def TestFGainedImageCodec():
    compressai.set_entropy_coder("ans")

    ckpt = '/home/user/桌面/LHB/DVC/OpenDVC/GainedDVC/v3/logs/ImageFGained/checkpoints/checkpoint_499.pth'
    img_path = './data/image/kodim'
    codec = FGainedMeanScale()
    state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
    # print(state_dict.keys())
    # print(state_dict['HyperGain'])
    # exit()
    state_dict = load_pretrained(state_dict)
    codec.load_state_dict(state_dict)
    codec.update(force=True)

    M = 5
    alpha_list = [1 - j / M for j in range(5)]

    lambda_list = torch.tensor([0.2000, 0.1000, 0.0500, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018])
    lambda_onehot = torch.eye(len(lambda_list))

    Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
    for index_rand in range(len(lambda_list) - 1):
        for alpha_rand in alpha_list:
            l_onehot = alpha_rand * lambda_onehot[index_rand] + (1 - alpha_rand) * lambda_onehot[index_rand + 1]
            images = glob.glob(os.path.join(img_path, '*.png'))
            print(f'* Find {len(images)} Images')
            _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
            for path in images:
                name = path.split('/')[-1].split('.')[0]
                image = read_image(path).unsqueeze(0)

                num_pixels = image.size(0) * image.size(2) * image.size(3)
                # print(image.shape, image.shape[2] % 64, image.shape[3] % 64)

                with torch.no_grad():
                    start = time.time()
                    out_enc = codec.compress(image, l_onehot)
                    enc_time = time.time() - start
                    # encT.append(enc_time)
                    _encT.append(enc_time)

                    start = time.time()
                    out_dec = codec.decompress(out_enc["strings"], out_enc["shape"], l_onehot)
                    dec_time = time.time() - start
                    # decT.append(dec_time)
                    _decT.append(dec_time)

                    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                    psnr = cal_psnr(image, out_dec["x_hat"])
                    ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()
                    # PSNR.append(psnr)
                    # Bpp.append(bpp)
                    # MS_SSIM.append(ms_ssim1)
                    _PSNR.append(psnr)
                    _Bpp.append(bpp)
                    _MS_SSIM.append(ms_ssim1)

                print(
                    f"{name} | Quality {l_onehot} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                    f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
            print(
                f'Quality {l_onehot} | Average BPP {np.mean(_Bpp):.4f} | PSRN {np.mean(_PSNR):.4f} | MSSSIM {np.mean(_MS_SSIM):.4f}'
                f' | Encode Time {np.mean(_encT):.4f} | Decode Time {np.mean(_decT):.4f}')

            PSNR.append(np.mean(_PSNR))
            MS_SSIM.append(np.mean(_MS_SSIM))
            Bpp.append(np.mean(_Bpp))
            encT.append(np.mean(_encT))
            decT.append(np.mean(_decT))
    print(f'BPP: {Bpp}')
    print(f'PSNR : {PSNR}')
    print(f'MSSSIM : {MS_SSIM}')

    results = {"psnr": PSNR, "ms-ssim": MS_SSIM, "bpp": Bpp, "encoding_time": encT, "decoding_time": decT}
    output = {"name": 'FGainedMeanScale', "description": "Inference (ans)", "results": results}
    with open("./output/FGainedMeanScale.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)
    return 0


def TestGainedImageCodec1():
    epoch = 4
    # save_path = './logs/ICIP2020ResBVB1_v0True_0.2548'
    # ckpt = f'./logs/ICIP2020ResBVB1_v0True_0.2548/checkpoints/checkpoint_{epoch}.pth'

    save_path = './ckpt'
    ckpt = f'./ckpt/ICIP2020ResBVB1_psnr.pth'
    img_path = '/tdx/LHB/code/torch/IMC/data/image/kodim'

    compressai.set_entropy_coder("ans")
    codec = ICIP2020ResBVB1(v0=True)
    state_dict = torch.load(ckpt, map_location='cpu')
    state_dict = load_pretrained(state_dict["state_dict"])
    codec.load_state_dict(state_dict)
    codec.update(force=True)
    codec.cuda()

    levels_intervals = []
    print(codec.levels - 1)
    # for level in range(0, codec.levels - 1):
    #     levels_intervals.append((level, 1.0))
    # for level in range(0, codec.levels - 1):
    #     for interval in np.arange(1., 0., -0.1):
    #         levels_intervals.append((level, interval))

    # for level in [0, 1, 2, 3]:
    #     intervals = [1] if level != 3 else [1, 0.05]
    #     for interval in intervals:
    #         levels_intervals.append((codec.levels - 4 - level, 1.0 - interval))
    # print(levels_intervals)
    # levels_intervals = [(4, 0.0), (3, 0.0), (2, 0.0), (1, 0.0), (1, 0.95)]
    # exit()

    Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
    for level, interval in levels_intervals:
        images = sorted(glob.glob(os.path.join(img_path, '*.png')))
        print(f'* Find {len(images)} Images')
        _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
        for path in images:
            name = path.split('/')[-1].split('.')[0]
            image = read_image(path).unsqueeze(0).cuda()
            num_pixels = image.size(0) * image.size(2) * image.size(3)

            with torch.no_grad():
                start = time.time()
                out_enc = codec.compress(image, [level], interval)
                enc_time = time.time() - start

                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"], [level], interval)
                dec_time = time.time() - start

                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                psnr = cal_psnr(image, out_dec["x_hat"])
                ms_ssim1 = ms_ssim(image, out_dec["x_hat"], data_range=1.0).item()

                _encT.append(enc_time)
                _decT.append(dec_time)
                _PSNR.append(psnr)
                _Bpp.append(bpp)
                _MS_SSIM.append(ms_ssim1)

            print(
                f"{name} | Quality {level, interval} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                f"| MS-SSIM {ms_ssim1:.4f} | PSNR {psnr:.4f}")
        print(
            f'Quality {level, interval} | Average BPP {np.mean(_Bpp):.4f} | PSRN {np.mean(_PSNR):.4f} | MSSSIM {np.mean(_MS_SSIM):.4f}'
            f' | Encode Time {np.mean(_encT):.4f} | Decode Time {np.mean(_decT):.4f}')

        PSNR.append(np.mean(_PSNR))
        MS_SSIM.append(np.mean(_MS_SSIM))
        Bpp.append(np.mean(_Bpp))
        encT.append(np.mean(_encT))
        decT.append(np.mean(_decT))
    print(f'BPP: {Bpp}')
    print(f'PSNR : {PSNR}')
    print(f'MSSSIM : {MS_SSIM}')

    results = {"psnr": PSNR, "ms-ssim": MS_SSIM, "bpp": Bpp, "encoding_time": encT, "decoding_time": decT}
    output = {"name": f'mark', "description": "Inference (ans)", "results": results}
    with open(f"{save_path}/epoch_{epoch}.json", 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=2)

    Bpp1 = [
      0.3097127278645833,
      0.4498155381944444,
      0.6407640245225694,
      0.8683946397569445,
      1.1432020399305556
    ]
    PSNR1 = [
      32.377160696098976,
      34.10368974430478,
      35.96023583934226,
      37.754727226130875,
      39.49347830473297
    ]
    plt.plot(Bpp1, PSNR1, "g--s", label='ICIP2020ResBx3')

    Bpp1 = [
      1.01220703125,
      0.8363884819878473,
      0.6334025065104166,
      0.46280585394965273,
      0.3238389756944444,
      0.22132025824652776
    ]
    PSNR1 = [
      38.617502210046716,
      37.535266589595345,
      35.957909463787736,
      34.27419433083306,
      32.55073839290734,
      30.9140300775
    ]
    plt.plot(Bpp1, PSNR1, "b--s", label='gained_ICIP2020ResBx3')
    plt.plot(Bpp, PSNR, "r-o", label='gained_ICIP2020ResB_newLambdas')
    plt.grid()
    plt.ylabel("PSNR (dB)")
    plt.xlabel("bpp (bit/pixel)")
    plt.legend()
    plt.savefig(f"{save_path}/epoch_{epoch}.png")
    plt.show()

    return 0


if __name__ == "__main__":
    # ImageCodec()
    # TestImageCodec()
    # TestGainedImageCodec()
    # BPP: [0.736175537109375, 0.44211493598090285, 0.3187594943576389]
    # PSNR: [34.966544021244424, 32.72069323164284, 31.1944067680973]
    TestGainedImageCodec1()

# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import matplotlib.pyplot as plt
from utils import AverageMeter
import torch
import time
import numpy as np
from pytorch_msssim import ms_ssim
from image_model import ICIP2020ResB, ICIP2020ResBVB1
from video_model import LHB_DVC_WOSM, LHB_DVC, LHB_DVC_WOSM_VB
from utils import read_image, cal_psnr, load_pretrained, crop, pad
from compressai.zoo import mbt2018
import json
import glob
import math
import matplotlib


torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

TEST_DATA = {
    'HEVC_B': {
        'path': '/tdx/LHB/data/TestSets/ClassB',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': {
            'BasketballDrive_1920x1080_50',
            'BQTerrace_1920x1080_60',
            'Cactus_1920x1080_50',
            'Kimono1_1920x1080_24',
            'ParkScene_1920x1080_24',
        },
    },

    'HEVC_B1': {
        'path': '/tdx/LHB/data/TestSets/ClassB',
        'frames': 96,
        'gop': 12,
        'org_resolution': '640x640',
        'x64_resolution': '640x640',
        'sequences': {
            'Kimono1_640x640_24_f96_gauss_0.002',
        },
    },

    'HEVC_C1': {
        'path': '/tdx/LHB/data/TestSets/ClassC/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_C': {
        'path': '/tdx/LHB/data/TestSets/ClassC/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'BasketballDrill_832x480_50',
            'BQMall_832x480_60',
            'PartyScene_832x480_50',
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_D': {
        'path': '/tdx/LHB/data/TestSets/ClassD/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '416x240',
        'x64_resolution': '384x192',
        'sequences': [
            'BasketballPass_416x240_50',
            'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60',
            'RaceHorses_416x240_30',
        ],
    },

    'HEVC_E': {
        'path': '/tdx/LHB/data/TestSets/ClassE/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1280x720',
        'x64_resolution': '1280x704',
        'sequences': [
            'FourPeople_1280x720_60',
            'Johnny_1280x720_60',
            'KristenAndSara_1280x720_60',
        ],
    },

    'UVG': {
        'path': '/tdx/LHB/data/TestSets/UVG/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': [
            'Beauty_1920x1080_120fps_420_8bit_YUV',
            'Bosphorus_1920x1080_120fps_420_8bit_YUV',
            'HoneyBee_1920x1080_120fps_420_8bit_YUV',
            'Jockey_1920x1080_120fps_420_8bit_YUV',
            'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV',
            'ShakeNDry_1920x1080_120fps_420_8bit_YUV',
            'YachtRide_1920x1080_120fps_420_8bit_YUV',
        ],
    },

    "MCL-JCV": {
        "path": "/tdx/LHB/data/TestSets/MCL-JCV",
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',  # 18,20,24,25
        "sequences": [
            "videoSRC01_1920x1080_30",
            "videoSRC02_1920x1080_30",
            "videoSRC03_1920x1080_30",
            "videoSRC04_1920x1080_30",
            "videoSRC05_1920x1080_25",
            "videoSRC06_1920x1080_25",
            "videoSRC07_1920x1080_25",
            "videoSRC08_1920x1080_25",
            "videoSRC09_1920x1080_25",
            "videoSRC10_1920x1080_30",
            "videoSRC11_1920x1080_30",
            "videoSRC12_1920x1080_30",
            "videoSRC13_1920x1080_30",
            "videoSRC14_1920x1080_30",
            "videoSRC15_1920x1080_30",
            "videoSRC16_1920x1080_30",
            "videoSRC17_1920x1080_24",
            "videoSRC18_1920x1080_25",
            "videoSRC19_1920x1080_30",
            "videoSRC20_1920x1080_25",
            "videoSRC21_1920x1080_24",
            "videoSRC22_1920x1080_24",
            "videoSRC23_1920x1080_24",
            "videoSRC24_1920x1080_24",
            "videoSRC25_1920x1080_24",
            "videoSRC26_1920x1080_30",
            "videoSRC27_1920x1080_30",
            "videoSRC28_1920x1080_30",
            "videoSRC29_1920x1080_24",
            "videoSRC30_1920x1080_30",
        ]
    },

    'VTL': {
        'path': '/tdx/LHB/data/TestSets/VTL',
        'frames': 96,
        'gop': 12,
        'org_resolution': '352x288',
        'x64_resolution': '320x256',
        'sequences': [
            'akiyo_cif',
            'BigBuckBunny_CIF_24fps',
            'bridge-close_cif',
            'bridge-far_cif',
            'bus_cif',
            'coastguard_cif',
            'container_cif',
            'ElephantsDream_CIF_24fps',
            'flower_cif',
            'foreman_cif',
            'hall_cif',
            'highway_cif',
            'mobile_cif',
            'mother-daughter_cif',
            'news_cif',
            'paris_cif',
            'silent_cif',
            'stefan_cif',
            'tempete_cif',
            'waterfall_cif',
        ],
    },
}


TEST_DATA_ORG = {
    'HEVC_B': {
        'path': '/tdx/LHB/data/ORGreslolutions/HEVC_B',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': {
            'BasketballDrive_1920x1080_50',
            'BQTerrace_1920x1080_60',
            'Cactus_1920x1080_50',
            'Kimono1_1920x1080_24',
            'ParkScene_1920x1080_24',
        },
    },

    'HEVC_C': {
        'path': '/tdx/LHB/data/ORGreslolutions/HEVC_C/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '832x480',
        'x64_resolution': '832x448',
        'sequences': [
            'BasketballDrill_832x480_50',
            'BQMall_832x480_60',
            'PartyScene_832x480_50',
            'RaceHorses_832x480_30',
        ],
    },

    'HEVC_D': {
        'path': '/tdx/LHB/data/ORGreslolutions/HEVC_D/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '416x240',
        'x64_resolution': '384x192',
        'sequences': [
            'BasketballPass_416x240_50',
            'BlowingBubbles_416x240_50',
            'BQSquare_416x240_60',
            'RaceHorses_416x240_30',
        ],
    },

    'HEVC_E': {
        'path': '/tdx/LHB/data/ORGreslolutions/HEVC_E/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1280x720',
        'x64_resolution': '1280x704',
        'sequences': [
            'FourPeople_1280x720_60',
            'Johnny_1280x720_60',
            'KristenAndSara_1280x720_60',
        ],
    },

    'UVG': {
        'path': '/tdx/LHB/data/ORGreslolutions/UVG/',
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',
        'sequences': [
            'Beauty_1920x1080_120fps_420_8bit_YUV',
            'Bosphorus_1920x1080_120fps_420_8bit_YUV',
            'HoneyBee_1920x1080_120fps_420_8bit_YUV',
            'Jockey_1920x1080_120fps_420_8bit_YUV',
            'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV',
            'ShakeNDry_1920x1080_120fps_420_8bit_YUV',
            'YachtRide_1920x1080_120fps_420_8bit_YUV',
        ],
    },

    "MCL-JCV": {
        "path": "/tdx/LHB/data/ORGreslolutions/MCL-JCV",
        'frames': 96,
        'gop': 12,
        'org_resolution': '1920x1080',
        'x64_resolution': '1920x1024',  # 18,20,24,25
        "sequences": [
            "videoSRC01_1920x1080_30",
            "videoSRC02_1920x1080_30",
            "videoSRC03_1920x1080_30",
            "videoSRC04_1920x1080_30",
            "videoSRC05_1920x1080_25",
            "videoSRC06_1920x1080_25",
            "videoSRC07_1920x1080_25",
            "videoSRC08_1920x1080_25",
            "videoSRC09_1920x1080_25",
            "videoSRC10_1920x1080_30",
            "videoSRC11_1920x1080_30",
            "videoSRC12_1920x1080_30",
            "videoSRC13_1920x1080_30",
            "videoSRC14_1920x1080_30",
            "videoSRC15_1920x1080_30",
            "videoSRC16_1920x1080_30",
            "videoSRC17_1920x1080_24",
            "videoSRC18_1920x1080_25",
            "videoSRC19_1920x1080_30",
            "videoSRC20_1920x1080_25",
            "videoSRC21_1920x1080_24",
            "videoSRC22_1920x1080_24",
            "videoSRC23_1920x1080_24",
            "videoSRC24_1920x1080_24",
            "videoSRC25_1920x1080_24",
            "videoSRC26_1920x1080_30",
            "videoSRC27_1920x1080_30",
            "videoSRC28_1920x1080_30",
            "videoSRC29_1920x1080_24",
            "videoSRC30_1920x1080_30",
        ]
    },

    'VTL': {
        'path': '/tdx/LHB/data/ORGreslolutions/VTL',
        'frames': 96,
        'gop': 12,
        'org_resolution': '352x288',
        'x64_resolution': '320x256',
        'sequences': [
            'akiyo_cif',
            'BigBuckBunny_CIF_24fps',
            'bridge-close_cif',
            'bridge-far_cif',
            'bus_cif',
            'coastguard_cif',
            'container_cif',
            'ElephantsDream_CIF_24fps',
            'flower_cif',
            'foreman_cif',
            'hall_cif',
            'highway_cif',
            'mobile_cif',
            'mother-daughter_cif',
            'news_cif',
            'paris_cif',
            'silent_cif',
            'stefan_cif',
            'tempete_cif',
            'waterfall_cif',
        ],
    },
}


def draw_result(porposed_bpp, porposed_PSNR, cla='D', path='', mark=''):
    LineWidth = 3
    proposed, = plt.plot(porposed_bpp, porposed_PSNR, marker='x', color='black',
                         linewidth=LineWidth, label='proposed')

    if cla == 'D':
        bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
        psnr = [28.41229473, 29.72853673, 31.1727661, 32.53451213]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')
        # h264
        bpp = [0.6135016547, 0.3672749837, 0.2190138075, 0.1305982802]
        psnr = [34.30692118, 31.91254879, 29.68591275, 27.60142272]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264 Very fast')
        # h265
        bpp = [0.7361206055, 0.4330858019, 0.2476169162, 0.1408860948]
        psnr = [35.73861849, 33.21075298, 30.79006456, 28.48721492]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265 Very fast')

        bpp = [0.15289062500000006, 0.22266574435763864, 0.3445203993055555, 0.51271728515625]
        psnr = [29.70085155104196, 31.509928766799106, 33.25338537709541, 34.84352046410405]
        DVC2, = plt.plot(bpp, psnr, "b--v", linewidth=LineWidth, label='DVC2')

        plt.legend(handles=[DVC, h264, h265, proposed, DVC2], loc=4)
        plt.grid()
        plt.xlabel('BPP')
        plt.ylabel('PSNR(dB)')
        plt.title('HEVC Class D dataset')
        plt.savefig(f'{path}/{mark}')
        plt.show()
        plt.clf()

    elif cla == 'UVG':
        psnr = [34.54747736, 35.52100014, 36.68785979, 37.69306177, 38.26703496]
        bpp = [0.0601350119, 0.07662807143, 0.1085946071, 0.1853347262, 0.2388951667]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        # Ours very fast
        bpp = [0.4390169126, 0.187701634, 0.08420500256, 0.01396013948]
        psnr = [38.06400364, 36.52492848, 35.05371762, 33.56996097]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.3945488049, 0.1656631906, 0.0740901838, 0.01525909631]
        psnr = [38.82807785, 37.29259129, 35.88754733, 34.46536634]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

        plt.legend(handles=[DVC, h264, h265, proposed], loc=4)
        plt.grid()
        plt.xlabel('BPP')
        plt.ylabel('PSNR(dB)')
        plt.title('UVG dataset')
        plt.savefig('./test_UVG.png')
        plt.show()
        plt.clf()

    return 0


def get_quality(l_PSNR=0, bpg=False):
    if bpg:
        QP = 0
        if l_PSNR == 256:
            QP = 37
        elif l_PSNR == 512:
            QP = 32
        elif l_PSNR == 1024:
            QP = 27
        elif l_PSNR == 2048:
            QP = 22
        elif l_PSNR == 4096:
            QP = 17
        return QP
    else:
        I_lamdba_p, I_lamdba_m = 0.0, 0.0
        if l_PSNR == 80:
            I_lamdba_p, I_lamdba_m = 0.0067, 8.73
        elif l_PSNR == 160:
            I_lamdba_p, I_lamdba_m = 0.013, 16.64
        elif l_PSNR == 320:
            I_lamdba_p, I_lamdba_m = 0.025, 31.73
        elif l_PSNR == 640:
            I_lamdba_p, I_lamdba_m = 0.0483, 60.5
        elif l_PSNR == 1280:
            I_lamdba_p, I_lamdba_m = 0.0932, 115.37
        return I_lamdba_p, I_lamdba_m


class Process(torch.nn.Module):
    def __init__(self):
        super(Process, self).__init__()

    def forward(self, tenInput, inverse=False):
        # img_norm_cfg = dict(
        #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        if not inverse:
            tenBlue = (tenInput[:, 0:1, :, :] - 123.675) / 58.395
            tenGreen = (tenInput[:, 1:2, :, :] - 116.28) / 57.12
            tenRed = (tenInput[:, 2:3, :, :] - 103.53) / 57.375
        else:
            tenBlue = tenInput[:, 0:1, :, :] * 58.395 + 123.675
            tenGreen = tenInput[:, 1:2, :, :] * 57.12 + 116.28
            tenRed = tenInput[:, 2:3, :, :] * 57.375 + 103.53
        return torch.cat([tenRed, tenGreen, tenBlue], 1)


def get_quality1(l_PSNR=0):
    I_lamdba_p, I_lamdba_m = 0.0, 0.0
    # if l_PSNR == 4:
    #     I_lamdba_p, I_lamdba_m = 0.0067, 8.73
    # elif l_PSNR == 3:
    #     I_lamdba_p, I_lamdba_m = 0.013, 16.64
    # elif l_PSNR == 2:
    #     I_lamdba_p, I_lamdba_m = 0.025, 31.73
    # elif l_PSNR == 1:
    #     I_lamdba_p, I_lamdba_m = 0.0483, 60.5
    # elif l_PSNR == 0:
    #     I_lamdba_p, I_lamdba_m = 0.0932, 115.37

    if l_PSNR == 4:
        I_lamdba_p, I_lamdba_m = 0.0932, 8.73
    elif l_PSNR == 3:
        I_lamdba_p, I_lamdba_m = 0.0483, 16.64
    elif l_PSNR == 2:
        I_lamdba_p, I_lamdba_m = 0.025, 31.73
    elif l_PSNR == 1:
        I_lamdba_p, I_lamdba_m = 0.013, 60.5
    elif l_PSNR == 0:
        I_lamdba_p, I_lamdba_m = 0.0067, 115.37
    return I_lamdba_p, I_lamdba_m


def get_result_wosm(indicator, test_tgt):
    device = 'cuda:0'
    test_info = TEST_DATA[test_tgt]
    resolution_tgt = 'x64_resolution'
    GOP = test_info['gop']
    total_frame_num = test_info['frames']
    resolution = test_info[resolution_tgt]
    W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')

    # 'step_740000', 'step_760000', 'step_780000'
    # 'step_920000', 'step_932000', 'step_936000', 'step_940000'
    for step in ['step_940000']:
        result_save_path = f'./output/WOSM_VB1_GOP12F96_icip/{indicator.upper()}/{test_tgt}/{step}_all'
        os.makedirs(result_save_path, exist_ok=True)

        porposed_psnr, porposed_bpp, porposed_msssim, porposed_bpp2l = [], [], [], []
        porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
        porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
        porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
        porposed_mcmsssim, porposed_warmsssim = [], []
        porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
        porposed_ent, porposed_dec = [], []
        with torch.no_grad():
            if indicator == 'mse':
                restore_path = f'./logs/WOSM_VB1_icip2020/checkpoints/{step}.pth'
                # epoch = 'checkpoint_20'
            else:
                pass
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            print(f"INFO Load Pretrained P-Model From Epoch {pcheckpoint['epoch']}...")
            p_model = LHB_DVC_WOSM_VB()
            p_model.load_state_dict(pcheckpoint["state_dict"])
            p_model.eval()
            p_model.update(force=True)
            p_model.to(device)
            for l in [0, 1, 2]:
                for interval in np.arange(1., 0., -0.2):
                    log_txt = open(f'{result_save_path}/log_{test_tgt}_{l}_{step}.txt', 'w')
                    I_lamdba_p, I_lamdba_m = get_quality1(l)
                    I_lambda = I_lamdba_p if indicator == 'mse' else I_lamdba_m

                    i_model = ICIP2020ResB()
                    i_restore_path = f'/tdx/LHB/pretrained/ICIP2020ResB/mse/lambda_{I_lambda}.pth'
                    icheckpoint = torch.load(i_restore_path, map_location='cpu')
                    print(f"INFO Load Pretrained I-Model From Epoch {icheckpoint['epoch']}...")
                    state_dict = load_pretrained(icheckpoint["state_dict"])
                    i_model.load_state_dict(state_dict)
                    # i_model = mbt2018(l + 5, 'mse', True)
                    i_model.update(force=True)
                    i_model.to(device)
                    i_model.eval()

                    PSNR, MSSSIM, Bits, Bits2l = [], [], [], []
                    iPSNR, iMSSSIM, iBits = [], [], []
                    pPSNR, pMSSSIM, pBits = [], [], []
                    mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
                    mcMSSSIM, warpMSSSIM = [], []
                    iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
                    for ii, seq_info in enumerate(test_info['sequences']):
                        _PSNR, _MSSSIM, _Bits, _Bits2l = [], [], [], []
                        _iPSNR, _iMSSSIM, _iBits = [], [], []
                        _pPSNR, _pMSSSIM, _pBits = [], [], []
                        _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                        _mcMSSSIM, _warpMSSSIM = [], []
                        _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []

                        video_frame_path = os.path.join(test_info['path'], 'PNG_Frames',
                                                        seq_info.replace(test_info['org_resolution'],
                                                                         test_info[resolution_tgt]))
                        images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                        print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                        image = read_image(images[0]).unsqueeze(0)
                        num_pixels = image.size(0) * image.size(2) * image.size(3)
                        feature = None
                        for i, im in enumerate(images):
                            if i >= total_frame_num:
                                break
                            curr_frame = read_image(im).unsqueeze(0).to(device)
                            if i % GOP == 0:
                                feature = None

                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_enc = i_model.compress(curr_frame)
                                torch.cuda.synchronize()
                                elapsed_enc = time.perf_counter() - start_time
                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"])
                                torch.cuda.synchronize()
                                elapsed_dec = time.perf_counter() - start_time

                                i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                                i_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                                i_ms_ssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()

                                _iPSNR.append(i_psnr)
                                _iMSSSIM.append(i_ms_ssim)
                                _iBits.append(i_bpp)
                                _Bits2l.append(i_bpp)
                                _PSNR.append(i_psnr)
                                _MSSSIM.append(i_ms_ssim)
                                _Bits.append(i_bpp)
                                _iEnc.append(elapsed_enc)
                                _iDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(
                                    f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                                ref_frame = i_out_dec["x_hat"]
                            else:
                                torch.cuda.synchronize()
                                start = time.time()
                                mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame,
                                                                           n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_enc = time.time() - start

                                torch.cuda.synchronize()
                                start = time.time()
                                feature1, dec_p_frame, warped_frame, predict_frame = \
                                    p_model.decompress(ref_frame, mv_out_enc, res_out_enc,
                                                       n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_dec = time.time() - start

                                mse = torch.mean((curr_frame - dec_p_frame).pow(2)).item()
                                p_psnr = 10 * np.log10(1.0 / mse).item()
                                w_mse = torch.mean((curr_frame - warped_frame).pow(2)).item()
                                w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                mc_mse = torch.mean((curr_frame - predict_frame).pow(2)).item()
                                mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                p_ms_ssim = ms_ssim(curr_frame, dec_p_frame, data_range=1.0).item()
                                p_warp_ms_ssim = ms_ssim(curr_frame, warped_frame, data_range=1.0).item()
                                p_mc_ms_ssim = ms_ssim(curr_frame, predict_frame, data_range=1.0).item()
                                res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                p_bpp = mv_bpp + res_bpp

                                ref_frame = dec_p_frame.detach()
                                feature = feature1.detach()

                                _PSNR.append(p_psnr)
                                _MSSSIM.append(p_ms_ssim)
                                _Bits.append(p_bpp)
                                _pPSNR.append(p_psnr)
                                _pMSSSIM.append(p_ms_ssim)
                                _pBits.append(p_bpp)

                                _mcPSNR.append(mc_psnr)
                                _warpPSNR.append(w_psnr)
                                _mcMSSSIM.append(p_mc_ms_ssim)
                                _warpMSSSIM.append(p_warp_ms_ssim)
                                _mvBits.append(mv_bpp)
                                _resBits.append(res_bpp)
                                _Bits2l.append(mv_bpp)

                                _pEnc.append(elapsed_enc)
                                _pDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                      f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                      f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}]"
                                      f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                    f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                    f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}] "
                                    f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                        print(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                        print(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                        print(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                        log_txt.write(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')

                        PSNR.append(np.average(_PSNR))
                        MSSSIM.append(np.average(_MSSSIM))
                        Bits.append(np.average(_Bits))
                        Bits2l.append(np.average(_Bits2l))
                        iPSNR.append(np.average(_iPSNR))
                        iMSSSIM.append(np.average(_iMSSSIM))
                        iBits.append(np.average(_iBits))
                        pPSNR.append(np.average(_pPSNR))
                        pMSSSIM.append(np.average(_pMSSSIM))
                        pBits.append(np.average(_pBits))
                        mcPSNR.append(np.average(_mcPSNR))
                        warpPSNR.append(np.average(_warpPSNR))
                        mvBits.append(np.average(_mvBits))
                        resBits.append(np.average(_resBits))
                        mcMSSSIM.append(np.average(_mcMSSSIM))
                        warpMSSSIM.append(np.average(_warpMSSSIM))
                        iEnc.append(np.average(_iEnc))
                        iDec.append(np.average(_iDec))
                        pEnc.append(np.average(_pEnc))
                        pDec.append(np.average(_pDec))
                        Enc.append(np.average(_Enc))
                        Dec.append(np.average(_Dec))

                    porposed_psnr.append(np.average(PSNR))
                    porposed_bpp.append(np.average(Bits))
                    porposed_bpp2l.append(np.average(Bits2l))
                    porposed_msssim.append(np.average(MSSSIM))
                    porposed_ipsnr.append(np.average(iPSNR))
                    porposed_ibpp.append(np.average(iBits))
                    porposed_imsssim.append(np.average(iMSSSIM))
                    porposed_ppsnr.append(np.average(pPSNR))
                    porposed_pbpp.append(np.average(pBits))
                    porposed_pmsssim.append(np.average(pMSSSIM))

                    porposed_mcpsnr.append(np.average(mcPSNR))
                    porposed_warppsnr.append(np.average(warpPSNR))
                    porposed_mvbpp.append(np.average(mvBits))
                    porposed_resbpp.append(np.average(resBits))
                    porposed_mcmsssim.append(np.average(mcMSSSIM))
                    porposed_warmsssim.append(np.average(warpMSSSIM))
                    porposed_ienc.append(np.average(iEnc))
                    porposed_idec.append(np.average(iDec))
                    porposed_pent.append(np.average(pEnc))
                    porposed_pdec.append(np.average(pDec))
                    porposed_ent.append(np.average(Enc))
                    porposed_dec.append(np.average(Dec))

                log_txt.close()
            print(porposed_bpp)
            print(porposed_psnr)
            print(porposed_msssim)
            results = {
                "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim, "bpp2l": porposed_bpp2l,
                "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
                "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "pmsssim": porposed_pmsssim,
                "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
                "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
                "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
                "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
            }
            output = {
                "name": f'{test_tgt}',
                "description": "Inference (ans)",
                "results": results,
            }
            with open(os.path.join(result_save_path, f'{test_tgt}_{step}.json'), 'w',
                      encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)

            # Bpp1 = [
            #     0.07034640842013888,
            #     0.10103013780381945,
            #     0.14580055519386576,
            #     0.20352964048032407,
            #     0.28302114981192134
            # ]
            # PSNR1 = [
            #     28.76402706280694,
            #     30.19062937067432,
            #     31.67859629351529,
            #     32.96192115190013,
            #     34.12191420336034
            # ]
            # plt.plot(Bpp1, PSNR1, "b--s", label='org_wosm')
            # plt.plot(porposed_bpp, porposed_psnr, "r-o", label='gained_wosm')
            # plt.grid()
            # plt.ylabel("PSNR (dB)")
            # plt.xlabel("bpp (bit/pixel)")
            # plt.legend()
            # plt.savefig(os.path.join(result_save_path, f'{test_tgt}_{step}.png'))
    return None


def wosm_vbi_and_vbp(indicator, test_tgt):
    device = 'cuda:0'
    test_info = TEST_DATA[test_tgt]
    resolution_tgt = 'x64_resolution'
    GOP = test_info['gop']
    total_frame_num = test_info['frames']
    resolution = test_info[resolution_tgt]
    W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')

    for step in ['step_1260000']:
        # result_save_path = f'./output/WOSM_VB10_GOP12F96_icip_Final_step_108w/{indicator.upper()}/{test_tgt}'
        result_save_path = f'./output/WOSM_VB10_SSIM_GOP12F96_icip_Final_step_108w/{indicator.upper()}/{test_tgt}/{step}'
        os.makedirs(result_save_path, exist_ok=True)

        porposed_psnr, porposed_bpp, porposed_msssim, porposed_bpp2l = [], [], [], []
        porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
        porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
        porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
        porposed_mcmsssim, porposed_warmsssim = [], []
        porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
        porposed_ent, porposed_dec = [], []
        with torch.no_grad():
            if indicator == 'mse':
                restore_path = f'./logs/WOSM_VB10_icip/checkpoints/{step}.pth'
                # restore_path = f'./logs/WOSM_VB01_icip/checkpoints/{step}.pth'
                # epoch = 'checkpoint_20'
            else:
                restore_path = f'./logs/WOSM_VB10_icip_SSIM/checkpoints/{step}.pth'
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            print(f"INFO Load Pretrained P-Model From Epoch {pcheckpoint['epoch']}...")
            p_model = LHB_DVC_WOSM_VB()
            p_model.load_state_dict(pcheckpoint["state_dict"])
            p_model.eval()
            p_model.update(force=True)
            p_model.to(device)
            for l in [0, 1, 2]:
                # for interval in np.arange(1., 0., -0.1):
                for interval in np.arange(1., 0., -0.1):
                    log_txt = open(f'{result_save_path}/log_{test_tgt}_{l}_{interval:.2f}_{step}.txt', 'w')

                    if indicator == 'msssim':
                        i_model = ICIP2020ResBVB1(v0=True, psnr=False)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_mssim.pth'
                    else:
                        i_model = ICIP2020ResBVB1(v0=True)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_psnr.pth'
                    icheckpoint = torch.load(i_restore_path, map_location='cpu')
                    print(f"INFO Load Pretrained I-Model From Epoch {icheckpoint['epoch']}...")
                    state_dict = load_pretrained(icheckpoint["state_dict"])
                    i_model.load_state_dict(state_dict)
                    # i_model = mbt2018(l + 5, 'mse', True)
                    i_model.update(force=True)
                    i_model.to(device)
                    i_model.eval()

                    PSNR, MSSSIM, Bits, Bits2l = [], [], [], []
                    iPSNR, iMSSSIM, iBits = [], [], []
                    pPSNR, pMSSSIM, pBits = [], [], []
                    mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
                    mcMSSSIM, warpMSSSIM = [], []
                    iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
                    for ii, seq_info in enumerate(test_info['sequences']):
                        _PSNR, _MSSSIM, _Bits, _Bits2l = [], [], [], []
                        _iPSNR, _iMSSSIM, _iBits = [], [], []
                        _pPSNR, _pMSSSIM, _pBits = [], [], []
                        _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                        _mcMSSSIM, _warpMSSSIM = [], []
                        _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []

                        video_frame_path = os.path.join(test_info['path'], 'PNG_Frames',
                                                        seq_info.replace(test_info['org_resolution'],
                                                                         test_info[resolution_tgt]))
                        images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                        print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                        image = read_image(images[0]).unsqueeze(0)
                        num_pixels = image.size(0) * image.size(2) * image.size(3)
                        feature = None
                        for i, im in enumerate(images):
                            if i >= total_frame_num:
                                break
                            curr_frame = read_image(im).unsqueeze(0).to(device)
                            if i % GOP == 0:
                                feature = None

                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_enc = i_model.compress(curr_frame,
                                                             [i_model.levels - 2 - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_enc = time.perf_counter() - start_time
                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"],
                                                               [i_model.levels - 2 - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_dec = time.perf_counter() - start_time

                                i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                                i_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                                i_ms_ssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()

                                _iPSNR.append(i_psnr)
                                _iMSSSIM.append(i_ms_ssim)
                                _iBits.append(i_bpp)
                                _Bits2l.append(i_bpp)
                                _PSNR.append(i_psnr)
                                _MSSSIM.append(i_ms_ssim)
                                _Bits.append(i_bpp)
                                _iEnc.append(elapsed_enc)
                                _iDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(
                                    f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                                ref_frame = i_out_dec["x_hat"]
                            else:
                                torch.cuda.synchronize()
                                start = time.time()
                                mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame,
                                                                           n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_enc = time.time() - start

                                torch.cuda.synchronize()
                                start = time.time()
                                feature1, dec_p_frame, warped_frame, predict_frame = \
                                    p_model.decompress(ref_frame, mv_out_enc, res_out_enc,
                                                       n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_dec = time.time() - start

                                mse = torch.mean((curr_frame - dec_p_frame).pow(2)).item()
                                p_psnr = 10 * np.log10(1.0 / mse).item()
                                w_mse = torch.mean((curr_frame - warped_frame).pow(2)).item()
                                w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                mc_mse = torch.mean((curr_frame - predict_frame).pow(2)).item()
                                mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                p_ms_ssim = ms_ssim(curr_frame, dec_p_frame, data_range=1.0).item()
                                p_warp_ms_ssim = ms_ssim(curr_frame, warped_frame, data_range=1.0).item()
                                p_mc_ms_ssim = ms_ssim(curr_frame, predict_frame, data_range=1.0).item()
                                res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                p_bpp = mv_bpp + res_bpp

                                ref_frame = dec_p_frame.detach()
                                feature = feature1.detach()

                                _PSNR.append(p_psnr)
                                _MSSSIM.append(p_ms_ssim)
                                _Bits.append(p_bpp)
                                _pPSNR.append(p_psnr)
                                _pMSSSIM.append(p_ms_ssim)
                                _pBits.append(p_bpp)

                                _mcPSNR.append(mc_psnr)
                                _warpPSNR.append(w_psnr)
                                _mcMSSSIM.append(p_mc_ms_ssim)
                                _warpMSSSIM.append(p_warp_ms_ssim)
                                _mvBits.append(mv_bpp)
                                _resBits.append(res_bpp)
                                _Bits2l.append(mv_bpp)

                                _pEnc.append(elapsed_enc)
                                _pDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                      f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                      f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}]"
                                      f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                    f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                    f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}] "
                                    f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                        print(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                        print(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                        print(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                        log_txt.write(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')

                        PSNR.append(np.average(_PSNR))
                        MSSSIM.append(np.average(_MSSSIM))
                        Bits.append(np.average(_Bits))
                        Bits2l.append(np.average(_Bits2l))
                        iPSNR.append(np.average(_iPSNR))
                        iMSSSIM.append(np.average(_iMSSSIM))
                        iBits.append(np.average(_iBits))
                        pPSNR.append(np.average(_pPSNR))
                        pMSSSIM.append(np.average(_pMSSSIM))
                        pBits.append(np.average(_pBits))
                        mcPSNR.append(np.average(_mcPSNR))
                        warpPSNR.append(np.average(_warpPSNR))
                        mvBits.append(np.average(_mvBits))
                        resBits.append(np.average(_resBits))
                        mcMSSSIM.append(np.average(_mcMSSSIM))
                        warpMSSSIM.append(np.average(_warpMSSSIM))
                        iEnc.append(np.average(_iEnc))
                        iDec.append(np.average(_iDec))
                        pEnc.append(np.average(_pEnc))
                        pDec.append(np.average(_pDec))
                        Enc.append(np.average(_Enc))
                        Dec.append(np.average(_Dec))

                    porposed_psnr.append(np.average(PSNR))
                    porposed_bpp.append(np.average(Bits))
                    porposed_bpp2l.append(np.average(Bits2l))
                    porposed_msssim.append(np.average(MSSSIM))
                    porposed_ipsnr.append(np.average(iPSNR))
                    porposed_ibpp.append(np.average(iBits))
                    porposed_imsssim.append(np.average(iMSSSIM))
                    porposed_ppsnr.append(np.average(pPSNR))
                    porposed_pbpp.append(np.average(pBits))
                    porposed_pmsssim.append(np.average(pMSSSIM))

                    porposed_mcpsnr.append(np.average(mcPSNR))
                    porposed_warppsnr.append(np.average(warpPSNR))
                    porposed_mvbpp.append(np.average(mvBits))
                    porposed_resbpp.append(np.average(resBits))
                    porposed_mcmsssim.append(np.average(mcMSSSIM))
                    porposed_warmsssim.append(np.average(warpMSSSIM))
                    porposed_ienc.append(np.average(iEnc))
                    porposed_idec.append(np.average(iDec))
                    porposed_pent.append(np.average(pEnc))
                    porposed_pdec.append(np.average(pDec))
                    porposed_ent.append(np.average(Enc))
                    porposed_dec.append(np.average(Dec))

                log_txt.close()
            print(porposed_bpp)
            print(porposed_psnr)
            print(porposed_msssim)
            results = {
                "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim, "bpp2l": porposed_bpp2l,
                "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
                "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "pmsssim": porposed_pmsssim,
                "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
                "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
                "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
                "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
            }
            output = {
                "name": f'{test_tgt}',
                "description": "Inference (ans)",
                "results": results,
            }
            with open(os.path.join(result_save_path, f'{test_tgt}_{step}.json'), 'w',
                      encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)

            # Bpp1 = [
            #     0.07034640842013888,
            #     0.10103013780381945,
            #     0.14580055519386576,
            #     0.20352964048032407,
            #     0.28302114981192134
            # ]
            # PSNR1 = [
            #     28.76402706280694,
            #     30.19062937067432,
            #     31.67859629351529,
            #     32.96192115190013,
            #     34.12191420336034
            # ]
            # plt.plot(Bpp1, PSNR1, "b--s", label='org_wosm')
            # plt.plot(porposed_bpp, porposed_psnr, "r-o", label='gained_wosm')
            # plt.grid()
            # plt.ylabel("PSNR (dB)")
            # plt.xlabel("bpp (bit/pixel)")
            # plt.legend()
            # plt.savefig(os.path.join(result_save_path, f'{test_tgt}_{step}.png'))
    return None


def wosm_vbi_and_vbp1(indicator, test_tgt):
    device = 'cuda:0'
    test_info = TEST_DATA[test_tgt]
    resolution_tgt = 'x64_resolution'
    GOP = test_info['gop']
    total_frame_num = test_info['frames']
    resolution = test_info[resolution_tgt]
    W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')
    i_l_offset = 3
    if indicator == 'msssim':
        log_path = 'VB_loadMSE_MSSSIM_1.2_30.0_Final'
        _step = 'step_1816000'
    else:
        log_path = 'VB_60_1800_MSE_FINAL'
        _step = 'step_1036000'

    for step in [_step]:
        result_save_path = f'./logs/{log_path}/{indicator.upper()}_i_l_offset{i_l_offset}/{test_tgt}'
        # result_save_path = f'./logs/{log_path}/{indicator.upper()}_{i_l_offset}_{test_tgt}'
        os.makedirs(result_save_path, exist_ok=True)

        porposed_psnr, porposed_bpp, porposed_msssim, porposed_bpp2l = [], [], [], []
        porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
        porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
        porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
        porposed_mcmsssim, porposed_warmsssim = [], []
        porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
        porposed_ent, porposed_dec = [], []
        with torch.no_grad():
            if indicator == 'mse':
                # restore_path = f'./logs/WOSM_VB10_better/WOSM_VB10_icip/checkpoints/{step}.pth'
                restore_path = f'./logs/VB_60_1800_MSE_FINAL/checkpoints/{step}.pth'
                # restore_path = f'./logs/Trained_onA40/WOSM_VB_Step1_80_1280/checkpoints/{step}.pth'
                # restore_path = f'./logs/Trained_onA40/WOSM_VB_Step1_80_2800/{step}.pth'
                # epoch = 'checkpoint_20'
            else:
                # restore_path = f'./logs/WOSM_VB10_icip_SSIM/checkpoints/{step}.pth'
                restore_path = f'./logs/{log_path}/checkpoints/{step}.pth'
                # restore_path = f'./logs/VB_loadMSE_MSSSIM_1.2_35.8_20230707_221355/checkpoints/{step}.pth'
            print(restore_path)
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            print(f"INFO Load Pretrained P-Model From Epoch {pcheckpoint['epoch']}...")
            p_model = LHB_DVC_WOSM_VB()
            p_model.load_state_dict(pcheckpoint["state_dict"])
            p_model.eval()
            p_model.update(force=True)
            p_model.to(device)
            # for l in [0, 1, 2, 3]:
            #     intervals = [1] if l != 3 else [1, 0.05]

            for l in [0, 1, 2, 3]:
                L1 = [1.,  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                L2 = [1.,  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.001]
                intervals = L1 if l != 3 else L2
                # for interval in np.arange(1., 0., -0.1):
                for interval in intervals:
                    log_txt = open(f'{result_save_path}/log_{test_tgt}_{l}_{interval:.2f}_{step}.txt', 'w')

                    if indicator == 'msssim':
                        i_model = ICIP2020ResBVB1(v0=True, psnr=False)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_mssim.pth'
                    else:
                        i_model = ICIP2020ResBVB1(v0=True)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_psnr.pth'
                        # i_restore_path = './logs/ICIP2020ResBVB1_v0True_20230624_204641/checkpoints/checkpoint_16.pth'
                    print(i_restore_path)
                    icheckpoint = torch.load(i_restore_path, map_location='cpu')
                    print(f"INFO Load Pretrained I-Model From Epoch {icheckpoint['epoch']}...")
                    state_dict = load_pretrained(icheckpoint["state_dict"])
                    i_model.load_state_dict(state_dict)
                    # i_model = mbt2018(l + 5, 'mse', True)
                    i_model.update(force=True)
                    i_model.to(device)
                    i_model.eval()

                    PSNR, MSSSIM, Bits, Bits2l = [], [], [], []
                    iPSNR, iMSSSIM, iBits = [], [], []
                    pPSNR, pMSSSIM, pBits = [], [], []
                    mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
                    mcMSSSIM, warpMSSSIM = [], []
                    iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
                    for ii, seq_info in enumerate(test_info['sequences']):
                        _PSNR, _MSSSIM, _Bits, _Bits2l = [], [], [], []
                        _iPSNR, _iMSSSIM, _iBits = [], [], []
                        _pPSNR, _pMSSSIM, _pBits = [], [], []
                        _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                        _mcMSSSIM, _warpMSSSIM = [], []
                        _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []

                        video_frame_path = os.path.join(test_info['path'], 'PNG_Frames',
                                                        seq_info.replace(test_info['org_resolution'],
                                                                         test_info[resolution_tgt]))
                        images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                        print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                        image = read_image(images[0]).unsqueeze(0)
                        num_pixels = image.size(0) * image.size(2) * image.size(3)
                        feature = None
                        for i, im in enumerate(images):
                            if i >= total_frame_num:
                                break
                            curr_frame = read_image(im).unsqueeze(0).to(device)
                            if i % GOP == 0:
                                feature = None

                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_enc = i_model.compress(curr_frame,
                                                             [i_model.levels - i_l_offset - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_enc = time.perf_counter() - start_time
                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"],
                                                               [i_model.levels - i_l_offset - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_dec = time.perf_counter() - start_time

                                i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                                i_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                                i_ms_ssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()

                                _iPSNR.append(i_psnr)
                                _iMSSSIM.append(i_ms_ssim)
                                _iBits.append(i_bpp)
                                _Bits2l.append(i_bpp)
                                _PSNR.append(i_psnr)
                                _MSSSIM.append(i_ms_ssim)
                                _Bits.append(i_bpp)
                                _iEnc.append(elapsed_enc)
                                _iDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(
                                    f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                                ref_frame = i_out_dec["x_hat"]
                            else:
                                torch.cuda.synchronize()
                                start = time.time()
                                mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame,
                                                                           n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_enc = time.time() - start

                                torch.cuda.synchronize()
                                start = time.time()
                                feature1, dec_p_frame, warped_frame, predict_frame = \
                                    p_model.decompress(ref_frame, mv_out_enc, res_out_enc,
                                                       n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_dec = time.time() - start

                                mse = torch.mean((curr_frame - dec_p_frame).pow(2)).item()
                                p_psnr = 10 * np.log10(1.0 / mse).item()
                                w_mse = torch.mean((curr_frame - warped_frame).pow(2)).item()
                                w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                mc_mse = torch.mean((curr_frame - predict_frame).pow(2)).item()
                                mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                p_ms_ssim = ms_ssim(curr_frame, dec_p_frame, data_range=1.0).item()
                                p_warp_ms_ssim = ms_ssim(curr_frame, warped_frame, data_range=1.0).item()
                                p_mc_ms_ssim = ms_ssim(curr_frame, predict_frame, data_range=1.0).item()
                                res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                p_bpp = mv_bpp + res_bpp

                                ref_frame = dec_p_frame.detach()
                                feature = feature1.detach()

                                _PSNR.append(p_psnr)
                                _MSSSIM.append(p_ms_ssim)
                                _Bits.append(p_bpp)
                                _pPSNR.append(p_psnr)
                                _pMSSSIM.append(p_ms_ssim)
                                _pBits.append(p_bpp)

                                _mcPSNR.append(mc_psnr)
                                _warpPSNR.append(w_psnr)
                                _mcMSSSIM.append(p_mc_ms_ssim)
                                _warpMSSSIM.append(p_warp_ms_ssim)
                                _mvBits.append(mv_bpp)
                                _resBits.append(res_bpp)
                                _Bits2l.append(mv_bpp)

                                _pEnc.append(elapsed_enc)
                                _pDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                      f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                      f"| MS-SSIM [{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}|{p_ms_ssim:.3f}]"
                                      f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                    f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                    f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}] "
                                    f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                        print(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                        print(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                        print(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                        log_txt.write(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')

                        PSNR.append(np.average(_PSNR))
                        MSSSIM.append(np.average(_MSSSIM))
                        Bits.append(np.average(_Bits))
                        Bits2l.append(np.average(_Bits2l))
                        iPSNR.append(np.average(_iPSNR))
                        iMSSSIM.append(np.average(_iMSSSIM))
                        iBits.append(np.average(_iBits))
                        pPSNR.append(np.average(_pPSNR))
                        pMSSSIM.append(np.average(_pMSSSIM))
                        pBits.append(np.average(_pBits))
                        mcPSNR.append(np.average(_mcPSNR))
                        warpPSNR.append(np.average(_warpPSNR))
                        mvBits.append(np.average(_mvBits))
                        resBits.append(np.average(_resBits))
                        mcMSSSIM.append(np.average(_mcMSSSIM))
                        warpMSSSIM.append(np.average(_warpMSSSIM))
                        iEnc.append(np.average(_iEnc))
                        iDec.append(np.average(_iDec))
                        pEnc.append(np.average(_pEnc))
                        pDec.append(np.average(_pDec))
                        Enc.append(np.average(_Enc))
                        Dec.append(np.average(_Dec))

                    porposed_psnr.append(np.average(PSNR))
                    porposed_bpp.append(np.average(Bits))
                    porposed_bpp2l.append(np.average(Bits2l))
                    porposed_msssim.append(np.average(MSSSIM))
                    porposed_ipsnr.append(np.average(iPSNR))
                    porposed_ibpp.append(np.average(iBits))
                    porposed_imsssim.append(np.average(iMSSSIM))
                    porposed_ppsnr.append(np.average(pPSNR))
                    porposed_pbpp.append(np.average(pBits))
                    porposed_pmsssim.append(np.average(pMSSSIM))

                    porposed_mcpsnr.append(np.average(mcPSNR))
                    porposed_warppsnr.append(np.average(warpPSNR))
                    porposed_mvbpp.append(np.average(mvBits))
                    porposed_resbpp.append(np.average(resBits))
                    porposed_mcmsssim.append(np.average(mcMSSSIM))
                    porposed_warmsssim.append(np.average(warpMSSSIM))
                    porposed_ienc.append(np.average(iEnc))
                    porposed_idec.append(np.average(iDec))
                    porposed_pent.append(np.average(pEnc))
                    porposed_pdec.append(np.average(pDec))
                    porposed_ent.append(np.average(Enc))
                    porposed_dec.append(np.average(Dec))

                log_txt.close()
            print(porposed_bpp)
            print(porposed_psnr)
            print(porposed_msssim)
            results = {
                "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim, "bpp2l": porposed_bpp2l,
                "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
                "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "pmsssim": porposed_pmsssim,
                "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
                "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
                "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
                "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
            }
            output = {
                "name": f'{test_tgt}',
                "description": "Inference (ans)",
                "results": results,
            }
            with open(os.path.join(result_save_path, f'{test_tgt}_{step}.json'), 'w',
                      encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)

            # Bpp1 = [
            #     0.07034640842013888,
            #     0.10103013780381945,
            #     0.14580055519386576,
            #     0.20352964048032407,
            #     0.28302114981192134
            # ]
            # PSNR1 = [
            #     28.76402706280694,
            #     30.19062937067432,
            #     31.67859629351529,
            #     32.96192115190013,
            #     34.12191420336034
            # ]
            # plt.plot(Bpp1, PSNR1, "b--s", label='org_wosm')
            # plt.plot(porposed_bpp, porposed_psnr, "r-o", label='gained_wosm')
            # plt.grid()
            # plt.ylabel("PSNR (dB)")
            # plt.xlabel("bpp (bit/pixel)")
            # plt.legend()
            # plt.savefig(os.path.join(result_save_path, f'{test_tgt}_{step}.png'))
    return None


def wosm_vbi_and_vbp1_pad(indicator, test_tgt):
    device = 'cuda:0'
    test_info = TEST_DATA_ORG[test_tgt]
    resolution_tgt = 'org_resolution'
    GOP = test_info['gop']
    total_frame_num = test_info['frames']
    resolution = test_info[resolution_tgt]
    W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')
    i_l_offset = 3
    # log_path = 'VB_60_1800_MSE_FINAL'  # step_1036000
    # log_path = 'VB_loadMSE_MSSSIM_1.2_30.0_Final'  # step_1816000
    if indicator == 'msssim':
        log_path = 'VB_loadMSE_MSSSIM_1.2_30.0_Final'
        _step = 'step_1816000'
    else:
        log_path = 'VB_60_1800_MSE_FINAL'
        _step = 'step_1036000'

    for step in [_step]:
        result_save_path = f'./logs/{log_path}/{indicator.upper()}_i_l_offset{i_l_offset}_pad/{test_tgt}'
        # result_save_path = f'./logs/{log_path}/{indicator.upper()}_{i_l_offset}_{test_tgt}'
        os.makedirs(result_save_path, exist_ok=True)

        porposed_psnr, porposed_bpp, porposed_msssim, porposed_bpp2l = [], [], [], []
        porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
        porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
        porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
        porposed_mcmsssim, porposed_warmsssim = [], []
        porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
        porposed_ent, porposed_dec = [], []
        with torch.no_grad():
            if indicator == 'mse':
                # restore_path = f'./logs/WOSM_VB10_better/WOSM_VB10_icip/checkpoints/{step}.pth'
                restore_path = f'./logs/VB_60_1800_MSE_FINAL/checkpoints/{step}.pth'
                # restore_path = f'./logs/Trained_onA40/WOSM_VB_Step1_80_1280/checkpoints/{step}.pth'
                # restore_path = f'./logs/Trained_onA40/WOSM_VB_Step1_80_2800/{step}.pth'
                # epoch = 'checkpoint_20'
            else:
                # restore_path = f'./logs/WOSM_VB10_icip_SSIM/checkpoints/{step}.pth'
                restore_path = f'./logs/{log_path}/checkpoints/{step}.pth'
                # restore_path = f'./logs/VB_loadMSE_MSSSIM_1.2_35.8_20230707_221355/checkpoints/{step}.pth'
            print(restore_path)
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            print(f"INFO Load Pretrained P-Model From Epoch {pcheckpoint['epoch']}...")
            p_model = LHB_DVC_WOSM_VB()
            p_model.load_state_dict(pcheckpoint["state_dict"])
            p_model.eval()
            p_model.update(force=True)
            p_model.to(device)
            # for l in [0, 1, 2, 3]:
            #     intervals = [1] if l != 3 else [1, 0.05]

            for l in [0, 1, 2, 3]:
                L1 = [1.,  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                L2 = [1.,  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.001]
                intervals = L1 if l != 3 else L2
                # for interval in np.arange(1., 0., -0.1):
                for interval in intervals:
                    log_txt = open(f'{result_save_path}/log_{test_tgt}_{l}_{interval:.2f}_{step}.txt', 'w')

                    if indicator == 'msssim':
                        i_model = ICIP2020ResBVB1(v0=True, psnr=False)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_mssim.pth'
                    else:
                        i_model = ICIP2020ResBVB1(v0=True)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_psnr.pth'
                        # i_restore_path = './logs/ICIP2020ResBVB1_v0True_20230624_204641/checkpoints/checkpoint_16.pth'
                    print(i_restore_path)
                    icheckpoint = torch.load(i_restore_path, map_location='cpu')
                    print(f"INFO Load Pretrained I-Model From Epoch {icheckpoint['epoch']}...")
                    state_dict = load_pretrained(icheckpoint["state_dict"])
                    i_model.load_state_dict(state_dict)
                    # i_model = mbt2018(l + 5, 'mse', True)
                    i_model.update(force=True)
                    i_model.to(device)
                    i_model.eval()

                    PSNR, MSSSIM, Bits, Bits2l = [], [], [], []
                    iPSNR, iMSSSIM, iBits = [], [], []
                    pPSNR, pMSSSIM, pBits = [], [], []
                    mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
                    mcMSSSIM, warpMSSSIM = [], []
                    iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
                    for ii, seq_info in enumerate(test_info['sequences']):
                        _PSNR, _MSSSIM, _Bits, _Bits2l = [], [], [], []
                        _iPSNR, _iMSSSIM, _iBits = [], [], []
                        _pPSNR, _pMSSSIM, _pBits = [], [], []
                        _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                        _mcMSSSIM, _warpMSSSIM = [], []
                        _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []

                        video_frame_path = os.path.join(test_info['path'], seq_info)
                        images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                        print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                        # image = read_image(images[0]).unsqueeze(0)
                        # num_pixels = image.size(0) * image.size(2) * image.size(3)
                        image = read_image(images[0]).unsqueeze(0)
                        _, _, org_h, org_w = image.size()
                        feature = None
                        for i, im in enumerate(images):
                            if i >= total_frame_num:
                                break
                            curr_frame_org = read_image(im).unsqueeze(0).to(device)
                            curr_frame = pad(curr_frame_org, 64)
                            num_pixels = curr_frame_org.size(0) * curr_frame_org.size(2) * curr_frame_org.size(3)
                            # curr_frame = read_image(im).unsqueeze(0).to(device)
                            # print(curr_frame_org.shape, curr_frame.shape)
                            # exit()
                            if i % GOP == 0:
                                feature = None

                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_enc = i_model.compress(curr_frame,
                                                             [i_model.levels - i_l_offset - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_enc = time.perf_counter() - start_time
                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"],
                                                               [i_model.levels - i_l_offset - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_dec = time.perf_counter() - start_time

                                i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                                # i_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                                # i_ms_ssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()
                                i_psnr = cal_psnr(curr_frame_org, crop(i_out_dec["x_hat"], (org_h, org_w)))
                                i_ms_ssim = ms_ssim(curr_frame_org, crop(i_out_dec["x_hat"], (org_h, org_w)),
                                                    data_range=1.0).item()

                                _iPSNR.append(i_psnr)
                                _iMSSSIM.append(i_ms_ssim)
                                _iBits.append(i_bpp)
                                _Bits2l.append(i_bpp)
                                _PSNR.append(i_psnr)
                                _MSSSIM.append(i_ms_ssim)
                                _Bits.append(i_bpp)
                                _iEnc.append(elapsed_enc)
                                _iDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(
                                    f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                                ref_frame = i_out_dec["x_hat"]
                            else:
                                torch.cuda.synchronize()
                                start = time.time()
                                mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame,
                                                                           n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_enc = time.time() - start

                                torch.cuda.synchronize()
                                start = time.time()
                                feature1, dec_p_frame, warped_frame, predict_frame = \
                                    p_model.decompress(ref_frame, mv_out_enc, res_out_enc,
                                                       n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_dec = time.time() - start

                                # mse = torch.mean((curr_frame - dec_p_frame).pow(2)).item()
                                # p_psnr = 10 * np.log10(1.0 / mse).item()
                                # w_mse = torch.mean((curr_frame - warped_frame).pow(2)).item()
                                # w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                # mc_mse = torch.mean((curr_frame - predict_frame).pow(2)).item()
                                # mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                # p_ms_ssim = ms_ssim(curr_frame, dec_p_frame, data_range=1.0).item()
                                # p_warp_ms_ssim = ms_ssim(curr_frame, warped_frame, data_range=1.0).item()
                                # p_mc_ms_ssim = ms_ssim(curr_frame, predict_frame, data_range=1.0).item()
                                # res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                # mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                # p_bpp = mv_bpp + res_bpp

                                mse = torch.mean((curr_frame_org - crop(dec_p_frame, (org_h, org_w))).pow(2)).item()
                                p_psnr = 10 * np.log10(1.0 / mse).item()
                                w_mse = torch.mean((curr_frame_org - crop(warped_frame, (org_h, org_w))).pow(2)).item()
                                w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                mc_mse = torch.mean(
                                    (curr_frame_org - crop(predict_frame, (org_h, org_w))).pow(2)).item()
                                mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                p_ms_ssim = ms_ssim(curr_frame_org, crop(dec_p_frame, (org_h, org_w)),
                                                    data_range=1.0).item()
                                p_warp_ms_ssim = ms_ssim(curr_frame_org, crop(warped_frame, (org_h, org_w)),
                                                         data_range=1.0).item()
                                p_mc_ms_ssim = ms_ssim(curr_frame_org, crop(predict_frame, (org_h, org_w)),
                                                       data_range=1.0).item()
                                res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                p_bpp = mv_bpp + res_bpp

                                ref_frame = dec_p_frame.detach()
                                feature = feature1.detach()

                                _PSNR.append(p_psnr)
                                _MSSSIM.append(p_ms_ssim)
                                _Bits.append(p_bpp)
                                _pPSNR.append(p_psnr)
                                _pMSSSIM.append(p_ms_ssim)
                                _pBits.append(p_bpp)

                                _mcPSNR.append(mc_psnr)
                                _warpPSNR.append(w_psnr)
                                _mcMSSSIM.append(p_mc_ms_ssim)
                                _warpMSSSIM.append(p_warp_ms_ssim)
                                _mvBits.append(mv_bpp)
                                _resBits.append(res_bpp)
                                _Bits2l.append(mv_bpp)

                                _pEnc.append(elapsed_enc)
                                _pDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                      f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                      f"| MS-SSIM [{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}|{p_ms_ssim:.3f}]"
                                      f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                    f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                    f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}] "
                                    f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                        print(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                        print(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                        print(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                        log_txt.write(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')

                        PSNR.append(np.average(_PSNR))
                        MSSSIM.append(np.average(_MSSSIM))
                        Bits.append(np.average(_Bits))
                        Bits2l.append(np.average(_Bits2l))
                        iPSNR.append(np.average(_iPSNR))
                        iMSSSIM.append(np.average(_iMSSSIM))
                        iBits.append(np.average(_iBits))
                        pPSNR.append(np.average(_pPSNR))
                        pMSSSIM.append(np.average(_pMSSSIM))
                        pBits.append(np.average(_pBits))
                        mcPSNR.append(np.average(_mcPSNR))
                        warpPSNR.append(np.average(_warpPSNR))
                        mvBits.append(np.average(_mvBits))
                        resBits.append(np.average(_resBits))
                        mcMSSSIM.append(np.average(_mcMSSSIM))
                        warpMSSSIM.append(np.average(_warpMSSSIM))
                        iEnc.append(np.average(_iEnc))
                        iDec.append(np.average(_iDec))
                        pEnc.append(np.average(_pEnc))
                        pDec.append(np.average(_pDec))
                        Enc.append(np.average(_Enc))
                        Dec.append(np.average(_Dec))

                    porposed_psnr.append(np.average(PSNR))
                    porposed_bpp.append(np.average(Bits))
                    porposed_bpp2l.append(np.average(Bits2l))
                    porposed_msssim.append(np.average(MSSSIM))
                    porposed_ipsnr.append(np.average(iPSNR))
                    porposed_ibpp.append(np.average(iBits))
                    porposed_imsssim.append(np.average(iMSSSIM))
                    porposed_ppsnr.append(np.average(pPSNR))
                    porposed_pbpp.append(np.average(pBits))
                    porposed_pmsssim.append(np.average(pMSSSIM))

                    porposed_mcpsnr.append(np.average(mcPSNR))
                    porposed_warppsnr.append(np.average(warpPSNR))
                    porposed_mvbpp.append(np.average(mvBits))
                    porposed_resbpp.append(np.average(resBits))
                    porposed_mcmsssim.append(np.average(mcMSSSIM))
                    porposed_warmsssim.append(np.average(warpMSSSIM))
                    porposed_ienc.append(np.average(iEnc))
                    porposed_idec.append(np.average(iDec))
                    porposed_pent.append(np.average(pEnc))
                    porposed_pdec.append(np.average(pDec))
                    porposed_ent.append(np.average(Enc))
                    porposed_dec.append(np.average(Dec))

                log_txt.close()
            print(porposed_bpp)
            print(porposed_psnr)
            print(porposed_msssim)
            results = {
                "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim, "bpp2l": porposed_bpp2l,
                "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
                "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "pmsssim": porposed_pmsssim,
                "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
                "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
                "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
                "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
            }
            output = {
                "name": f'{test_tgt}',
                "description": "Inference (ans)",
                "results": results,
            }
            with open(os.path.join(result_save_path, f'{test_tgt}_{step}.json'), 'w',
                      encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)

            # Bpp1 = [
            #     0.07034640842013888,
            #     0.10103013780381945,
            #     0.14580055519386576,
            #     0.20352964048032407,
            #     0.28302114981192134
            # ]
            # PSNR1 = [
            #     28.76402706280694,
            #     30.19062937067432,
            #     31.67859629351529,
            #     32.96192115190013,
            #     34.12191420336034
            # ]
            # plt.plot(Bpp1, PSNR1, "b--s", label='org_wosm')
            # plt.plot(porposed_bpp, porposed_psnr, "r-o", label='gained_wosm')
            # plt.grid()
            # plt.ylabel("PSNR (dB)")
            # plt.xlabel("bpp (bit/pixel)")
            # plt.legend()
            # plt.savefig(os.path.join(result_save_path, f'{test_tgt}_{step}.png'))
    return None


def wosm_vbi_and_vbp1_pad_Vtl(indicator, test_tgt):
    device = 'cuda:0'
    resolution_tgt = 'x64_resolution'
    GOP = 12
    total_frame_num = 96
    # resolution = test_info[resolution_tgt]
    # W, H = int(resolution.split('x')[0]), int(resolution.split('x')[1])
    # print(f'Test {test_tgt}, GOP={GOP}, H={H}, W={W}')
    i_l_offset = 3
    log_path = 'VB_loadMSE_MSSSIM_1.2_30.0_Final'
    # log_path = 'VB_60_1800_MSE_FINAL'

    for step in ['step_1816000']:
        result_save_path = f'./logs/{log_path}/{indicator.upper()}_i_l_offset{i_l_offset}_pad/{test_tgt}'
        # result_save_path = f'./logs/{log_path}/{indicator.upper()}_{i_l_offset}_{test_tgt}'
        os.makedirs(result_save_path, exist_ok=True)

        porposed_psnr, porposed_bpp, porposed_msssim, porposed_bpp2l = [], [], [], []
        porposed_ipsnr, porposed_ibpp, porposed_imsssim = [], [], []
        porposed_ppsnr, porposed_pbpp, porposed_pmsssim = [], [], []
        porposed_mcpsnr, porposed_warppsnr, porposed_mvbpp, porposed_resbpp = [], [], [], []
        porposed_mcmsssim, porposed_warmsssim = [], []
        porposed_ienc, porposed_idec, porposed_pent, porposed_pdec = [], [], [], []
        porposed_ent, porposed_dec = [], []
        with torch.no_grad():
            if indicator == 'mse':
                # restore_path = f'./logs/WOSM_VB10_better/WOSM_VB10_icip/checkpoints/{step}.pth'
                restore_path = f'./logs/VB_60_1800_MSE_FINAL/checkpoints/{step}.pth'
                # restore_path = f'./logs/Trained_onA40/WOSM_VB_Step1_80_1280/checkpoints/{step}.pth'
                # restore_path = f'./logs/Trained_onA40/WOSM_VB_Step1_80_2800/{step}.pth'
                # epoch = 'checkpoint_20'
            else:
                # restore_path = f'./logs/WOSM_VB10_icip_SSIM/checkpoints/{step}.pth'
                restore_path = f'./logs/{log_path}/checkpoints/{step}.pth'
                # restore_path = f'./logs/VB_loadMSE_MSSSIM_1.2_35.8_20230707_221355/checkpoints/{step}.pth'
            print(restore_path)
            pcheckpoint = torch.load(restore_path, map_location='cpu')
            print(f"INFO Load Pretrained P-Model From Epoch {pcheckpoint['epoch']}...")
            p_model = LHB_DVC_WOSM_VB()
            p_model.load_state_dict(pcheckpoint["state_dict"])
            p_model.eval()
            p_model.update(force=True)
            p_model.to(device)
            # for l in [0, 1, 2, 3]:
            #     intervals = [1] if l != 3 else [1, 0.05]

            for l in [0, 1, 2, 3]:
                L1 = [1.,  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                L2 = [1.,  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.001]
                intervals = L1 if l != 3 else L2
                # for interval in np.arange(1., 0., -0.1):
                for interval in intervals:
                    log_txt = open(f'{result_save_path}/log_{test_tgt}_{l}_{interval:.2f}_{step}.txt', 'w')

                    if indicator == 'msssim':
                        i_model = ICIP2020ResBVB1(v0=True, psnr=False)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_mssim.pth'
                    else:
                        i_model = ICIP2020ResBVB1(v0=True)
                        i_restore_path = f'./ckpt/ICIP2020ResBVB1_psnr.pth'
                        # i_restore_path = './logs/ICIP2020ResBVB1_v0True_20230624_204641/checkpoints/checkpoint_16.pth'
                    print(i_restore_path)
                    icheckpoint = torch.load(i_restore_path, map_location='cpu')
                    print(f"INFO Load Pretrained I-Model From Epoch {icheckpoint['epoch']}...")
                    state_dict = load_pretrained(icheckpoint["state_dict"])
                    i_model.load_state_dict(state_dict)
                    # i_model = mbt2018(l + 5, 'mse', True)
                    i_model.update(force=True)
                    i_model.to(device)
                    i_model.eval()

                    PSNR, MSSSIM, Bits, Bits2l = [], [], [], []
                    iPSNR, iMSSSIM, iBits = [], [], []
                    pPSNR, pMSSSIM, pBits = [], [], []
                    mcPSNR, warpPSNR, mvBits, resBits = [], [], [], []
                    mcMSSSIM, warpMSSSIM = [], []
                    iEnc, iDec, pEnc, pDec, Enc, Dec = [], [], [], [], [], []
                    seqs = sorted(os.listdir('/tdx/LHB/data/TestSets/VTL/PNG_Frames'))
                    for ii, seq_info in enumerate(seqs):
                        _PSNR, _MSSSIM, _Bits, _Bits2l = [], [], [], []
                        _iPSNR, _iMSSSIM, _iBits = [], [], []
                        _pPSNR, _pMSSSIM, _pBits = [], [], []
                        _mcPSNR, _warpPSNR, _mvBits, _resBits = [], [], [], []
                        _mcMSSSIM, _warpMSSSIM = [], []
                        _iEnc, _iDec, _pEnc, _pDec, _Enc, _Dec = [], [], [], [], [], []
                        video_frame_path = os.path.join('/tdx/LHB/data/TestSets/VTL/PNG_Frames', seq_info)

                        images = sorted(glob.glob(os.path.join(video_frame_path, '*.png')))
                        print(f'INFO Process {seq_info}, Find {len(images)} images, Default test frames {total_frame_num}')
                        # image = read_image(images[0]).unsqueeze(0)
                        # num_pixels = image.size(0) * image.size(2) * image.size(3)
                        image = read_image(images[0]).unsqueeze(0)
                        _, _, org_h, org_w = image.size()
                        feature = None
                        for i, im in enumerate(images):
                            if i >= total_frame_num:
                                break
                            curr_frame_org = read_image(im).unsqueeze(0).to(device)
                            curr_frame = pad(curr_frame_org, 64)
                            num_pixels = curr_frame_org.size(0) * curr_frame_org.size(2) * curr_frame_org.size(3)
                            # curr_frame = read_image(im).unsqueeze(0).to(device)
                            if i % GOP == 0:
                                feature = None

                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_enc = i_model.compress(curr_frame,
                                                             [i_model.levels - i_l_offset - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_enc = time.perf_counter() - start_time
                                torch.cuda.synchronize()
                                start_time = time.perf_counter()
                                i_out_dec = i_model.decompress(i_out_enc["strings"], i_out_enc["shape"],
                                                               [i_model.levels - i_l_offset - l], 1.0 - interval)
                                torch.cuda.synchronize()
                                elapsed_dec = time.perf_counter() - start_time

                                i_bpp = sum(len(s[0]) for s in i_out_enc["strings"]) * 8.0 / num_pixels
                                # i_psnr = cal_psnr(curr_frame, i_out_dec["x_hat"])
                                # i_ms_ssim = ms_ssim(curr_frame, i_out_dec["x_hat"], data_range=1.0).item()
                                i_psnr = cal_psnr(curr_frame_org, crop(i_out_dec["x_hat"], (org_h, org_w)))
                                i_ms_ssim = ms_ssim(curr_frame_org, crop(i_out_dec["x_hat"], (org_h, org_w)),
                                                    data_range=1.0).item()

                                _iPSNR.append(i_psnr)
                                _iMSSSIM.append(i_ms_ssim)
                                _iBits.append(i_bpp)
                                _Bits2l.append(i_bpp)
                                _PSNR.append(i_psnr)
                                _MSSSIM.append(i_ms_ssim)
                                _Bits.append(i_bpp)
                                _iEnc.append(elapsed_enc)
                                _iDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(
                                    f"i={i}, {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"i={i} {seq_info} I-Frame | bpp {i_bpp:.3f} | PSNR {i_psnr:.3f} | MS-SSIM {i_ms_ssim:.3f} | Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                                ref_frame = i_out_dec["x_hat"]
                            else:
                                torch.cuda.synchronize()
                                start = time.time()
                                mv_out_enc, res_out_enc = p_model.compress(ref_frame, curr_frame,
                                                                           n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_enc = time.time() - start

                                torch.cuda.synchronize()
                                start = time.time()
                                feature1, dec_p_frame, warped_frame, predict_frame = \
                                    p_model.decompress(ref_frame, mv_out_enc, res_out_enc,
                                                       n=[l], l=interval, feature=feature)
                                torch.cuda.synchronize()
                                elapsed_dec = time.time() - start

                                # mse = torch.mean((curr_frame - dec_p_frame).pow(2)).item()
                                # p_psnr = 10 * np.log10(1.0 / mse).item()
                                # w_mse = torch.mean((curr_frame - warped_frame).pow(2)).item()
                                # w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                # mc_mse = torch.mean((curr_frame - predict_frame).pow(2)).item()
                                # mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                # p_ms_ssim = ms_ssim(curr_frame, dec_p_frame, data_range=1.0).item()
                                # p_warp_ms_ssim = ms_ssim(curr_frame, warped_frame, data_range=1.0).item()
                                # p_mc_ms_ssim = ms_ssim(curr_frame, predict_frame, data_range=1.0).item()
                                # res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                # mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                # p_bpp = mv_bpp + res_bpp

                                mse = torch.mean((curr_frame_org - crop(dec_p_frame, (org_h, org_w))).pow(2)).item()
                                p_psnr = 10 * np.log10(1.0 / mse).item()
                                w_mse = torch.mean((curr_frame_org - crop(warped_frame, (org_h, org_w))).pow(2)).item()
                                w_psnr = 10 * np.log10(1.0 / w_mse).item()
                                mc_mse = torch.mean(
                                    (curr_frame_org - crop(predict_frame, (org_h, org_w))).pow(2)).item()
                                mc_psnr = 10 * np.log10(1.0 / mc_mse).item()
                                p_ms_ssim = ms_ssim(curr_frame_org, crop(dec_p_frame, (org_h, org_w)),
                                                    data_range=1.0).item()
                                p_warp_ms_ssim = ms_ssim(curr_frame_org, crop(warped_frame, (org_h, org_w)),
                                                         data_range=1.0).item()
                                p_mc_ms_ssim = ms_ssim(curr_frame_org, crop(predict_frame, (org_h, org_w)),
                                                       data_range=1.0).item()
                                res_bpp = sum(len(s[0]) for s in res_out_enc["strings"]) * 8.0 / num_pixels
                                mv_bpp = sum(len(s[0]) for s in mv_out_enc["strings"]) * 8.0 / num_pixels
                                p_bpp = mv_bpp + res_bpp

                                ref_frame = dec_p_frame.detach()
                                feature = feature1.detach()

                                _PSNR.append(p_psnr)
                                _MSSSIM.append(p_ms_ssim)
                                _Bits.append(p_bpp)
                                _pPSNR.append(p_psnr)
                                _pMSSSIM.append(p_ms_ssim)
                                _pBits.append(p_bpp)

                                _mcPSNR.append(mc_psnr)
                                _warpPSNR.append(w_psnr)
                                _mcMSSSIM.append(p_mc_ms_ssim)
                                _warpMSSSIM.append(p_warp_ms_ssim)
                                _mvBits.append(mv_bpp)
                                _resBits.append(res_bpp)
                                _Bits2l.append(mv_bpp)

                                _pEnc.append(elapsed_enc)
                                _pDec.append(elapsed_dec)
                                _Enc.append(elapsed_enc)
                                _Dec.append(elapsed_dec)
                                print(f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                      f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                      f"| MS-SSIM [{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}|{p_ms_ssim:.3f}]"
                                      f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s ")
                                log_txt.write(
                                    f"{l, interval}, i={i}, {seq_info} P-Frame | bpp [{mv_bpp:.3f}, {res_bpp:.4f}, {p_bpp:.3f}] "
                                    f"| PSNR [{w_psnr:.3f}|{mc_psnr:.3f}|{p_psnr:.3f}] "
                                    f"| MS-SSIM [{p_ms_ssim:.3f}|{p_warp_ms_ssim:.3f}|{p_mc_ms_ssim:.3f}] "
                                    f"| Encoded in {elapsed_enc:.3f}s | Decoded in {elapsed_dec:.3f}s\n")
                                log_txt.flush()

                        print(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}')
                        print(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}')
                        print(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}')

                        log_txt.write(f'{l, interval}, I-Frame | Average BPP {np.average(_iBits):.4f} | PSRN {np.average(_iPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, P-Frame | Average BPP {np.average(_pBits):.4f} | PSRN {np.average(_pPSNR):.4f}\n')
                        log_txt.write(f'{l, interval}, Frame | Average BPP {np.average(_Bits):.4f} | PSRN {np.average(_PSNR):.4f}\n')

                        PSNR.append(np.average(_PSNR))
                        MSSSIM.append(np.average(_MSSSIM))
                        Bits.append(np.average(_Bits))
                        Bits2l.append(np.average(_Bits2l))
                        iPSNR.append(np.average(_iPSNR))
                        iMSSSIM.append(np.average(_iMSSSIM))
                        iBits.append(np.average(_iBits))
                        pPSNR.append(np.average(_pPSNR))
                        pMSSSIM.append(np.average(_pMSSSIM))
                        pBits.append(np.average(_pBits))
                        mcPSNR.append(np.average(_mcPSNR))
                        warpPSNR.append(np.average(_warpPSNR))
                        mvBits.append(np.average(_mvBits))
                        resBits.append(np.average(_resBits))
                        mcMSSSIM.append(np.average(_mcMSSSIM))
                        warpMSSSIM.append(np.average(_warpMSSSIM))
                        iEnc.append(np.average(_iEnc))
                        iDec.append(np.average(_iDec))
                        pEnc.append(np.average(_pEnc))
                        pDec.append(np.average(_pDec))
                        Enc.append(np.average(_Enc))
                        Dec.append(np.average(_Dec))

                    porposed_psnr.append(np.average(PSNR))
                    porposed_bpp.append(np.average(Bits))
                    porposed_bpp2l.append(np.average(Bits2l))
                    porposed_msssim.append(np.average(MSSSIM))
                    porposed_ipsnr.append(np.average(iPSNR))
                    porposed_ibpp.append(np.average(iBits))
                    porposed_imsssim.append(np.average(iMSSSIM))
                    porposed_ppsnr.append(np.average(pPSNR))
                    porposed_pbpp.append(np.average(pBits))
                    porposed_pmsssim.append(np.average(pMSSSIM))

                    porposed_mcpsnr.append(np.average(mcPSNR))
                    porposed_warppsnr.append(np.average(warpPSNR))
                    porposed_mvbpp.append(np.average(mvBits))
                    porposed_resbpp.append(np.average(resBits))
                    porposed_mcmsssim.append(np.average(mcMSSSIM))
                    porposed_warmsssim.append(np.average(warpMSSSIM))
                    porposed_ienc.append(np.average(iEnc))
                    porposed_idec.append(np.average(iDec))
                    porposed_pent.append(np.average(pEnc))
                    porposed_pdec.append(np.average(pDec))
                    porposed_ent.append(np.average(Enc))
                    porposed_dec.append(np.average(Dec))

                log_txt.close()
            print(porposed_bpp)
            print(porposed_psnr)
            print(porposed_msssim)
            results = {
                "psnr": porposed_psnr, "bpp": porposed_bpp, "msssim": porposed_msssim, "bpp2l": porposed_bpp2l,
                "ipsnr": porposed_ipsnr, "ibpp": porposed_ibpp, "imsssim": porposed_imsssim,
                "ppsnr": porposed_ppsnr, "pbpp": porposed_pbpp, "pmsssim": porposed_pmsssim,
                "mcpsnr": porposed_mcpsnr, "warppsnr": porposed_warppsnr, "mvbpp": porposed_mvbpp,
                "resbpp": porposed_resbpp, "mcmsssim": porposed_mcmsssim, "warmsssim": porposed_warmsssim,
                "ienc": porposed_ienc, "idec": porposed_idec, "pent": porposed_pent,
                "pdec": porposed_pdec, "ent": porposed_ent, "dec": porposed_dec,
            }
            output = {
                "name": f'{test_tgt}',
                "description": "Inference (ans)",
                "results": results,
            }
            with open(os.path.join(result_save_path, f'{test_tgt}_{step}.json'), 'w',
                      encoding='utf-8') as json_file:
                json.dump(output, json_file, indent=2)

            # Bpp1 = [
            #     0.07034640842013888,
            #     0.10103013780381945,
            #     0.14580055519386576,
            #     0.20352964048032407,
            #     0.28302114981192134
            # ]
            # PSNR1 = [
            #     28.76402706280694,
            #     30.19062937067432,
            #     31.67859629351529,
            #     32.96192115190013,
            #     34.12191420336034
            # ]
            # plt.plot(Bpp1, PSNR1, "b--s", label='org_wosm')
            # plt.plot(porposed_bpp, porposed_psnr, "r-o", label='gained_wosm')
            # plt.grid()
            # plt.ylabel("PSNR (dB)")
            # plt.xlabel("bpp (bit/pixel)")
            # plt.legend()
            # plt.savefig(os.path.join(result_save_path, f'{test_tgt}_{step}.png'))
    return None


if __name__ == "__main__":
    pass
    # HEVC_B  HEVC_C  HEVC_D  HEVC_E  UVG  MCL-JCV
    # wosm_vbi_and_vbp1('msssim', 'HEVC_D')
    # wosm_vbi_and_vbp1('mse', 'VTL')
    # wosm_vbi_and_vbp1_pad('msssim', 'VTL')
    # wosm_vbi_and_vbp1_pad('mse', 'VTL')

    # wosm_vbi_and_vbp1('msssim', 'HEVC_B')

    # for tgt in ['HEVC_D', 'HEVC_B', 'HEVC_C', 'HEVC_E', 'UVG']:
    #     wosm_vbi_and_vbp1_pad('msssim', tgt)
    # wosm_vbi_and_vbp1_pad('mse', 'MCL-JCV')

    # for indicator in ['mse', 'msssim']:
    #     for tgt in ['HEVC_B', 'HEVC_C', 'HEVC_D', 'HEVC_E', 'UVG', 'MCL-JCV']:
    #         get_result_wosm(indicator, tgt)

# -*- coding: utf-8 -*-
import math
import numpy as np

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck

from modules import ME_Spynet, MyMCNetWOsm, RefineNet, FeatureExtraction, \
    MyRecNetWOsm, torch_warp, MyRecNet, MyMCNet
from image_model import ChannelSplitICIP2020ResB, ChannelSplitICIP2020ResBGain
from pytorch_msssim import ms_ssim


# 23218033
class LHB_DVC_WOSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.opticFlow = ME_Spynet()
        self.mv_codec = ChannelSplitICIP2020ResB(8, 64, 2)
        self.res_codec = ChannelSplitICIP2020ResB(64 + 6, 96, 64)
        self.MC = MyMCNetWOsm()

        self.RefineMvNet = RefineNet(5, 64, 2)
        self.RefineResiNet = RefineNet(64 + 3, 64, 64)

        self.FeatureExtractor = FeatureExtraction(3, 64)
        self.enhance = MyRecNetWOsm(32 + 64, 64, 3, return_fea=True)

    def forward(self, ref_frame, curr_frame, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_loss = torch.mean((warped_frame - curr_frame).pow(2))
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)
        mc_loss = torch.mean((predict_frame - curr_frame).pow(2))

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1))
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        # print(predict_frame.shape, recon_res.shape)
        # exit()

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        # feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        # print(feature.shape, recon_image.shape)
        # exit()
        # distortion
        mse_loss = torch.mean((recon_image - curr_frame).pow(2))
        bpp = bpp_mv + bpp_res

        return recon_image, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp

    def compress(self, ref_frame, curr_frame, feature=None):
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_out_enc = self.mv_codec.compress(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"])['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea

        res_out_enc = self.res_codec.compress(torch.cat([ref_frame, res, predict_frame], 1))
        return mv_out_enc, res_out_enc

    def decompress(self, ref_frame, mv_out_enc, res_out_enc, feature):
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"])['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        recon_res = self.res_codec.decompress(res_out_enc["strings"], res_out_enc["shape"])['x_hat']
        recon_res = self.RefineResiNet(recon_res, ref_frame)

        recon_image_fea = predict_frame_fea + recon_res
        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))

        return feature, recon_image.clamp(0., 1.), warped_frame.clamp(0., 1.), predict_frame.clamp(0., 1.)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def mv_aux_loss(self):
        return sum(m.loss() for m in self.mv_codec.modules() if isinstance(m, EntropyBottleneck))

    def res_aux_loss(self):
        return sum(m.loss() for m in self.res_codec.modules() if isinstance(m, EntropyBottleneck))

    def update(self, force=False):
        updated = self.mv_codec.update(force=force)
        updated |= self.res_codec.update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        mv_codec_dict = {k[len('mv_codec.'):]: v for k, v in state_dict.items() if 'mv_codec' in k}
        res_codec_dict = {k[len('res_codec.'):]: v for k, v in state_dict.items() if 'res_codec' in k}

        self.mv_codec.load_state_dict(mv_codec_dict)
        self.res_codec.load_state_dict(res_codec_dict)

        super().load_state_dict(state_dict)


class LHB_DVC_WOSM_VB(nn.Module):
    def __init__(self, levels=5):
        super().__init__()
        self.opticFlow = ME_Spynet()
        self.mv_codec = ChannelSplitICIP2020ResBGain(8, 64, 2, levels)
        self.res_codec = ChannelSplitICIP2020ResBGain(64 + 6, 96, 64, levels)
        self.MC = MyMCNetWOsm()

        self.RefineMvNet = RefineNet(5, 64, 2)
        self.RefineResiNet = RefineNet(64 + 3, 64, 64)

        self.FeatureExtractor = FeatureExtraction(3, 64)
        self.enhance = MyRecNetWOsm(32 + 64, 64, 3, return_fea=True)

    def forward(self, ref_frame, curr_frame, n, l=1, feature=None, avg_dim=(1, 2, 3)):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1), n, l)
        recon_mv1 = mv_enc_out['x_hat']
        # print(recon_mv1.shape)
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_loss = torch.mean((warped_frame - curr_frame).pow(2), dim=avg_dim)
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)
        mc_loss = torch.mean((predict_frame - curr_frame).pow(2), dim=avg_dim)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1), n, l)
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        # print(predict_frame.shape, recon_res.shape)
        # exit()

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        # print(feature.shape, recon_image.shape)
        # exit()
        # distortion
        mse_loss = torch.mean((recon_image - curr_frame).pow(2), dim=avg_dim)
        bpp = bpp_mv + bpp_res

        return recon_image, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp

    def forward_msssim(self, ref_frame, curr_frame, n, l=1, feature=None, avg_dim=(1, 2, 3)):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1), n, l)
        recon_mv1 = mv_enc_out['x_hat']
        # print(recon_mv1.shape)
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_msssim = ms_ssim(warped_frame, curr_frame, 1.0)
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)
        mc_msssim = ms_ssim(predict_frame, curr_frame, 1.0)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1), n, l)
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        # print(predict_frame.shape, recon_res.shape)
        # exit()

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))
        # print(feature.shape, recon_image.shape)
        # exit()
        # distortion
        msssim = ms_ssim(recon_image, curr_frame, 1.0)
        bpp = bpp_mv + bpp_res

        return recon_image, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp

    def compress(self, ref_frame, curr_frame, n, l=1, feature=None):
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_out_enc = self.mv_codec.compress(torch.cat([curr_frame, estimated_mv, ref_frame], 1), n, l)
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"], n, l)['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea

        res_out_enc = self.res_codec.compress(torch.cat([ref_frame, res, predict_frame], 1), n, l)
        return mv_out_enc, res_out_enc

    def decompress(self, ref_frame, mv_out_enc, res_out_enc, n, l=1, feature=None):
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"], n, l)['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        recon_res = self.res_codec.decompress(res_out_enc["strings"], res_out_enc["shape"], n, l)['x_hat']
        recon_res = self.RefineResiNet(recon_res, ref_frame)

        recon_image_fea = predict_frame_fea + recon_res
        feature, recon_image = self.enhance(torch.cat([recon_image_fea, warp_fea], 1))

        return feature, recon_image.clamp(0., 1.), warped_frame.clamp(0., 1.), predict_frame.clamp(0., 1.)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def mv_aux_loss(self):
        return sum(m.loss() for m in self.mv_codec.modules() if isinstance(m, EntropyBottleneck))

    def res_aux_loss(self):
        return sum(m.loss() for m in self.res_codec.modules() if isinstance(m, EntropyBottleneck))

    def update(self, force=False):
        updated = self.mv_codec.update(force=force)
        updated |= self.res_codec.update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        mv_codec_dict = {k[len('mv_codec.'):]: v for k, v in state_dict.items() if 'mv_codec' in k}
        res_codec_dict = {k[len('res_codec.'):]: v for k, v in state_dict.items() if 'res_codec' in k}

        self.mv_codec.load_state_dict(mv_codec_dict)
        self.res_codec.load_state_dict(res_codec_dict)

        super().load_state_dict(state_dict)


# 25431409 -> 32521009
class LHB_DVC(nn.Module):
    def __init__(self, L=False):
        super().__init__()
        if not L:
            # 25431409
            self.opticFlow = ME_Spynet()
            self.mv_codec = ChannelSplitICIP2020ResB(8, 64, 2)
            self.res_codec = ChannelSplitICIP2020ResB(64 + 6, 96, 64)
            self.MC = MyMCNet(in_ch=3, hidden=64, up_out=32, out_ch=3)

            self.RefineMvNet = RefineNet(5, 64, 2)
            self.RefineResiNet = RefineNet(64 + 3, 64, 64)

            self.FeatureExtractor = FeatureExtraction(3, 64)
            self.enhance = MyRecNet(32 + 64, 64, 3, return_fea=True)
        else:
            self.opticFlow = ME_Spynet()
            self.mv_codec = ChannelSplitICIP2020ResB(8, 72, 2)
            self.res_codec = ChannelSplitICIP2020ResB(96 + 6, 128, 96)
            self.MC = MyMCNet(in_ch=3, hidden=64, up_out=64, out_ch=3, fea_ch=128)

            self.RefineMvNet = RefineNet(5, 64, 2)
            self.RefineResiNet = RefineNet(96 + 3, 96, 96)

            self.FeatureExtractor = FeatureExtraction(3, 96)
            self.enhance = MyRecNet(96 + 64, 128, 3, return_fea=True)

    def forward(self, ref_frame, curr_frame, sm_fea, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_loss = torch.mean((warped_frame - curr_frame).pow(2))
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        # MC_input = torch.cat([ref_frame, warped_frame], dim=1)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)
        mc_loss = torch.mean((predict_frame - curr_frame).pow(2))

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1))
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)
        # print(predict_frame.shape, recon_res.shape)
        # exit()

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(recon_image_fea, warp_fea, sm_fea)
        print(feature.shape, recon_image.shape)
        exit()
        mse_loss = torch.mean((recon_image - curr_frame).pow(2))
        bpp = bpp_mv + bpp_res

        return recon_image, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp

    def forward_msssim(self, ref_frame, curr_frame, sm_fea, feature=None):
        pixels = np.prod(curr_frame.size()) // curr_frame.size()[1]

        # motion estimation
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_enc_out = self.mv_codec(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv1 = mv_enc_out['x_hat']
        recon_mv = self.RefineMvNet(recon_mv1, ref_frame)

        # motion compensation
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_msssim = ms_ssim(warped_frame, curr_frame, data_range=1.0)
        bpp_mv = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in mv_enc_out["likelihoods"].values()
        )

        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)
        mc_msssim = ms_ssim(predict_frame, curr_frame, data_range=1.0)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea
        res_enc_out = self.res_codec(torch.cat([ref_frame, res, predict_frame], 1))
        recon_res1 = res_enc_out['x_hat']
        bpp_res = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * pixels))
            for likelihoods in res_enc_out["likelihoods"].values()
        )
        recon_res = self.RefineResiNet(recon_res1, ref_frame)

        recon_image_fea = predict_frame_fea + recon_res

        feature, recon_image = self.enhance(recon_image_fea, warp_fea, sm_fea)
        msssim = ms_ssim(recon_image, curr_frame, data_range=1.0)
        bpp = bpp_mv + bpp_res

        return recon_image, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp

    def compress(self, ref_frame, curr_frame, sm_fea, feature=None):
        estimated_mv = self.opticFlow(curr_frame, ref_frame)
        mv_out_enc = self.mv_codec.compress(torch.cat([curr_frame, estimated_mv, ref_frame], 1))
        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"])['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        curr_frame_fea = self.FeatureExtractor(curr_frame)
        res = curr_frame_fea - predict_frame_fea

        res_out_enc = self.res_codec.compress(torch.cat([ref_frame, res, predict_frame], 1))
        return mv_out_enc, res_out_enc

    def decompress(self, ref_frame, mv_out_enc, res_out_enc, sm_fea, feature):

        recon_mv = self.mv_codec.decompress(mv_out_enc["strings"], mv_out_enc["shape"])['x_hat']
        recon_mv = self.RefineMvNet(recon_mv, ref_frame)
        warped_frame = torch_warp(ref_frame, recon_mv)
        warp_fea, predict_frame = self.MC(ref_frame, warped_frame, recon_mv, sm_fea, feature)

        predict_frame_fea = self.FeatureExtractor(predict_frame)
        recon_res = self.res_codec.decompress(res_out_enc["strings"], res_out_enc["shape"])['x_hat']
        recon_res = self.RefineResiNet(recon_res, ref_frame)

        recon_image_fea = predict_frame_fea + recon_res
        feature, recon_image = self.enhance(recon_image_fea, warp_fea, sm_fea)

        return feature, recon_image.clamp(0., 1.), warped_frame.clamp(0., 1.), predict_frame.clamp(0., 1.)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def mv_aux_loss(self):
        return sum(m.loss() for m in self.mv_codec.modules() if isinstance(m, EntropyBottleneck))

    def res_aux_loss(self):
        return sum(m.loss() for m in self.res_codec.modules() if isinstance(m, EntropyBottleneck))

    def update(self, force=False):
        updated = self.mv_codec.update(force=force)
        updated |= self.res_codec.update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        mv_codec_dict = {k[len('mv_codec.'):]: v for k, v in state_dict.items() if 'mv_codec' in k}
        res_codec_dict = {k[len('res_codec.'):]: v for k, v in state_dict.items() if 'res_codec' in k}

        self.mv_codec.load_state_dict(mv_codec_dict)
        self.res_codec.load_state_dict(res_codec_dict)

        super().load_state_dict(state_dict)


if __name__ == "__main__":
    # print(1280 * 1.7258e-4, 2.0525e-4 * 640, 5.14e-4 * 320, 5.13e-4 * 160, 80 * 9.4768e-5)
    # exit()
    h, w = 256, 256

    model = LHB_DVC_WOSM_VB()
    print(f'[*] Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # exit()
    curr, ref = torch.rand((5, 3, h, w)), torch.rand((5, 3, h, w))
    fea = torch.rand((1, 256, h // 4, w // 4))
    fea1 = torch.rand((5, 64, h, w))
    out = model(curr, ref, n=torch.arange(0, 5), l=1, feature=fea1)

# -*- coding: utf-8 -*-
import os
import warnings
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import subpel_conv3x3, ResidualBlock, conv3x3, \
    GDN, ResidualBlockUpsample, ResidualBlockWithStride, AttentionBlock
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np
import random

from typing import Any
import time
from torch import Tensor

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

# [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]
# [0.1800, 0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]


class LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class GainModule(nn.Module):
    def __init__(self, n=3, N=128):
        super(GainModule, self).__init__()
        self.gain_matrix = nn.Parameter(torch.ones(n, N))

    def forward(self, x, n=None, l=1):
        if l != 1:
            gain1 = self.gain_matrix[n]
            gain2 = self.gain_matrix[n + 1]
            gain = (torch.abs(gain1) ** l) * (torch.abs(gain2) ** (1 - l))
            gain = gain.squeeze(0)
            # print(11, gain.shape)
            # exit()
        else:
            gain = torch.abs(self.gain_matrix[n])
            # print(22, gain.shape)
            # exit()
            # reshaped_gain = gain.unsqueeze(2).unsqueeze(3)

        reshaped_gain = gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # print(x.shape, reshaped_gain.shape, gain.shape)
        # exit()
        rescaled_latent = reshaped_gain * x
        return rescaled_latent


class GainModule0(nn.Module):
    def __init__(self, n=3, N=128):
        super(GainModule0, self).__init__()
        self.gain_matrix = nn.Parameter(torch.ones(n, N))

    def forward(self, x, n=None, l=1):
        if l != 1:
            gain1 = self.gain_matrix[n]
            gain2 = self.gain_matrix[n[0] + 1]
            gain = (torch.abs(gain1) ** l) * (torch.abs(gain2) ** (1 - l))
        else:
            gain = torch.abs(self.gain_matrix[n])

        reshaped_gain = gain.unsqueeze(2).unsqueeze(3)
        rescaled_latent = reshaped_gain * x
        return rescaled_latent


class FeatureModulation(nn.Module):
    def __init__(self, n=3, N=128):
        super(FeatureModulation, self).__init__()
        self.Linear1 = nn.Linear(n, N, bias=False)
        torch.nn.init.ones_(self.Linear1.weight)
        # self.Linear2 = nn.Linear(N // 2, N)
        # self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.Softplus()

    def forward(self, x_input, one_hot):
        mask = self.Linear1(one_hot)
        # mask = self.act1(mask)
        # mask = self.Linear2(mask)
        mask = self.act2(mask)
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x_input * mask


class g_a(nn.Module):
    def __init__(self, in_ch=3, hidden=128, out_ch=128, level=10):
        super(g_a, self).__init__()
        self.conv1 = conv(in_ch, hidden)
        self.conv2 = conv(hidden, hidden)
        self.conv3 = conv(hidden, hidden)
        self.conv4 = conv(hidden, out_ch)
        self.act1 = GDN(hidden)
        self.act2 = GDN(hidden)
        self.act3 = GDN(hidden)

        self.modulation1 = FeatureModulation(level, hidden)
        self.modulation2 = FeatureModulation(level, hidden)
        self.modulation3 = FeatureModulation(level, hidden)

    def forward(self, x_input, one_hot):
        out = self.act1(self.modulation1(self.conv1(x_input), one_hot))
        out = self.act2(self.modulation2(self.conv2(out), one_hot))
        out = self.act3(self.modulation3(self.conv3(out), one_hot))
        return self.conv4(out)


class g_s(nn.Module):
    def __init__(self, in_ch=320, hidden=192, out_ch=3, level=10):
        super(g_s, self).__init__()
        self.conv1 = deconv(in_ch, hidden)
        self.conv2 = deconv(hidden, hidden)
        self.conv3 = deconv(hidden, hidden)
        self.conv4 = deconv(hidden, out_ch)
        self.act1 = GDN(hidden)
        self.act2 = GDN(hidden)
        self.act3 = GDN(hidden)

        self.modulation1 = FeatureModulation(level, hidden)
        self.modulation2 = FeatureModulation(level, hidden)
        self.modulation3 = FeatureModulation(level, hidden)

    def forward(self, x_input, one_hot):
        out = self.act1(self.modulation1(self.conv1(x_input), one_hot))
        out = self.act2(self.modulation2(self.conv2(out), one_hot))
        out = self.act3(self.modulation3(self.conv3(out), one_hot))
        return self.conv4(out)


class g_a1(nn.Module):
    def __init__(self, in_ch=3, hidden=128, out_ch=128, level=10):
        super(g_a1, self).__init__()

        self.conv1 = nn.Sequential(
            ResidualBlockWithStride(in_ch, hidden, stride=2),
            ResidualBlock(hidden, hidden),
        )
        self.conv2 = nn.Sequential(
            ResidualBlockWithStride(hidden, hidden, stride=2),
            ResidualBlock(hidden, hidden),
        )
        self.conv3 = nn.Sequential(
            ResidualBlockWithStride(hidden, hidden, stride=2),
            ResidualBlock(hidden, hidden),
        )
        self.conv4 = conv3x3(hidden, out_ch, stride=2)
        self.act = nn.LeakyReLU(inplace=True)

        self.modulation1 = FeatureModulation(level, hidden)
        self.modulation2 = FeatureModulation(level, hidden)
        self.modulation3 = FeatureModulation(level, hidden)

    def forward(self, x_input, one_hot):
        out = self.act(self.modulation1(self.conv1(x_input), one_hot))
        out = self.act(self.modulation2(self.conv2(out), one_hot))
        out = self.act(self.modulation3(self.conv3(out), one_hot))
        return self.conv4(out)


class g_s1(nn.Module):
    def __init__(self, in_ch=320, hidden=192, out_ch=3, level=10):
        super(g_s1, self).__init__()

        self.conv1 = nn.Sequential(
            ResidualBlock(in_ch, hidden),
            ResidualBlockUpsample(hidden, hidden, 2),
        )
        self.conv2 = nn.Sequential(
            ResidualBlock(hidden, hidden),
            ResidualBlockUpsample(hidden, hidden, 2),
        )
        self.conv3 = nn.Sequential(
            ResidualBlock(hidden, hidden),
            ResidualBlockUpsample(hidden, hidden, 2),
        )
        self.conv4 = nn.Sequential(
            ResidualBlock(hidden, hidden),
            subpel_conv3x3(hidden, out_ch, 2),
        )
        self.act = nn.LeakyReLU(inplace=True)

        self.modulation1 = FeatureModulation(level, hidden)
        self.modulation2 = FeatureModulation(level, hidden)
        self.modulation3 = FeatureModulation(level, hidden)

    def forward(self, x_input, one_hot):
        out = self.act(self.modulation1(self.conv1(x_input)), one_hot)
        out = self.act(self.modulation2(self.conv2(out)), one_hot)
        out = self.act(self.modulation3(self.conv3(out)), one_hot)
        return self.conv4(out)


# 17569891
class GainedMeanScale(nn.Module):
    def __init__(self, N=192, M=320, v0=False):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        # # [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]
        # [0.1800, 0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        # self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005]  # mxh add from HUAWEI CVPR2021 Gained...
        self.lmbda = [0.0708, 0.0595, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        self.levels = len(self.lmbda)  # 8
        if not v0:
            self.gain_unit = GainModule(n=self.levels, N=M)
            self.inv_gain_unit = GainModule(n=self.levels, N=M)

            self.hyper_gain_unit = GainModule(n=self.levels, N=N)
            self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)
        else:
            self.gain_unit = GainModule0(n=self.levels, N=M)
            self.inv_gain_unit = GainModule0(n=self.levels, N=M)

            self.hyper_gain_unit = GainModule0(n=self.levels, N=N)
            self.hyper_inv_gain_unit = GainModule0(n=self.levels, N=N)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# 17362723
class FGainedMeanScale(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()

        # [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
        self.lmbda = [0.2000, 0.1000, 0.0500, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        self.levels = len(self.lmbda)

        self.g_a = g_a(3, N, M, self.levels)
        self.g_s = g_s(M, N, 3, self.levels)
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, one_hot):
        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, one_hot):
        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, one_hot):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class GainedCheng2020(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        # self.g_a = nn.Sequential(
        #     ResidualBlockWithStride(3, N, stride=2),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N, stride=2),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N, stride=2),
        #     ResidualBlock(N, N),
        #     conv3x3(N, N, stride=2),
        # )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        # self.g_s = nn.Sequential(
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     subpel_conv3x3(N, 3, 2),
        # )

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        # self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...
        self.lmbda = [0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035]
        self.levels = len(self.lmbda)  # 8

        self.gain_unit = GainModule(n=self.levels, N=M)
        self.inv_gain_unit = GainModule(n=self.levels, N=M)

        self.hyper_gain_unit = GainModule(n=self.levels, N=N)
        self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class GainedCheng2020MultiE(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        entropy_bottleneck = []
        for _ in range(self.levels):
            entropy_bottleneck.append(EntropyBottleneck(channels=N))
        self.entropy_bottleneck = nn.Sequential(*entropy_bottleneck)
        self.gaussian_conditional = GaussianConditional(None)

        # self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...
        self.lmbda = [0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035]
        self.levels = len(self.lmbda)  # 8

        self.gain_unit = GainModule(n=self.levels, N=M)
        self.inv_gain_unit = GainModule(n=self.levels, N=M)

        self.hyper_gain_unit = GainModule(n=self.levels, N=N)
        self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck[n](scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# 2376682
class FGainedCheng2020(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        self.lmbda = [0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        self.levels = len(self.lmbda)
        self.lambda_onehot = torch.eye(self.levels)

        self.g_a = g_a1(3, N, M, self.levels)
        self.g_s = g_s1(M, N, 3, self.levels)
        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, index, alpha=None):
        assert index < self.levels - 1
        if alpha is None:
            alpha = random.choice([0.0, 0.5, 1.0])
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y)
        # print(z.shape)
        # exit()
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, index, alpha):
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, index, alpha):
        assert isinstance(strings, list) and len(strings) == 2
        one_hot = alpha * self.lambda_onehot[index_rand] + (1 - alpha) * self.lambda_onehot[index_rand + 1]
        one_hot = one_hot.to(x.device)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# 23872236
class FGainedChengMultiE(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        self.lmbda = [0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        self.levels = len(self.lmbda)
        self.lambda_onehot = torch.eye(self.levels)

        self.g_a = g_a1(3, N, M, self.levels)
        self.g_s = g_s1(M, N, 3, self.levels)
        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        entropy_bottleneck = []
        for _ in range(self.levels - 1):
            entropy_bottleneck.append(EntropyBottleneck(channels=N))
        self.entropy_bottleneck = nn.Sequential(*entropy_bottleneck)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, index, alpha=None):
        assert index < self.levels - 1
        if alpha is None:
            alpha = random.choice([0.0, 0.5, 1.0])
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_hat, z_likelihoods = self.entropy_bottleneck[index](z)
        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, index, alpha):
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_strings = self.entropy_bottleneck[index].compress(z)
        z_hat = self.entropy_bottleneck[index].decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, index, alpha):
        assert isinstance(strings, list) and len(strings) == 2

        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        z_hat = self.entropy_bottleneck[index].decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        for ii in range(self.levels):
            update_registered_buffers(
                self.gaussian_conditional[ii],
                "gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
            # Dynamically update the entropy bottleneck buffers related to the CDFs
            update_registered_buffers(
                self.entropy_bottleneck[ii],
                "entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class GainedMeanScaleMv(nn.Module):
    def __init__(self, N=64, M=128):
        super().__init__()

        self.g_a = Analysis_mv_net()
        self.g_s = Synthesis_mv_net()
        self.h_a = Analysis_mv_prior_net()
        self.h_s = Synthesis_mv_prior_net(in_ch=64, hidden=64 * 3 // 2, out_ch=128 * 2)

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.2000, 0.1000, 0.0500, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        self.levels = len(self.lmbda)  # 8

        self.gain_unit = GainModule(n=self.levels, N=M)
        self.inv_gain_unit = GainModule(n=self.levels, N=M)

        self.hyper_gain_unit = GainModule(n=self.levels, N=N)
        self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return x_hat, y_likelihoods, z_likelihoods

        # return {
        #     "x_hat": x_hat,
        #     "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        # }

    def compress(self, x, n, l):
        y = self.g_a(x)  # [4, 128, 16, 16]
        scaled_y = self.gain_unit(y, n, l)
        # print(scaled_y.shape)
        # exit()
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return y_strings, z_strings, z.size()[-2:]
        # return {
        #     "strings": [y_strings, z_strings],
        #     "shape": z.size()[-2:]
        # }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return x_hat

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class GainedMeanScaleRes(nn.Module):
    def __init__(self, N=64, M=96):
        super().__init__()

        self.g_a = Analysis_res_net()
        self.g_s = Synthesis_res_net()
        self.h_a = Analysis_res_prior_net()
        self.h_s = Synthesis_res_prior_net(in_ch=64, hidden=64 * 3 // 2, out_ch=96 * 2)

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.2000, 0.1000, 0.0500, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        self.levels = len(self.lmbda)  # 8

        self.gain_unit = GainModule(n=self.levels, N=M)
        self.inv_gain_unit = GainModule(n=self.levels, N=M)

        self.hyper_gain_unit = GainModule(n=self.levels, N=N)
        self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return x_hat, y_likelihoods, z_likelihoods

        # return {
        #     "x_hat": x_hat,
        #     "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        # }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return y_strings, z_strings, z.size()[-2:]

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return x_hat

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class CheckerboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y)
        return x * y[:, :, None, None]


class ConvBlockResidual(nn.Module):
    def __init__(self, ch_in, ch_out, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            SELayer(ch_out) if se_layer else nn.Identity(),
        )
        self.up_dim = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up_dim(x)
        return x2 + x1


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class CheckerboardFGainedCheng(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()
        self.N = N
        self.M = M

        self.lmbda = [0.40, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002]
        self.levels = len(self.lmbda)
        self.lambda_onehot = torch.eye(self.levels)

        self.g_a = g_a1(3, N, M, self.levels)
        self.g_s = g_s1(M, N, 16, self.levels)
        # self.h_a = h_a1(M, N, N, self.levels)
        # self.h_s = h_s1(N, M, M * 2, self.levels)

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )

        self.context_prediction = CheckerboardMaskedConv2d(M, M * 2, 5, 1, 2)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, index, alpha):
        assert index < self.levels - 1
        if alpha is None:
            alpha = random.choice([0.0, 0.5, 1.0])
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        batch_size, channel, x_height, x_width = x.shape
        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        params = self.h_s(z_hat, one_hot)

        anchor = torch.zeros_like(y_hat).to(x.device)
        non_anchor = torch.zeros_like(y_hat).to(x.device)

        anchor[:, :, 0::2, 1::2] = y_hat[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y_hat[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y_hat[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y_hat[:, :, 1::2, 1::2]

        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        # compress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_hat = torch.zeros([batch_size, self.M, x_height // 16, x_width // 16]).to(x.device)
        means_hat = torch.zeros([batch_size, self.M, x_height // 16, x_width // 16]).to(x.device)

        scales_hat[:, :, 0::2, 1::2] = scales_anchor[:, :, 0::2, 1::2]
        scales_hat[:, :, 1::2, 0::2] = scales_anchor[:, :, 1::2, 0::2]
        scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
        scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
        means_hat[:, :, 0::2, 1::2] = means_anchor[:, :, 0::2, 1::2]
        means_hat[:, :, 1::2, 0::2] = means_anchor[:, :, 1::2, 0::2]
        means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
        means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]

        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)
        x_hat = self.refine(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, index, alpha):
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        batch_size, channel, x_height, x_width = x.shape
        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat, one_hot)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]

        # compress anchor
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_strings = self.gaussian_conditional.compress(anchor, indexes_anchor, means_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor)

        # compress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(non_anchor, indexes_non_anchor, means=means_non_anchor)

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, index, alpha):
        # assert isinstance(strings, list) and len(strings) == 2
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat, one_hot)

        batch_size, channel, z_height, z_width = z_hat.shape

        # decompress anchor
        ctx_params_anchor = torch.zeros([batch_size, 2 * self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_anchor)

        # decompress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor)
        non_anchor_quantized = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor,
                                                                    means=means_non_anchor)

        y_hat = anchor_quantized + non_anchor_quantized
        x_hat = self.g_s(y_hat, one_hot)
        x_hat = self.refine(x_hat)

        return {"x_hat": x_hat}

    def compress_slice_concatenate(self, x, one_hot):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        batch_size, channel, x_height, x_width = x.shape

        y = self.g_a(x, one_hot)

        y_a = y[:, :, 0::2, 0::2]
        y_d = y[:, :, 1::2, 1::2]
        y_b = y[:, :, 0::2, 1::2]
        y_c = y[:, :, 1::2, 0::2]

        z = self.h_a(y, one_hot)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat, one_hot)

        anchor = torch.zeros_like(y).to(x.device)
        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)

        anchor_quantized = torch.zeros_like(y).to(x.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means=means_d)

        return {
            "strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress_slice_concatenate(self, strings, shape, one_hot):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        start_time = time.process_time()

        z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        params = self.h_s(z_hat, one_hot)

        batch_size, channel, z_height, z_width = z_hat.shape
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        anchor_quantized = torch.zeros([batch_size, self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        # Add non_anchor_quantized
        anchor_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        x_hat = self.g_s(anchor_quantized, one_hot)
        x_hat = self.refine(x_hat)

        end_time = time.process_time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class FGainedMeanScaleMv(nn.Module):
    def __init__(self, lmbda=None):
        super().__init__()

        self.lmbda = lmbda
        self.levels = len(self.lmbda)
        self.lambda_onehot = torch.eye(self.levels)

        self.g_a = Analysis_mv_net()
        self.g_s = Synthesis_mv_net()
        self.h_a = Analysis_mv_prior_net()
        self.h_s = Synthesis_mv_prior_net(in_ch=64, hidden=64 * 3 // 2, out_ch=128 * 2)

        self.entropy_bottleneck = EntropyBottleneck(channels=64)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, index, alpha=None):
        assert index < self.levels - 1
        if alpha is None:
            alpha = random.choice([0.0, 0.5, 1.0])
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)

        return x_hat, y_likelihoods, z_likelihoods
        # return {
        #     "x_hat": x_hat,
        #     "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        # }

    def compress(self, x, index, alpha):
        assert index < self.levels - 1

        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, index, alpha):
        assert isinstance(strings, list) and len(strings) == 2
        assert index < self.levels - 1
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class FGainedMeanScaleRes(nn.Module):
    def __init__(self, lmbda=None):
        super().__init__()

        self.lmbda = lmbda
        self.levels = len(self.lmbda)
        self.lambda_onehot = torch.eye(self.levels)

        self.g_a = Analysis_res_net()
        self.g_s = Synthesis_res_net()
        self.h_a = Analysis_res_prior_net()
        self.h_s = Synthesis_res_prior_net(in_ch=64, hidden=64 * 3 // 2, out_ch=96 * 2)

        self.entropy_bottleneck = EntropyBottleneck(channels=64)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, index, alpha=None):
        assert index < self.levels - 1
        if alpha is None:
            alpha = random.choice([0.0, 0.5, 1.0])
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)

        return x_hat, y_likelihoods, z_likelihoods

    def compress(self, x, index, alpha):
        assert index < self.levels - 1
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        y = self.g_a(x, one_hot)
        z = self.h_a(y, one_hot)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, index, alpha):
        assert isinstance(strings, list) and len(strings) == 2
        assert index < self.levels - 1
        one_hot = alpha * self.lambda_onehot[index] + (1 - alpha) * self.lambda_onehot[index + 1]
        one_hot = one_hot.to(x.device)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat, one_hot)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat, one_hot)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class LibinCheng(nn.Module):
    def __init__(self, N=224):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.0932, 0.0483, 0.0250, 0.0130, 0.0067]
        self.levels = len(self.lmbda)

        self.q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))
        self.q_scale = nn.Parameter(torch.ones((self.levels, 1, 1, 1)))

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2)
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 16, 2),
        )

        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )

        self._initialize_weights()

    def get_curr_q(self, q_scale):
        q_basic = LowerBound.apply(self.q_basic, 0.5)
        return q_basic * q_scale

    def forward(self, x, q_index=None, q_scale=None):
        if q_scale is None:
            q_scale = self.q_scale[q_index]
        curr_q = self.get_curr_q(q_scale)

        y = self.g_a(x)
        y = y / curr_q
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * curr_q
        x_hat = self.g_s(y_hat)
        x_hat = self.refine(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out


class CFT(nn.Module):
    def __init__(self, n=3, N=128):
        super(CFT, self).__init__()
        self.n = n
        self.Linear1 = nn.Linear(n, N, bias=False)
        torch.nn.init.ones_(self.Linear1.weight)
        # self.Linear2 = nn.Linear(N // 2, N)
        # self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.Softplus()

    def forward(self, x_input, lmbda):
        lmbda = lmbda.unsqueeze(0).repeat((x_input.size(0), self.n))
        mask = self.Linear1(lmbda)
        # mask = self.act1(mask)
        # mask = self.Linear2(mask)
        mask = self.act2(mask)
        mask = mask.unsqueeze(2).unsqueeze(3)
        # print(mask.shape)
        # exit()
        return x_input * mask


class SFTResblk(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)

        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)

    def forward(self, x, qmap):
        dx = self.conv_0(self.actvn(self.norm_0(x, qmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, qmap)))
        out = x + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# 6887392
class AnalysisTransform(nn.Module):
    def __init__(self, in_ch=3, hidden=192, out_ch=320, prior_nc=64, levels=8):
        super(AnalysisTransform, self).__init__()

        self.conv1 = conv(in_ch, hidden)
        self.conv2 = conv(hidden, hidden)
        self.conv3 = conv(hidden, hidden)
        self.conv4 = conv(hidden, out_ch)
        self.act1 = GDN(hidden)
        self.act2 = GDN(hidden)
        self.act3 = GDN(hidden)

        self.res1 = ResidualBlock(hidden, hidden)
        self.res2 = ResidualBlock(hidden, hidden)
        self.res3 = ResidualBlock(hidden, hidden)

        self.q_g1 = nn.Sequential(
            conv(in_ch + 1, hidden // 2),
            nn.LeakyReLU(0.1, True),
            conv(hidden // 2, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.q_g2 = nn.Sequential(
            conv(prior_nc, prior_nc),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.q_g3 = nn.Sequential(
            conv(prior_nc, prior_nc),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.sft1 = SFT(hidden, prior_nc, 3)
        self.sft2 = SFT(hidden, prior_nc, 3)
        self.sft3 = SFT(hidden, prior_nc, 3)

        self.cft = CFT(levels, hidden)

    def forward(self, x_input, lambda_map):
        q_map1 = self.q_g1(torch.cat([x_input, lambda_map], dim=1))
        x1 = self.act1(self.conv1(x_input))
        x1 = self.sft1(x1, q_map1)
        x1 = self.res1(x1)
        # torch.Size([2, 64, 128, 128]) torch.Size([2, 192, 128, 128])

        q_map2 = self.q_g2(q_map1)
        x2 = self.act2(self.conv2(x1))
        x2 = self.sft2(x2, q_map2)
        x2 = self.res2(x2)
        # torch.Size([2, 64, 64, 64]) torch.Size([2, 192, 64, 64])

        q_map3 = self.q_g3(q_map2)
        x3 = self.act3(self.conv3(x2))
        x3 = self.sft3(x3, q_map3)
        x3 = self.res3(x3)
        # torch.Size([2, 64, 32, 32]) torch.Size([2, 192, 32, 32])

        x3 = self.cft(x3, lambda_map[0, 0, 0, 0])
        return self.conv4(x3)


# 8262403
class SynthesisTransform(nn.Module):
    def __init__(self, in_ch=320, hidden=192, out_ch=16, prior_nc=64, levels=8):
        super(SynthesisTransform, self).__init__()

        self.conv1 = deconv(in_ch, hidden)
        self.conv2 = deconv(hidden, hidden)
        self.conv3 = deconv(hidden, hidden)
        self.conv4 = deconv(hidden, out_ch)
        self.act1 = GDN(hidden, inverse=True)
        self.act2 = GDN(hidden, inverse=True)
        self.act3 = GDN(hidden, inverse=True)

        self.res1 = ResidualBlock(hidden, hidden)
        self.res2 = ResidualBlock(hidden, hidden)
        self.res3 = ResidualBlock(hidden, hidden)

        self.q_g1 = nn.Sequential(
            deconv(in_ch + 1, hidden // 2),
            nn.LeakyReLU(0.1, True),
            conv(hidden // 2, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.q_g2 = nn.Sequential(
            deconv(prior_nc, prior_nc),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.q_g3 = nn.Sequential(
            deconv(prior_nc, prior_nc),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.sft1 = SFT(hidden, prior_nc, 3)
        self.sft2 = SFT(hidden, prior_nc, 3)
        self.sft3 = SFT(hidden, prior_nc, 3)

        self.cft = CFT(levels, hidden)

    def forward(self, x_input, lambda_map):
        q_map1 = self.q_g1(torch.cat([x_input, lambda_map], dim=1))
        x1 = self.act1(self.conv1(x_input))
        x1 = self.sft1(x1, q_map1)
        x1 = self.res1(x1)
        # torch.Size([2, 64, 128, 128]) torch.Size([2, 192, 128, 128])

        q_map2 = self.q_g2(q_map1)
        x2 = self.act2(self.conv2(x1))
        x2 = self.sft2(x2, q_map2)
        x2 = self.res2(x2)
        # torch.Size([2, 64, 64, 64]) torch.Size([2, 192, 64, 64])

        q_map3 = self.q_g3(q_map2)
        x3 = self.act3(self.conv3(x2))
        x3 = self.sft3(x3, q_map3)
        x3 = self.res3(x3)
        # torch.Size([2, 64, 32, 32]) torch.Size([2, 192, 32, 32])

        x3 = self.cft(x3, lambda_map[0, 0, 0, 0])
        return self.conv4(x3)


# 5377984
class HyperAnalysisTransform(nn.Module):
    def __init__(self, in_ch=320, hidden=192, out_ch=192, prior_nc=64):
        super(HyperAnalysisTransform, self).__init__()

        self.conv1 = conv(in_ch, hidden, stride=1, kernel_size=3)
        self.conv2 = conv(hidden, hidden)
        self.conv3 = conv(hidden, out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.q_g1 = nn.Sequential(
            conv(in_ch + 1, hidden // 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(hidden // 2, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.q_g2 = nn.Sequential(
            conv(prior_nc, prior_nc),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.sft1 = SFT(hidden, prior_nc, 3)
        self.sft2 = SFT(hidden, prior_nc, 3)

    def forward(self, x_input, lambda_map):
        q_map1 = self.q_g1(torch.cat([x_input, lambda_map], dim=1))
        x1 = self.act1(self.conv1(x_input))
        x1 = self.sft1(x1, q_map1)
        # torch.Size([2, 64, 128, 128]) torch.Size([2, 192, 128, 128])

        q_map2 = self.q_g2(q_map1)
        x2 = self.act2(self.conv2(x1))
        x2 = self.sft2(x2, q_map2)
        # torch.Size([2, 64, 64, 64]) torch.Size([2, 192, 64, 64])

        return self.conv3(x2)


class HyperSynthesisTransform(nn.Module):
    def __init__(self, in_ch=192, hidden=320, out_ch=320 * 2, prior_nc=64):
        super(HyperSynthesisTransform, self).__init__()

        self.conv1 = deconv(in_ch, hidden)
        self.conv2 = deconv(hidden, hidden * 3 // 2)
        self.conv3 = conv(hidden * 3 // 2, out_ch, 3, 1)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.q_g1 = nn.Sequential(
            deconv(in_ch + 1, hidden // 2),
            nn.LeakyReLU(0.1, True),
            conv(hidden // 2, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.q_g2 = nn.Sequential(
            deconv(prior_nc, prior_nc),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 3, 1)
        )

        self.sft1 = SFT(hidden, prior_nc, 3)
        self.sft2 = SFT(hidden * 3 // 2, prior_nc, 3)
        # print(hidden * 3 // 2)

    def forward(self, x_input, lambda_map):
        q_map1 = self.q_g1(torch.cat([x_input, lambda_map], dim=1))
        x1 = self.act1(self.conv1(x_input))
        # print(x1.shape, q_map1.shape)
        x1 = self.sft1(x1, q_map1)
        # torch.Size([2, 64, 128, 128]) torch.Size([2, 192, 128, 128])

        q_map2 = self.q_g2(q_map1)
        x2 = self.act2(self.conv2(x1))
        # print(x2.shape, q_map2.shape)
        x2 = self.sft2(x2, q_map2)
        # print(x2.shape)
        return self.conv3(x2)


class SpatialTemporalFlexibleRateIMC(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]
        self.levels = len(self.lmbda)  # 8

        self.g_a = AnalysisTransform()
        self.g_s = SynthesisTransform()
        self.h_a = HyperAnalysisTransform()
        self.h_s = HyperSynthesisTransform()

        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, index):
        b, c, h, w = x.size()
        lmbda = self.lmbda[index]

        lambda_map = torch.tensor(lmbda, device=x.device, dtype=x.dtype)
        x_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((b, 1, h, w))
        y_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((b, 1, h // 16, w // 16))
        z_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((b, 1, h // 64, w // 64))

        y = self.g_a(x, x_lambda_map)
        z = self.h_a(y, y_lambda_map)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat, z_lambda_map)
        # print(y.shape, gaussian_params.shape)
        # exit()
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, y_lambda_map)
        x_hat = self.refine(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, index):
        b, c, h, w = x.size()
        lmbda = self.lmbda[index]

        lambda_map = torch.tensor(lmbda, device=x.device, dtype=x.dtype)
        x_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((b, 1, h, w))
        y_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((b, 1, h // 16, w // 16))
        z_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((b, 1, h // 64, w // 64))

        y = self.g_a(torch.cat([x, x_lambda_map], 1))
        z = self.h_a(torch.cat([y, y_lambda_map], 1))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(torch.cat([z_hat, z_lambda_map], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, index):
        assert isinstance(strings, list) and len(strings) == 2
        lmbda = self.lmbda[index]

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        lambda_map = torch.tensor(lmbda, device=z_hat.device, dtype=z_hat.dtype)

        z_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((z_hat.size(0), 1, z_hat.size(2), z_hat.size(3)))

        gaussian_params = self.h_s(torch.cat([z_hat, z_lambda_map], 1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_lambda_map = lambda_map.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat((y_hat.size(0), 1, y_hat.size(2), y_hat.size(3)))
        x_hat = self.g_s(torch.cat([y_hat, y_lambda_map], 1))
        x_hat = self.refine(x_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# train form 0
class GainedCheng2020v1(nn.Module):
    def __init__(self, N=224, M=224):
        super().__init__()

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.0483, 0.0130, 0.0067, 0.0035]
        self.levels = len(self.lmbda)  # 8

        self.gain_unit = GainModule(n=self.levels, N=M)
        self.inv_gain_unit = GainModule(n=self.levels, N=M)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# train form 0
class GainedCheng2020v2(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.0500, 0.0250, 0.0125, 0.00625] # mxh add from HUAWEI CVPR2021 Gained...
        # self.lmbda = [0.40, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005]
        self.levels = len(self.lmbda)  # 8

        self.gain_unit = GainModule(n=self.levels, N=M)
        self.inv_gain_unit = GainModule(n=self.levels, N=M)

        self.hyper_gain_unit = GainModule(n=self.levels, N=N)
        self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled_y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


# 31178571
class SpatialTemporalEfficient(nn.Module):
    def __init__(self, N=192):
        super().__init__()
        self.N = int(N)
        self.lmbda = [0.070, 0.035, 0.0175, 0.00875]
        self.levels = len(self.lmbda)
        self.anchor_num = int(self.levels)

        self.q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))
        self.q_scale = nn.Parameter(torch.ones((self.levels, 1, 1, 1)))

        self.g_a, self.g_s = get_enc_dec_models(3, 16, N)
        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )
        self.h_a, self.h_s = get_hyper_enc_dec_models(N, N)

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(N * 2, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(N * 4, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 2, 3, padding=1)
        )

        self._initialize_weights()

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

    def get_curr_q(self, q_scale):
        q_basic = LowerBound.apply(self.q_basic, 0.5)
        return q_basic * q_scale

    def forward(self, x, q_index=None, q_scale=None):
        if q_scale is None:
            q_scale = self.q_scale[q_index]
        curr_q = self.get_curr_q(q_scale)

        y = self.g_a(x)  # [b, N, w//16, h//16]
        y = y / curr_q
        z = self.h_a(y)  # [b, N, w//64, h//64]

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        gaussian_params = self.h_s(z_hat)  # [b, N * 2, w//16, h//16]
        q_step, scales, means = self.y_prior_fusion(gaussian_params).chunk(3, 1)  # [b, N, w//16, h//16]

        y_hat, y_likelihoods = self.forward_spatial_temporal(y, means, scales, q_step)

        y_hat = y_hat * curr_q
        x_hat = self.g_s(y_hat)
        x_hat = self.refine(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def forward_spatial_temporal(self, y, means, scales, quant_step):
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, device)

        quant_step = LowerBound.apply(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1 = y.chunk(2, 1)

        scales_0, scales_1 = scales.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)

        y_hat_0_0, mean_hat_0_0, scales_hat_0_0, y_likelihood_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_hat_1_1, mean_hat_1_1, scales_hat_1_1, y_likelihood_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
        scales_0, means_0, scales_1, means_1 = self.y_spatial_prior(params).chunk(4, 1)

        y_hat_0_1, mean_hat_0_1, scales_hat_0_1, y_likelihood_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_hat_1_0, mean_hat_1_0, scales_hat_1_0, y_likelihood_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)

        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1
        mean_hat_0 = mean_hat_0_0 + mean_hat_0_1
        scales_hat_1 = scales_hat_1_0 + scales_hat_1_1
        mean_hat_1 = mean_hat_1_0 + mean_hat_1_1

        # scales_hat = torch.cat((scales_hat_0, scales_hat_1), dim=1)
        # mean_hat = torch.cat((mean_hat_0, mean_hat_1), dim=1)
        # _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=mean_hat)
        y_likelihoods0 = torch.cat([y_likelihood_0_0, y_likelihood_1_1], 1)
        y_likelihoods1 = torch.cat([y_likelihood_0_1, y_likelihood_1_0], 1)

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        y_hat_1 = y_hat_1_1 + y_hat_1_0

        y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        y_hat = y_hat * quant_step

        return y_hat, y_likelihoods

    @staticmethod
    def get_mask(height, width, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=torch.float32, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return mask_0, mask_1

    def process_with_mask(self, y, scales, means, mask):
        # scales_hat = scales * mask
        # means_hat = means * mask
        # y_res = (y - means_hat) * mask
        # y_q = self.quant(y_res)
        # y_hat = y_q + means_hat

        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        y_q = ste_round(y_res)
        y_hat = y_q + means_hat

        _, y_likelihood = self.gaussian_conditional(y, scales_hat, means_hat)

        return y_hat, means_hat, scales_hat, y_likelihood

    def process_with_mask1(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y * mask, indexes, means_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)

        return y_hat, y_strings, means_hat, scales_hat

    def compress_spatial_temporal(self, y, means, scales, quant_step):
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, device)

        quant_step = LowerBound.apply(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1 = y.chunk(2, 1)

        scales_0, scales_1 = scales.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)

        y_hat_0_0, y_strings_0_0, mean_hat_0_0, scales_hat_0_0 = \
            self.process_with_mask1(y_0, scales_0, means_0, mask_0)
        y_hat_1_1, y_strings_1_1, mean_hat_1_1, scales_hat_1_1 = \
            self.process_with_mask1(y_1, scales_1, means_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
        scales_0, means_0, scales_1, means_1 = self.y_spatial_prior(params).chunk(4, 1)

        y_hat_0_1, y_strings_0_1, mean_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask1(y_0, scales_0, means_0, mask_0)
        y_hat_1_0, y_strings_1_0, mean_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask1(y_1, scales_1, means_1, mask_1)

        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1
        means_hat_0 = mean_hat_0_0 + mean_hat_0_1

        scales_hat_1 = scales_hat_1_0 + scales_hat_1_1
        means_hat_1 = mean_hat_1_0 + mean_hat_1_1

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        y_hat_1 = y_hat_1_1 + y_hat_1_0

        # channel
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        # encode
        # channel 0
        index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
        y_q_0 = self.gaussian_conditional.quantize(y_hat_0, "symbols", means_hat_0)
        # y_hat_0 = y_q_0 + means_hat_0
        symbols_list.extend(y_q_0.reshape(-1).tolist())
        indexes_list.extend(index_0.reshape(-1).tolist())
        # channel 1
        index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
        y_q_1 = self.gaussian_conditional.quantize(y_hat_1, "symbols", means_hat_1)
        # y_hat_1 = y_q_1 + means_hat_1
        symbols_list.extend(y_q_1.reshape(-1).tolist())
        indexes_list.extend(index_1.reshape(-1).tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        # decode
        y_hat_slices = []
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        # channel 0
        rv_0 = decoder.decode_stream(index_0.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv_0 = torch.Tensor(rv_0).reshape(1, -1, y.shape[2], y.shape[3])
        y_hat_0 = self.gaussian_conditional.dequantize(rv_0, means_hat_0)
        y_hat_slices.append(y_hat_0)
        # channel 1
        rv_1 = decoder.decode_stream(index_1.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv_1 = torch.Tensor(rv_1).reshape(1, -1, y.shape[2], y.shape[3])
        y_hat_0 = self.gaussian_conditional.dequantize(rv_1, means_hat_1)
        y_hat_slices.append(y_hat_0)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = y_hat * quant_step

        return y_hat, y_strings

    def compress_and_decompress(self, x, q_scale):
        curr_q = self.get_curr_q(q_scale)

        y = self.g_a(x)
        y = y / curr_q
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_hat, y_strings = self.compress_spatial_temporal(y, means, scales, q_step)

        y_hat = y_hat * curr_q

        x_hat = self.refine(self.g_s(y_hat)).clamp_(0, 1)
        return {
            "strings": [y_strings, z_strings],
            "x_hat": x_hat
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)


# 31178571
class SpatialTemporalEfficient1(nn.Module):
    def __init__(self, N=192):
        super().__init__()
        self.N = int(N)
        self.lmbda = [0.070, 0.035, 0.0175, 0.00875]
        self.levels = len(self.lmbda)
        self.anchor_num = int(self.levels)

        self.q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))
        self.q_scale = nn.Parameter(torch.ones((self.levels, 1, 1, 1)))

        self.g_a, self.g_s = get_enc_dec_models(3, 16, N)
        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )
        self.h_a, self.h_s = get_hyper_enc_dec_models(N, N)

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(N * 2, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(N * 4, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 2, 3, padding=1)
        )

        self._initialize_weights()

        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)

    def get_curr_q(self, q_scale):
        q_basic = LowerBound.apply(self.q_basic, 0.5)
        return q_basic * q_scale

    def forward(self, x, q_index=None, q_scale=None):
        if q_scale is None:
            q_scale = self.q_scale[q_index]
        curr_q = self.get_curr_q(q_scale)

        y = self.g_a(x)  # [b, N, w//16, h//16]
        y = y / curr_q
        z = self.h_a(y)  # [b, N, w//64, h//64]

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        gaussian_params = self.h_s(z_hat)  # [b, N * 2, w//16, h//16]
        gaussian_params = self.y_prior_fusion(gaussian_params)  # [b, N * 3, w//16, h//16]
        q_step, scales, means = gaussian_params[:, 0::3, :, :], gaussian_params[:, 1::3, :, :], \
                                gaussian_params[:, 2::3, :, :]
        # [b, N, w//16, h//16]
        # print(q_step.shape, scales.shape, means.shape)

        y_hat, y_likelihoods = self.forward_spatial_temporal(y, means, scales, q_step)

        y_hat = y_hat * curr_q
        x_hat = self.g_s(y_hat)
        x_hat = self.refine(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def forward_spatial_temporal(self, y, means, scales, quant_step):
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, device)

        quant_step = LowerBound.apply(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1 = y[:, 0::2, :, :], y[:, 1::2, :, :]

        scales_0, scales_1 = scales[:, 0::2, :, :], scales[:, 1::2, :, :]
        means_0, means_1 = means[:, 0::2, :, :], means[:, 1::2, :, :]

        y_hat_0_0, mean_hat_0_0, scales_hat_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_hat_1_1, mean_hat_1_1, scales_hat_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
        params = self.y_spatial_prior(params)
        scales_0, means_0 = params[:, 0::4, :, :], params[:, 1::4, :, :]
        scales_1, means_1 = params[:, 2::4, :, :], params[:, 3::4, :, :]

        y_hat_0_1, mean_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_hat_1_0, mean_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)

        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1
        mean_hat_0 = mean_hat_0_0 + mean_hat_0_1
        _, y_likelihoods_0 = self.gaussian_conditional(y_0, scales_hat_0, means=mean_hat_0)

        scales_hat_1 = scales_hat_1_0 + scales_hat_1_1
        mean_hat_1 = mean_hat_1_0 + mean_hat_1_1
        _, y_likelihoods_1 = self.gaussian_conditional(y_1, scales_hat_1, means=mean_hat_1)

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        y_hat_1 = y_hat_1_1 + y_hat_1_0

        y_hat = torch.zeros_like(y)
        y_hat[:, 0::2, :, :] = y_hat_0
        y_hat[:, 1::2, :, :] = y_hat_1
        # print(y_hat.shape)
        # exit()
        # y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        y_hat = y_hat * quant_step

        y_likelihoods = torch.cat((y_likelihoods_0, y_likelihoods_1), dim=1)
        return y_hat, y_likelihoods

    @staticmethod
    def get_mask(height, width, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=torch.float32, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return mask_0, mask_1

    def process_with_mask(self, y, scales, means, mask):
        # scales_hat = scales * mask
        # means_hat = means * mask
        # y_res = (y - means_hat) * mask
        # y_q = self.quant(y_res)
        # y_hat = y_q + means_hat

        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        y_q = ste_round(y_res)
        y_hat = y_q + means_hat

        return y_hat, means_hat, scales_hat

    def process_with_mask1(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y * mask, indexes, means_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)

        return y_hat, y_strings, means_hat, scales_hat

    def compress_spatial_temporal(self, y, means, scales, quant_step):
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, device)

        quant_step = LowerBound.apply(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1 = y[:, 0::2, :, :], y[:, 1::2, :, :]

        scales_0, scales_1 = scales[:, 0::2, :, :], scales[:, 1::2, :, :]
        means_0, means_1 = means[:, 0::2, :, :], means[:, 1::2, :, :]

        y_hat_0_0, y_strings_0_0, mean_hat_0_0, scales_hat_0_0 = \
            self.process_with_mask1(y_0, scales_0, means_0, mask_0)
        y_hat_1_1, y_strings_1_1, mean_hat_1_1, scales_hat_1_1 = \
            self.process_with_mask1(y_1, scales_1, means_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
        scales_0, means_0, scales_1, means_1 = self.y_spatial_prior(params).chunk(4, 1)

        y_hat_0_1, y_strings_0_1, mean_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask1(y_0, scales_0, means_0, mask_0)
        y_hat_1_0, y_strings_1_0, mean_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask1(y_1, scales_1, means_1, mask_1)

        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1
        means_hat_0 = mean_hat_0_0 + mean_hat_0_1

        scales_hat_1 = scales_hat_1_0 + scales_hat_1_1
        means_hat_1 = mean_hat_1_0 + mean_hat_1_1

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        y_hat_1 = y_hat_1_1 + y_hat_1_0

        # channel
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        # encode
        # channel 0
        index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
        y_q_0 = self.gaussian_conditional.quantize(y_hat_0, "symbols", means_hat_0)
        # y_hat_0 = y_q_0 + means_hat_0
        symbols_list.extend(y_q_0.reshape(-1).tolist())
        indexes_list.extend(index_0.reshape(-1).tolist())
        # channel 1
        index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
        y_q_1 = self.gaussian_conditional.quantize(y_hat_1, "symbols", means_hat_1)
        # y_hat_1 = y_q_1 + means_hat_1
        symbols_list.extend(y_q_1.reshape(-1).tolist())
        indexes_list.extend(index_1.reshape(-1).tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        # decode
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        # channel 0
        rv_0 = decoder.decode_stream(index_0.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv_0 = torch.Tensor(rv_0).reshape(1, -1, y.shape[2], y.shape[3])
        y_hat_0 = self.gaussian_conditional.dequantize(rv_0, means_hat_0)
        # channel 1
        rv_1 = decoder.decode_stream(index_1.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv_1 = torch.Tensor(rv_1).reshape(1, -1, y.shape[2], y.shape[3])
        y_hat_1 = self.gaussian_conditional.dequantize(rv_1, means_hat_1)

        y_hat = torch.zeros_like(y)
        y_hat[:, 0::2, :, :] = y_hat_0
        y_hat[:, 1::2, :, :] = y_hat_1
        y_hat = y_hat * quant_step

        return y_hat, y_strings

    def compress_and_decompress(self, x, q_scale):
        curr_q = self.get_curr_q(q_scale)

        y = self.g_a(x)
        y = y / curr_q
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_hat, y_strings = self.compress_spatial_temporal(y, means, scales, q_step)

        y_hat = y_hat * curr_q

        x_hat = self.refine(self.g_s(y_hat)).clamp_(0, 1)
        return {
            "strings": [y_strings, z_strings],
            "x_hat": x_hat
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)


class IAT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=nhidden, out_channels=nhidden, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=x_nc, out_channels=x_nc, kernel_size=1),
        )
        self.mlp_beta = nn.Sequential(
            nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=x_nc, out_channels=x_nc, kernel_size=1),
        )

    def forward(self, x, qmap, reverse=False):
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        if not reverse:
            out = x * (1 + gamma) + beta
        else:
            out = (x - beta)/(1 + gamma)
        return out


class ResBottleneckBlock(nn.Module):
    def __init__(self, channel, slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if slope < 0.0001:
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return x + out


class g_a_vb(nn.Module):
    def __init__(self, N=64, M=32):
        super(g_a_vb, self).__init__()
        self.conv1 = conv(3, N)
        self.resb1 = nn.Sequential(
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
        )
        self.iat1 = IAT(N)

        self.conv2 = conv(N, N)
        self.resb2 = nn.Sequential(
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
        )
        self.iat2 = IAT(N)

        self.conv3 = conv(N, N)
        self.resb3 = nn.Sequential(
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
        )
        self.iat3 = IAT(N)

        self.conv4 = conv(N, M)

    def forward(self, x, qmap, inverse=False):
        x = self.conv1(x)
        x = self.resb1(x)
        x = self.iat1(x, qmap, inverse)
        x = self.conv2(x)
        x = self.resb2(x)
        x = self.iat2(x, qmap, inverse)
        x = self.conv3(x)
        x = self.resb3(x)
        x = self.iat3(x, qmap, inverse)
        x = self.conv4(x)
        return x


class g_s_vb(nn.Module):
    def __init__(self, N=64, M=32):
        super(g_s_vb, self).__init__()

        self.conv1 = deconv(M, N)
        self.resb1 = nn.Sequential(
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
        )
        self.iat1 = IAT(N)

        self.conv2 = deconv(N, N)
        self.resb2 = nn.Sequential(
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
        )
        self.iat2 = IAT(N)

        self.conv3 = deconv(N, N)
        self.resb3 = nn.Sequential(
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
        )
        self.iat3 = IAT(N)

        self.conv4 = deconv(N, 3)

    def forward(self, x, qmap, inverse=False):
        x = self.conv1(x)
        x = self.resb1(x)
        x = self.iat1(x, qmap, inverse)
        x = self.conv2(x)
        x = self.resb2(x)
        x = self.iat2(x, qmap, inverse)
        x = self.conv3(x)
        x = self.resb3(x)
        x = self.iat3(x, qmap, inverse)
        x = self.conv4(x)
        return x


class ICIP2020ResBVB(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = g_a_vb(N=N, M=M)

        self.g_s = g_s_vb(N=N, M=M)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, qmap):
        y = self.g_a(x, qmap, inverse=False)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat, qmap, inverse=True)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, qmap, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class ICIP2020ResB(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(3, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class ChannelSplitICIP2020ResB(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3):
        super().__init__()
        self.N = N
        self.num_slices = 8
        self.max_support_slices = 4

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(in_ch, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, out_ch, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
        )

        self.h_scale_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i + 1, self.max_support_slices + 1), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, N // 2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N // 2, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]
        # print(latent_scales.shape, latent_means.shape)
        # exit()

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        updated = self.entropy_bottleneck.update(force=force)
        if scale_table is None:
            scale_table = get_scale_table()
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class ICIP2020ResBVB1(nn.Module):
    def __init__(self, N=192, M=320, v0=False, x5=False, psnr=True):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = 10
        self.max_support_slices = 5
        if psnr:
            self.lmbda = [0.0808, 0.0512, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
            # self.lmbda = [0.2648, 0.1124, 0.0512, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
            # self.lmbda = [0.2548, 0.1224, 0.0512, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
            # self.lmbda = [0.1606, 0.0932, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
            # self.lmbda = [0.0808, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        else:
            self.lmbda = [187.49, 115.37, 60.50, 31.73, 16.64, 8.73, 4.58, 2.40]
        # self.lmbda = [0.065, 0.0325, 0.01625, 0.008125, 0.004062, 0.00203125]
        self.levels = len(self.lmbda)  # 8

        if not v0:
            self.gain_unit = GainModule(n=self.levels, N=M)
            self.inv_gain_unit = GainModule(n=self.levels, N=M)
            self.hyper_gain_unit = GainModule(n=self.levels, N=N)
            self.hyper_inv_gain_unit = GainModule(n=self.levels, N=N)
        else:
            self.gain_unit = GainModule0(n=self.levels, N=M)
            self.inv_gain_unit = GainModule0(n=self.levels, N=M)
            self.hyper_gain_unit = GainModule0(n=self.levels, N=N)
            self.hyper_inv_gain_unit = GainModule0(n=self.levels, N=N)

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        if not x5:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, 3),
            )
        else:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                ResBottleneckBlock(N),
                deconv(N, 3),
            )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_mean_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        self.h_scale_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256),
            nn.LeakyReLU(inplace=True),
            conv(256, M, stride=1, kernel_size=3),
        )

        # self.slice_transform = nn.Sequential(
        #     conv(M, 224),
        #     nn.LeakyReLU(inplace=True),
        #     conv(224, 128),
        #     nn.LeakyReLU(inplace=True),
        #     conv(128, slice_depth, stride=1, kernel_size=3)
        # )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        y = self.gain_unit(y, n, l)
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        z = self.hyper_gain_unit(z, n, l)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n=None, l=1):
        y = self.g_a(x)
        y = self.gain_unit(y, n, l)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, n=None, l=1):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


class ChannelSplitICIP2020ResBGain(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3, levels=5):
        super().__init__()
        self.N = N
        self.num_slices = 8
        self.max_support_slices = 4

        self.levels = levels
        self.gain_unit = GainModule0(n=self.levels, N=N)
        self.inv_gain_unit = GainModule0(n=self.levels, N=N)
        self.hyper_gain_unit = GainModule0(n=self.levels, N=N)
        self.hyper_inv_gain_unit = GainModule0(n=self.levels, N=N)

        slice_depth = self.N // self.num_slices
        if slice_depth * self.num_slices != self.N:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.N}/{self.num_slices})")

        self.g_a = nn.Sequential(
            conv(in_ch, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            conv(N, N, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, N, kernel_size=5, stride=2),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            ResBottleneckBlock(N),
            deconv(N, out_ch, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
        )

        self.h_scale_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            subpel_conv3x3(N, N, 2),
            nn.GELU(),
            conv3x3(N, N),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i, self.max_support_slices), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
                nn.GELU(),
                conv(32, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(N + slice_depth * min(i + 1, self.max_support_slices + 1), N, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N, N // 2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(N // 2, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, n=None, l=1):
        y = self.g_a(x)  # [b, 3, w, h]->[b, 320, w//16, h//16]
        y_shape = y.shape[2:]
        # print(1, y.shape)
        y = self.gain_unit(y, n, l)
        # print(2, y.shape)
        z = self.h_a(y)  # [b, 320, w//16, h//16]->[b, 192, w//64, h//64]
        z = self.hyper_gain_unit(z, n, l)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        latent_scales = self.h_scale_s(z_hat)  # [b, 320, w//16, h//16]
        latent_means = self.h_mean_s(z_hat)  # [b, 320, w//16, h//16]

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # print(slice_index)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            # print(mean_support.shape)
            # exit()
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print(mu.shape, y.shape)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # if slice_index == 0:
            #     print(y_slice.shape, scale.shape, mu.shape)
            #     exit()
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            # print(slice_index, lrp_support.shape)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n=None, l=1):
        y = self.g_a(x)
        y = self.gain_unit(y, n, l)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z = self.hyper_gain_unit(z, n, l)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, n=None, l=1):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        updated = self.entropy_bottleneck.update(force=force)
        if scale_table is None:
            scale_table = get_scale_table()
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


if __name__ == "__main__":
    fun = 2
    h, w = 256, 256
    model = GainedMeanScale()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x1 = torch.rand((2, 3, h, w))
    q = torch.rand((2, 1, h, w))
    out = model(x1, 0)
    print(out['x_hat'].shape)
    exit()

    model = SpatialTemporalEfficient().cuda()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x = torch.rand((1, 3, h, w))
    out = model(x.cuda(), 3)
    print(out['x_hat'].shape)
    # exit()
    model.update(force=True)
    out = model.compress_and_decompress(x.cuda(), torch.tensor(2.0).cuda())
    print(out['x_hat'].shape)
    exit()

    lambda_list = torch.tensor([50, 160, 300, 480, 710, 1000, 1350, 1780, 2302, 2915])
    lambda_onehot = torch.eye(len(lambda_list))

    index_rand = np.random.randint(0, len(lambda_list) - 1)
    weight_alpha_list = [0.0, 0.5, 1.0]
    alpha_rand = random.choice(weight_alpha_list)
    l_onehot = alpha_rand * lambda_onehot[index_rand] + (1 - alpha_rand) * lambda_onehot[index_rand + 1]

    lambda_train = torch.sum(torch.multiply(l_onehot, lambda_list))
    lambda_test = alpha_rand * lambda_list[index_rand] + (1 - alpha_rand) * lambda_list[index_rand + 1]
    # print(l_onehot)
    # print(lambda_train, lambda_test)
    # exit()
    if fun == 0:
        x = torch.rand((1, 128, 32, 32)).cuda()
        model = FeatureModulation(10, 128).cuda()
        print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        out = model(x, l_onehot.cuda())
        print(out.shape)
    elif fun == 1:
        x = torch.rand((1, 192, 16, 16))
        model = h_s(192, 320, 128, 10)
        print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        out = model(x, l_onehot)
        print(out.shape)

    model = SpatialTemporalEfficient().cuda()
    print(f'Total Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    x = torch.rand((1, 3, h, w))
    out = model(x.cuda(), 3)
    print(out['x_hat'].shape)
    exit()
    model.update(force=True)
    enc_out = model.compress(x.cuda(), 2)
    dec_out = model.decompress(enc_out['strings'], enc_out['shape'], 2)
    print(dec_out['x_hat'].shape)
    exit()

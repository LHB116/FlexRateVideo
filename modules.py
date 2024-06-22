import math

import torch
from torch import nn
import torch.nn.functional as F

from compressai.layers import subpel_conv3x3

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

backward_grid = [{} for _ in range(9)]  # 0~7 for GPU, -1 for CPU


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=feature.device, dtype=feature.dtype).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=feature.device, dtype=feature.dtype).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)

    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                   self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                      torch_warp(im2_list[img_index], flow_up),
                                                      flow_up], 1))

        return flow


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


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 3 * 3 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, flow):
        extra_feat = torch.cat([x, flow], dim=1)
        out = self.conv_offset(extra_feat)
        offset, mask = torch.split(out, [2 * 3 * 3 * self.deform_groups, 3 * 3 * self.deform_groups], dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(offset)
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        # mask
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class ContextualEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96, out=32):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, out, 2)

    def forward(self, x, context2, context3):
        # print(x.shape)
        # exit()
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class MyMCNetWOsm(nn.Module):
    def __init__(self, in_ch=3, hidden=64, out_ch=3):
        super(MyMCNetWOsm, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            ResBlock(hidden),
        )
        self.in_conv1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            ResBlock(hidden),
        )

        self.deform_align = nn.ModuleList(
            SecondOrderDeformableAlignment(
                hidden,
                hidden,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=5
            ) for _ in range(3)
        )

        self.fea_ext = FeatureExtractor(hidden)
        self.down = ContextualEncoder(hidden, hidden)
        self.up = ContextualDecoder(hidden, hidden)

        self.out_conv = nn.Conv2d(32, out_ch, 3, stride=1, padding=1)

        self.weight = nn.Sequential(
            nn.Conv2d(32, hidden, 3, stride=1, padding=1),
            ResBlock(hidden),
            nn.Conv2d(hidden, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.lrelu = nn.LeakyReLU(True)

    def forward(self, ref_frame, warped, mv, feature=None):
        if feature is None:
            fea = self.lrelu(self.in_conv(ref_frame))
        else:
            fea = self.lrelu(self.in_conv1(feature))
        fea1, fea2, fea3 = self.fea_ext(fea)
        mv2 = bilineardownsacling(mv) / 2.0
        mv3 = bilineardownsacling(mv2) / 2.0
        deform_fea1 = self.deform_align[0](fea1, mv)
        deform_fea2 = self.deform_align[1](fea2, mv2)
        deform_fea3 = self.deform_align[2](fea3, mv3)

        down_out = self.down(warped, deform_fea1, deform_fea2, deform_fea3)
        up_out = self.up(down_out, deform_fea2, deform_fea3)
        w = self.weight(up_out)
        out = w * warped + (1 - w) * self.out_conv(up_out)

        return up_out, out


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


class RefineNet(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=64, out_ch=2):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, 3, stride=1, padding=1),
            ResBottleneckBlock(hidden_channel),
            ResBottleneckBlock(hidden_channel),
            ResBottleneckBlock(hidden_channel),
            nn.Conv2d(hidden_channel, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x, ref_frame):
        return x + self.refine(torch.cat([x, ref_frame], 1))


class FeatureExtraction(nn.Module):
    def __init__(self, in_ch=6, nf=64, k=3, s=1):
        super(FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, k, s, k // 2, bias=True)
        self.rsb1 = nn.Sequential(
            ResBlock(nf, 0),
            ResBlock(nf, 0),
            ResBlock(nf, 0),
        )

    def forward(self, x):
        x = self.conv1(x)
        res1 = x + self.rsb1(x)
        return res1


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


class UNetWOsm(nn.Module):
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
        # print(x.shape, x3.shape)
        # exit()

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class MyRecNetWOsm(nn.Module):
    def __init__(self, in_ch=64, channel=64, out_ch=3, return_fea=True):
        super().__init__()
        self.return_fea = return_fea
        self.first_conv = nn.Conv2d(in_ch, channel, 3, stride=1, padding=1)
        self.unet_1 = UNetWOsm(channel, channel)
        self.unet_2 = UNetWOsm(channel, channel)
        self.recon_conv1 = nn.Conv2d(channel, out_ch, 3, stride=1, padding=1)
        self.recon_conv2 = nn.Conv2d(channel, out_ch, 3, stride=1, padding=1)
        self.recon_conv3 = nn.Conv2d(channel * 2, out_ch, 3, stride=1, padding=1)

        self.weight1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            ResBlock(channel),
            nn.Conv2d(channel, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.weight2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            ResBlock(channel),
            nn.Conv2d(channel, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feature = self.first_conv(x)
        feature1 = self.unet_1(feature)
        feature2 = self.unet_2(feature)
        recon1 = self.recon_conv1(feature1)
        recon2 = self.recon_conv2(feature2)
        recon3 = self.recon_conv3(torch.cat([feature1, feature2], 1))

        w1 = self.weight1(feature1)
        w2 = self.weight2(feature2)
        recon = w1 * recon1 + w2 * recon2 + (1 - w1 - w2) * recon3
        # print(feature.shape, recon.shape)
        if self.return_fea:
            return feature, recon
        else:
            return recon


class UNet1(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            ResBlock(128),
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
        )

        self.fea_convert = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            ResBlock(128),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x, sm_fea):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        sm_fea = self.fea_convert(sm_fea)
        x3 = self.context_refine(torch.cat([x3, sm_fea], 1))

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class MyRecNet(nn.Module):
    def __init__(self, in_ch=64, channel=64, out_ch=3, return_fea=True):
        super().__init__()
        self.return_fea = return_fea
        self.conv1 = nn.Conv2d(in_ch, channel, 3, stride=1, padding=1)

        self.unet_1 = UNet1(channel, channel)
        self.unet_2 = UNet1(channel, channel)
        self.recon_conv1 = nn.Conv2d(channel, out_ch, 3, stride=1, padding=1)
        self.recon_conv2 = nn.Conv2d(channel, out_ch, 3, stride=1, padding=1)
        self.recon_conv3 = nn.Conv2d(channel * 2, out_ch, 3, stride=1, padding=1)

        self.weight1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            ResBlock(channel),
            nn.Conv2d(channel, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.weight2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            ResBlock(channel),
            nn.Conv2d(channel, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rec_fea, mc_fea, sm_fea):
        feature = self.conv1(torch.cat([rec_fea, mc_fea], 1))
        feature1 = self.unet_1(feature, sm_fea)
        feature2 = self.unet_2(feature, sm_fea)
        recon1 = self.recon_conv1(feature1)
        recon2 = self.recon_conv2(feature2)
        recon3 = self.recon_conv3(torch.cat([feature1, feature2], 1))

        w1 = self.weight1(feature1)
        w2 = self.weight2(feature2)
        recon = w1 * recon1 + w2 * recon2 + (1 - w1 - w2) * recon3
        # print(feature.shape, recon.shape)
        if self.return_fea:
            return feature, recon
        else:
            return recon


class MyMCNet(nn.Module):
    def __init__(self, in_ch=3, hidden=64, up_out=32, out_ch=3, fea_ch=64):
        super(MyMCNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            ResBlock(hidden),
        )
        self.in_conv1 = nn.Sequential(
            nn.Conv2d(fea_ch, hidden, 3, padding=1),
            ResBlock(hidden),
        )

        self.fea_convert = nn.Sequential(
            nn.Conv2d(256, hidden, 3, padding=1),
            ResBlock(hidden),
        )

        self.fea_embd = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1),
            ResBlock(hidden),
            ResBlock(hidden, start_from_relu=False),
        )

        self.deform_align = nn.ModuleList(
            SecondOrderDeformableAlignment(
                hidden,
                hidden,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=5
            ) for _ in range(3)
        )

        self.fea_ext = FeatureExtractor(hidden)
        self.down = ContextualEncoder(hidden, hidden)
        self.up = ContextualDecoder(hidden, hidden, out=up_out)

        self.out_conv = nn.Conv2d(up_out, out_ch, 3, stride=1, padding=1)

        self.weight = nn.Sequential(
            nn.Conv2d(up_out, hidden, 3, stride=1, padding=1),
            ResBlock(hidden),
            nn.Conv2d(hidden, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.lrelu = nn.LeakyReLU(True)

    def forward(self, ref_frame, warped, mv, curr_fea, feature=None):
        # fea = self.lrelu(self.in_conv(ref_frame))
        if feature is None:
            fea = self.lrelu(self.in_conv(ref_frame))
        else:
            fea = self.lrelu(self.in_conv1(feature))
        fea1, fea2, fea3 = self.fea_ext(fea)
        mv2 = bilineardownsacling(mv) / 2.0
        mv3 = bilineardownsacling(mv2) / 2.0
        # [4, 64, 256, 256], [4, 64, 128, 128], [4, 64, 64, 64]
        # print(fea1.shape, fea2.shape, fea3.shape, curr_fea.shape)
        # print(curr_fea1.shape, curr_fea2.shape, curr_fea3.shape)
        # exit()
        deform_fea1 = self.deform_align[0](fea1, mv)
        deform_fea2 = self.deform_align[1](fea2, mv2)
        deform_fea3 = self.deform_align[2](fea3, mv3)
        # print(torch.cat([deform_fea1, curr_fea], 1).shape)
        # exit()
        curr_fea = self.fea_convert(curr_fea)
        deform_fea3 = self.fea_embd(torch.cat([deform_fea3, curr_fea], 1))

        down_out = self.down(warped, deform_fea1, deform_fea2, deform_fea3)
        # print(down_out.shape)
        # exit()
        up_out = self.up(down_out, deform_fea2, deform_fea3)
        w = self.weight(up_out)
        out = w * warped + (1 - w) * self.out_conv(up_out)

        # print(up_out.shape, out.shape, w.shape)
        return up_out, out


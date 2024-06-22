import os
import torch
import datetime
import utils
from utils import AverageMeter, load_pretrained
from tqdm import tqdm
from image_model import GainedMeanScale, FGainedMeanScale, ICIP2020ResBVB, ICIP2020ResB, ICIP2020ResBVB1
from video_model import LHB_DVC_WOSM_VB
import json
import logging
from dataset import get_dataset, get_loader, get_vb_dataloader
from dataset1 import get_dataset11, get_dataset1
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
import numpy as np
import torch.nn as nn
import math
import random
import shutil
from glob import glob
from compressai.zoo import mbt2018
# mbt2018(8, 'mse', True)
# exit()
from tensorboardX import SummaryWriter


def random_index(rate):
    start = 0
    index = 0
    rand = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if rand <= start:
            break
    return index


class RateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class RateDistortionLoss0(nn.Module):
    def __init__(self, lmbdas, msssim=False):
        super().__init__()
        if msssim:
            self.lmbdas = torch.tensor([i for i in lmbdas])
        else:
            self.lmbdas = torch.tensor([i * (255 ** 2) for i in lmbdas])
        self.msssim = msssim

    def forward(self, output, target):
        # [0.0708, 0.0595, 0.0483, 0.0250, 0.0130, 0.0067, 0.0035, 0.0018]
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if not self.msssim:
            out["mse_loss"] = torch.mean((output["x_hat"] - target) ** 2, dim=(1, 2, 3))
            dist = torch.mean(out["mse_loss"] * self.lmbdas.to(target.device))
        else:
            out["msssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0, size_average=False)
            dist = torch.mean(out["msssim_loss"] * self.lmbdas.to(target.device))
        out["loss"] = dist + out["bpp_loss"]
        return out


def quality2lambda(qlevel):
    return 1e-3 * torch.exp(4.382 * qlevel)


class PixelwiseRateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target, lmbdalevel):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_pixels)
            for likelihoods in output['likelihoods'].values()
        )

        mse = self.mse(output['x_hat'], target)
        lmbdalevel = lmbdalevel.expand_as(mse)
        out['mse_loss'] = torch.mean(lmbdalevel * mse)
        out['loss'] = 255 ** 2 * out['mse_loss'] * 1.2 + out['bpp_loss']

        return out


class ImageCodecTrainerVB(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.grad_clip = 1.0
        self.test_train_code = False
        self.test_iters = 4
        self.milestones = [80, 130, 180]

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"ICIP2020ResBVB_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        self.training_loader, self.valid_loader = get_vb_dataloader(args.train_dataset_root,
                                                                    args.val_dataset_root,
                                                                    args.batch_size
                                                                    )
        self.logger.info(f'[*] Train File Account For {len(self.training_loader)}, '
                         f'val {len(self.valid_loader)}')

        self.graph = ICIP2020ResBVB().cuda()
        # self.logger.info("[*] Load Pretrained")
        # ckpt = './checkpoint/mbt2018-mean-8-8089ae3e.pth.tar'
        # tgt_model_dict = self.graph.state_dict()
        # src_pretrained_dict = torch.load(ckpt)
        # _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        # tgt_model_dict.update(_pretrained_dict)
        # self.graph.load_state_dict(tgt_model_dict)

        parameters, aux_parameters = [], []
        for n, p in self.graph.named_parameters():
            if not n.endswith(".quantiles"):
                parameters.append(p)
            else:
                aux_parameters.append(p)
        self.image_codec_optim = torch.optim.Adam(parameters, lr=args.lr)
        self.image_codec_aux_optim = torch.optim.Adam(aux_parameters, lr=args.lr)

        self.image_codec_criterion = PixelwiseRateDistortionLoss()

        # model
        self.mode_type = args.mode_type

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            self.adjust_lr()
            lr = self.image_codec_optim.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            train_loss = AverageMeter()
            train_bpp, train_aux = AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()

            self.graph.train()
            train_bar = tqdm(self.training_loader)
            for kk, (image, qmap) in enumerate(train_bar):
                self.image_codec_optim.zero_grad()
                self.image_codec_aux_optim.zero_grad()

                image = image.cuda()
                qmap = qmap.cuda()
                lmbdalevel = quality2lambda(qmap)

                image_codec_output = self.graph(image, qmap)
                out_criterion = self.image_codec_criterion(image_codec_output, image, lmbdalevel)
                rd_loss = out_criterion["loss"]
                bpp_loss = out_criterion["bpp_loss"]
                mse_loss = out_criterion["mse_loss"]
                rd_loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)

                self.image_codec_optim.step()
                aux_loss = self.graph.aux_loss()
                aux_loss.backward()
                self.image_codec_aux_optim.step()

                mse_loss1 = torch.mean((image - image_codec_output['x_hat']).pow(2))
                psrn = 10 * np.log10(1.0 ** 2 / mse_loss1.detach().cpu())

                train_psnr.update(psrn.mean().detach().item(), self.batch_size)
                train_bpp.update(bpp_loss.mean().detach().item(), self.batch_size)
                train_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('train_psnr', psrn.detach().item(), self.global_step)
                    self.writer.add_scalar('train_bpp', bpp_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_loss', rd_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)

                train_bar.desc = "TRAIN [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.2f}]". \
                    format(epoch + 1,
                           self.num_epochs,
                           rd_loss.mean().detach().item(),
                           psrn.mean().detach().item(),
                           bpp_loss.mean().detach().item(),
                           aux_loss.mean().detach().item()
                           )

                self.global_step += 1

                if self.test_train_code:
                    if kk > self.test_iters:
                        break

            self.logger.info("TRAIN [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.2f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_psnr.avg,
                                    train_bpp.avg,
                                    train_aux.avg
                                    )
                             )

            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            self.validate()
            # self.image_codec_lr_scheduler.step(val_loss)
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_loss = AverageMeter()
        L = 10
        levels = [-100] + [int(100 * (i / L)) for i in range(L + 1)]
        psnr_dict = {k: AverageMeter() for k in levels}
        bpp_dict = {k: AverageMeter() for k in levels}

        with torch.no_grad():
            for l, valid_loader in enumerate(self.valid_loader):
                valid_bar = tqdm(valid_loader)
                for k, (image, qmap) in enumerate(valid_bar):
                    image = image.cuda()
                    qmap = qmap.cuda()
                    lmbdalevel = quality2lambda(qmap)
                    # print(image.shape, qmap.shape)

                    image_codec_output = self.graph(image, qmap)
                    out_criterion = self.image_codec_criterion(image_codec_output, image, lmbdalevel)
                    rd_loss = out_criterion["loss"]
                    bpp_loss = out_criterion["bpp_loss"]
                    mse_loss = out_criterion["mse_loss"]

                    aux_loss = self.graph.aux_loss()
                    mse_loss1 = torch.mean((image - image_codec_output['x_hat']).pow(2))
                    psrn = 10 * np.log10(1.0 ** 2 / mse_loss1.detach().cpu())

                    val_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                    psnr_dict[levels[l]].update(psrn, self.batch_size)
                    bpp_dict[levels[l]].update(bpp_loss.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar(f'val_psnr_{levels[l]}', psrn.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_bpp_{levels[l]}', bpp_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_loss_{levels[l]}', rd_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_aux_{levels[l]}', aux_loss.detach().item(), self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.2f}]". \
                        format(self.global_epoch + 1,
                               self.num_epochs,
                               rd_loss.mean().detach().item(),
                               psrn.mean().detach().item(),
                               bpp_loss.mean().detach().item(),
                               aux_loss.mean().detach().item()
                               )

                    self.global_eval_step += 1

                    if self.test_train_code:
                        if k > self.test_iters:
                            break

            self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
            for ll in levels:
                self.logger.info(f"Val [{ll}], PSNR [{psnr_dict[ll].avg:.4f}], Bpp [{bpp_dict[ll].avg:.4f}]")

        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()
        return val_loss.avg

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location=self.device)
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.image_codec_aux_optim.load_state_dict(checkpoint["aux_optimizer"])
        self.image_codec_optim.load_state_dict(checkpoint["optimizer"])
        # self.image_codec_optim.param_groups[0]['lr'] = 1e-4
        # self.image_codec_aux_optim.param_groups[0]['lr'] = 1e-4

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        if self.global_epoch < self.milestones[0]:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr
        elif self.milestones[0] <= self.global_epoch < self.milestones[1]:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr / 10.0
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr / 10.0
        else:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr / 100.0
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr / 100.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.image_codec_optim.state_dict(),
            "aux_optimizer": self.image_codec_aux_optim.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))


class GainedImageCodecTrainer(object):
    def __init__(self, args):
        # args
        self.milestones = [80, 130, 180]
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.grad_clip = 1.0

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"GainedMeanScale_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        self.training_loader, self.valid_loader = get_loader(args.train_dataset_root,
                                                             args.val_dataset_root,
                                                             256,
                                                             args.batch_size,
                                                             args.num_workers
                                                             )

        self.logger.info(f'[*] Train File Account For {len(self.training_loader)}, '
                         f'val {len(self.valid_loader)}')

        self.graph = GainedMeanScale().cuda()
        self.logger.info("[*] Load Pretrained")
        ckpt = './checkpoint/mbt2018-mean-8-8089ae3e.pth.tar'
        tgt_model_dict = self.graph.state_dict()
        src_pretrained_dict = torch.load(ckpt)
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.graph.load_state_dict(tgt_model_dict)

        parameters, aux_parameters = [], []
        for n, p in self.graph.named_parameters():
            if not n.endswith(".quantiles"):
                parameters.append(p)
            else:
                aux_parameters.append(p)
        self.image_codec_optim = torch.optim.Adam(parameters, lr=args.lr)
        self.image_codec_aux_optim = torch.optim.Adam(aux_parameters, lr=args.lr)

        self.image_codec_criterion = RateDistortionLoss()
        self.image_codec_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.image_codec_optim, "min")

        # model
        self.mode_type = args.mode_type

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_loss = AverageMeter()
            train_bpp, train_aux = AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()

            # adjust learning_rate
            self.adjust_lr()
            lr = self.image_codec_optim.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')

            self.graph.train()
            train_bar = tqdm(self.training_loader)
            for kk, image in enumerate(train_bar):
                self.image_codec_optim.zero_grad()
                self.image_codec_aux_optim.zero_grad()

                image = image.cuda()

                s = np.random.randint(0, self.graph.levels)  # choose random level from [0, levels-1]
                image_codec_output = self.graph(image, s)
                out_criterion = self.image_codec_criterion(image_codec_output, image, self.graph.lmbda[s])
                rd_loss = out_criterion["loss"]
                bpp_loss = out_criterion["bpp_loss"]
                mse_loss = out_criterion["mse_loss"]
                rd_loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)

                self.image_codec_optim.step()
                aux_loss = self.graph.aux_loss()
                aux_loss.backward()
                self.image_codec_aux_optim.step()

                psrn = 10 * np.log10(1.0 ** 2 / mse_loss.detach().cpu())

                train_psnr.update(psrn.mean().detach().item(), self.batch_size)
                train_bpp.update(bpp_loss.mean().detach().item(), self.batch_size)
                train_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('train_psnr', psrn.detach().item(), self.global_step)
                    self.writer.add_scalar('train_bpp', bpp_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_loss', rd_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)

                train_bar.desc = "TRAIN [{}|{}] [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                    format(s, self.graph.lmbda[s], epoch + 1,
                           self.num_epochs,
                           rd_loss.mean().detach().item(),
                           psrn.mean().detach().item(),
                           bpp_loss.mean().detach().item(),
                           aux_loss.mean().detach().item()
                           )

                self.global_step += 1

                # if kk > 100:
                #     break

            self.logger.info("TRAIN [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_psnr.avg,
                                    train_bpp.avg,
                                    train_aux.avg
                                    )
                             )

            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            val_loss = self.validate()
            self.image_codec_lr_scheduler.step(val_loss)
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_loss = AverageMeter()
        psnr_dict = {k: AverageMeter() for k in range(self.graph.levels)}
        bpp_dict = {k: AverageMeter() for k in range(self.graph.levels)}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_loader)
            for k, image in enumerate(valid_bar):
                image = image.cuda()

                for s in range(0, self.graph.levels):
                    image_codec_output = self.graph(image, s)
                    out_criterion = self.image_codec_criterion(image_codec_output, image, self.graph.lmbda[s])
                    rd_loss = out_criterion["loss"]
                    bpp_loss = out_criterion["bpp_loss"]
                    mse_loss = out_criterion["mse_loss"]

                    aux_loss = self.graph.aux_loss()
                    psrn = 10 * np.log10(1.0 ** 2 / mse_loss.detach().cpu())

                    val_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                    psnr_dict[s].update(psrn, self.batch_size)
                    bpp_dict[s].update(bpp_loss.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar(f'val_psnr_{s}', psrn.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_bpp_{s}', bpp_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_loss_{s}', rd_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_aux_{s}', aux_loss.detach().item(), self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                        format(self.global_epoch + 1,
                               self.num_epochs,
                               rd_loss.mean().detach().item(),
                               psrn.mean().detach().item(),
                               bpp_loss.mean().detach().item(),
                               aux_loss.mean().detach().item()
                               )

                self.global_eval_step += 1
                #
                # if k > 20:
                #     break

            self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
            for s in range(self.graph.levels):
                self.logger.info(f"Val [{s}], PSNR [{psnr_dict[s].avg:.4f}], Bpp [{bpp_dict[s].avg:.4f}]")

        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()
        return val_loss.avg

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location=self.device)
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.image_codec_aux_optim.load_state_dict(checkpoint["aux_optimizer"])
        self.image_codec_optim.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        if self.global_epoch < self.milestones[0]:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr
        elif self.milestones[0] <= self.global_epoch < self.milestones[1]:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr / 10.0
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr / 10.0
        else:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr / 100.0
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr / 100.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.image_codec_optim.state_dict(),
            "aux_optimizer": self.image_codec_aux_optim.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))


class GainedImageCodecTrainer0(object):
    def __init__(self, args):
        # args
        self.milestones = [15, 22]
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.grad_clip = 1.0

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"ICIP2020ResBVB1_v0True_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        self.graph = ICIP2020ResBVB1(v0=True, x5=False).cuda()
        self.logger.info("[*] Load Pretrained")
        # ckpt = './checkpoint/mbt2018-mean-8-8089ae3e.pth.tar'
        ckpt = '/home/tdx/桌面/Project/LHB/pretrained/ICIP2020ResB/mse/lambda_0.0932.pth'
        tgt_model_dict = self.graph.state_dict()
        src_pretrained_dict = torch.load(ckpt)['state_dict']
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.graph.load_state_dict(tgt_model_dict)

        parameters, aux_parameters = [], []
        for n, p in self.graph.named_parameters():
            if not n.endswith(".quantiles"):
                parameters.append(p)
            else:
                aux_parameters.append(p)
        self.image_codec_optim = torch.optim.Adam(parameters, lr=args.lr)
        self.image_codec_aux_optim = torch.optim.Adam(aux_parameters, lr=args.lr)

        self.image_codec_criterion = RateDistortionLoss0(self.graph.lmbda)
        self.logger.info(f"[*] lmbdas {self.graph.lmbda}")
        self.image_codec_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.image_codec_optim, "min")

        self.training_loader, self.valid_loader = get_loader(args.train_dataset_root,
                                                             args.val_dataset_root,
                                                             256,
                                                             self.graph.levels,
                                                             args.num_workers
                                                             )

        self.logger.info(f'[*] Train File Account For {len(self.training_loader)}, '
                         f'val {len(self.valid_loader)}')

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_loss = AverageMeter()
            train_bpp, train_aux = AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()

            # adjust learning_rate
            self.adjust_lr()
            lr = self.image_codec_optim.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')

            self.graph.train()
            train_bar = tqdm(self.training_loader)
            for kk, image in enumerate(train_bar):
                self.image_codec_optim.zero_grad()
                self.image_codec_aux_optim.zero_grad()
                image = image.cuda()
                image_codec_output = self.graph(image, torch.arange(0, self.graph.levels).cuda())
                out_criterion = self.image_codec_criterion(image_codec_output, image)
                rd_loss = out_criterion["loss"]
                bpp_loss = out_criterion["bpp_loss"]
                mse_loss = torch.mean(out_criterion["mse_loss"])
                # print(mse_loss)
                # exit()
                rd_loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)

                self.image_codec_optim.step()
                aux_loss = self.graph.aux_loss()
                aux_loss.backward()
                self.image_codec_aux_optim.step()

                psrn = 10 * np.log10(1.0 ** 2 / mse_loss.detach().cpu())

                train_psnr.update(psrn.mean().detach().item(), self.batch_size)
                train_bpp.update(bpp_loss.mean().detach().item(), self.batch_size)
                train_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('train_psnr', psrn.detach().item(), self.global_step)
                    self.writer.add_scalar('train_bpp', bpp_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_loss', rd_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)

                train_bar.desc = "TRAIN [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                    format(epoch + 1,
                           self.num_epochs,
                           rd_loss.mean().detach().item(),
                           psrn.mean().detach().item(),
                           bpp_loss.mean().detach().item(),
                           aux_loss.mean().detach().item()
                           )

                self.global_step += 1

                # if kk > 100:
                #     break

            self.logger.info("TRAIN [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_psnr.avg,
                                    train_bpp.avg,
                                    train_aux.avg
                                    )
                             )

            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            val_loss = self.validate()
            self.image_codec_lr_scheduler.step(val_loss)
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_loss = AverageMeter()
        psnr_dict = {k: AverageMeter() for k in range(self.graph.levels)}
        bpp_dict = {k: AverageMeter() for k in range(self.graph.levels)}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_loader)
            for k, image in enumerate(valid_bar):
                image = image.cuda()

                for s in range(0, self.graph.levels):
                    image_codec_output = self.graph(image, [s], 1.)
                    out_criterion = self.image_codec_criterion(image_codec_output, image)
                    rd_loss = out_criterion["loss"]
                    bpp_loss = out_criterion["bpp_loss"]
                    mse_loss = torch.mean(out_criterion["mse_loss"])

                    aux_loss = self.graph.aux_loss()
                    psrn = 10 * np.log10(1.0 ** 2 / mse_loss.detach().cpu())

                    val_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                    psnr_dict[s].update(psrn, self.batch_size)
                    bpp_dict[s].update(bpp_loss.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar(f'val_psnr_{s}', psrn.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_bpp_{s}', bpp_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_loss_{s}', rd_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_aux_{s}', aux_loss.detach().item(), self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                        format(self.global_epoch + 1,
                               self.num_epochs,
                               rd_loss.mean().detach().item(),
                               psrn.mean().detach().item(),
                               bpp_loss.mean().detach().item(),
                               aux_loss.mean().detach().item()
                               )

                self.global_eval_step += 1
                #
                # if k > 20:
                #     break

            self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
            for s in range(self.graph.levels):
                self.logger.info(f"Val [{self.graph.lmbda[s]}],\tPSNR [{psnr_dict[s].avg:.4f}],\tBpp [{bpp_dict[s].avg:.4f}]")

        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()
        return val_loss.avg

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        # self.image_codec_aux_optim.load_state_dict(checkpoint["aux_optimizer"])
        # self.image_codec_optim.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        if self.global_epoch < self.milestones[0]:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr
        elif self.milestones[0] <= self.global_epoch < self.milestones[1]:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr / 5.0
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr / 5.0
        else:
            self.image_codec_optim.param_groups[0]['lr'] = self.args.lr / 20.0
            self.image_codec_aux_optim.param_groups[0]['lr'] = self.args.lr / 20.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.image_codec_optim.state_dict(),
            "aux_optimizer": self.image_codec_aux_optim.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))


class FGainedImageCodecTrainer(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)

        self.grad_clip = 1.0

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"ImageFGained_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')

        # data
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        self.training_loader, self.valid_loader = get_loader(args.train_dataset_root,
                                                             args.val_dataset_root,
                                                             256,
                                                             args.batch_size,
                                                             args.num_workers
                                                             )

        self.logger.info(f'[*] Train File Account For {len(self.training_loader)}, '
                         f'val {len(self.valid_loader)}')

        self.graph = FGainedMeanScale().cuda()

        parameters, aux_parameters = [], []
        for n, p in self.graph.named_parameters():
            if not n.endswith(".quantiles"):
                parameters.append(p)
            else:
                aux_parameters.append(p)
        self.image_codec_optim = torch.optim.Adam(parameters, lr=args.lr)
        self.image_codec_aux_optim = torch.optim.Adam(aux_parameters, lr=args.lr)

        self.image_codec_criterion = RateDistortionLoss()
        self.image_codec_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.image_codec_optim, "min")

        # model
        self.mode_type = args.mode_type

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_loss = AverageMeter()
            train_bpp, train_aux = AverageMeter(), AverageMeter()
            train_psnr, train_msssim = AverageMeter(), AverageMeter()

            # adjust learning_rate
            self.graph.train()
            train_bar = tqdm(self.training_loader)
            for kk, image in enumerate(train_bar):
                self.image_codec_optim.zero_grad()
                self.image_codec_aux_optim.zero_grad()

                image = image.cuda()

                index_rand = np.random.randint(0, self.graph.levels - 1)
                image_codec_output = self.graph(image, index_rand)
                out_criterion = self.image_codec_criterion(image_codec_output, image, self.graph.lmbda[index_rand])
                rd_loss = out_criterion["loss"]
                bpp_loss = out_criterion["bpp_loss"]
                mse_loss = out_criterion["mse_loss"]
                rd_loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)

                self.image_codec_optim.step()
                aux_loss = self.graph.aux_loss()
                aux_loss.backward()
                self.image_codec_aux_optim.step()

                psrn = 10 * np.log10(1.0 ** 2 / mse_loss.detach().cpu())

                train_psnr.update(psrn.mean().detach().item(), self.batch_size)
                train_bpp.update(bpp_loss.mean().detach().item(), self.batch_size)
                train_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('train_psnr', psrn.detach().item(), self.global_step)
                    self.writer.add_scalar('train_bpp', bpp_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_loss', rd_loss.detach().item(), self.global_step)
                    self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)

                train_bar.desc = "TRAIN [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                    format(epoch + 1,
                           self.num_epochs,
                           rd_loss.mean().detach().item(),
                           psrn.mean().detach().item(),
                           bpp_loss.mean().detach().item(),
                           aux_loss.mean().detach().item()
                           )

                self.global_step += 1

                if kk > 20:
                    break

            self.logger.info("TRAIN [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_psnr.avg,
                                    train_bpp.avg,
                                    train_aux.avg
                                    )
                             )

            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            val_loss = self.validate()
            self.image_codec_lr_scheduler.step(val_loss)
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_loss = AverageMeter()
        psnr_dict = {k: AverageMeter() for k in range(self.graph.levels)}
        bpp_dict = {k: AverageMeter() for k in range(self.graph.levels)}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_loader)
            for k, image in enumerate(valid_bar):
                image = image.cuda()

                for index_rand in range(0, self.graph.levels - 1):
                    image_codec_output = self.graph(image, index_rand)
                    out_criterion = self.image_codec_criterion(image_codec_output, image, self.graph.lmbda[index_rand])
                    rd_loss = out_criterion["loss"]
                    bpp_loss = out_criterion["bpp_loss"]
                    mse_loss = out_criterion["mse_loss"]

                    aux_loss = self.graph.aux_loss()
                    psrn = 10 * np.log10(1.0 ** 2 / mse_loss.detach().cpu())

                    val_loss.update(rd_loss.mean().detach().item(), self.batch_size)
                    psnr_dict[index_rand].update(psrn, self.batch_size)
                    bpp_dict[index_rand].update(bpp_loss.mean().detach().item(), self.batch_size)
                    self.writer.add_scalar(f'val_psnr_{index_rand}', psrn.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_bpp_{index_rand}', bpp_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_loss_{index_rand}', rd_loss.detach().item(), self.global_eval_step)
                    self.writer.add_scalar(f'val_aux_{index_rand}', aux_loss.detach().item(), self.global_eval_step)

                    valid_bar.desc = "VALID [{}|{}] LOSS[{:.3f}], PSNR[{:.3f}], BPP[{:.3f}], AUX[{:.3f}]". \
                        format(self.global_epoch + 1,
                               self.num_epochs,
                               rd_loss.mean().detach().item(),
                               psrn.mean().detach().item(),
                               bpp_loss.mean().detach().item(),
                               aux_loss.mean().detach().item()
                               )

                self.global_eval_step += 1

                if k > 20:
                    break

            self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
            for s in range(self.graph.levels):
                self.logger.info(f"Val [{s}], PSNR [{psnr_dict[s].avg:.4f}], Bpp [{bpp_dict[s].avg:.4f}]")

        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()
        return val_loss.avg

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location=self.device)
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.image_codec_aux_optim.load_state_dict(checkpoint["aux_optimizer"])
        self.image_codec_optim.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.image_codec_optim.state_dict(),
            "aux_optimizer": self.image_codec_aux_optim.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))


#  Flexible rate  step1
class GainedVideoTrainer(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)
        self.milestones = [70, 90]
        # self.lambda_list = [1280, 640, 320, 160]
        # self.lambda_list = [160, 320, 640, 1280]
        # self.lambda_list = [80, 160, 320, 640]
        # self.lambda_list = [160, 320, 640, 1280, 2560]
        # self.lambda_list = [1280, 640, 320, 160, 80]

        # self.lambda_list = [70, 190, 460, 1024, 2260]
        self.lambda_list = [160, 380, 960, 1920, 3860]
        self.lmbda = torch.tensor(self.lambda_list).cuda()
        self.l = torch.arange(0, len(self.lambda_list)).cuda()

        self.stage1_step = 2e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 3frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        self.grad_clip = 1.0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"WOSM_VB_mbt2018_160_3840_{date}")
        # self.log_dir = os.path.join(args.log_root, f"WOSM_VB_TrainStep1_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')
        self.logger.info(f'[*] Training Lambdas {self.lambda_list}')
        # data
        self.frames = 5
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        # training_set, valid_set = get_dataset(args)
        training_set, valid_set = get_dataset11(args, mf=4, crop=True, worgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = int(self.stage4_step)
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        self.key_frame_models = {}
        self.logger.info(f"[*] Try Load Pretrained Image Codec Model...")
        for q in [4, 5, 6, 7, 8]:
            self.key_frame_models[q - 4] = mbt2018(q, 'mse', True)

        # for i, I_lambda in enumerate([0.0067, 0.013, 0.025, 0.0483]):
        # for i, I_lambda in enumerate([0.013, 0.025, 0.0483, 0.0932]):
        #     codec = ICIP2020ResB()
        #     # /tdx/LHB/pretrained/ICIP2020ResB/msssim_from0
        #     ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/mse/lambda_{I_lambda}.pth'
        #     state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        #     state_dict = load_pretrained(state_dict)
        #     codec.load_state_dict(state_dict)
        #     self.key_frame_models[i] = codec

        for q in self.key_frame_models.keys():
            for param in self.key_frame_models[q].parameters():
                param.requires_grad = False
            self.key_frame_models[q] = self.key_frame_models[q].eval().cuda()

        self.mode_type = args.mode_type
        self.graph = LHB_DVC_WOSM_VB().cuda()
        self.logger.info(f"[*] Try Load Pretrained Video Codec Model...")
        ckpt = '/tdx/LHB/code/torch/LHBDVC/checkpoints/LHB_DVC_WOSM_bpg2048.pth'
        tgt_model_dict = self.graph.state_dict()
        src_pretrained_dict = torch.load(ckpt)['state_dict']
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        # for k, v in src_pretrained_dict.items():
        #     if k not in tgt_model_dict:
        #         print(k)
        tgt_model_dict.update(_pretrained_dict)
        self.graph.load_state_dict(tgt_model_dict)

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        # self.validate()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_psnr, train_mc_psnr = AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp = AverageMeter(), AverageMeter()
            train_psnr, train_msssim, train_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                if self.global_step > 0 and self.global_step % 4e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                f = self.get_f()
                warp_weight = 0.2
                feature = None
                ref_frame = []
                with torch.no_grad():
                    # print(frames[0].shape)
                    for q in self.key_frame_models.keys():
                        # print(q)
                        _ref_frame = self.key_frame_models[q](frames[0][q].unsqueeze(0))['x_hat']
                        ref_frame.append(_ref_frame)
                    ref_frame = torch.cat(ref_frame, dim=0)
                    i_mse = torch.mean((frames[0] - ref_frame).pow(2))
                    i_psnr = 10 * np.log10(1.0 / i_mse.detach().cpu())
                    # print(i_psnr)
                    # exit()

                if 0 <= self.global_step < self.stage2_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, self.l, 1, feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            dist = warp_loss + mc_loss
                        else:
                            dist = mc_loss
                        distortion = mse_loss + warp_weight * dist
                        # print('distortion', distortion)
                        # print('lambda', self.lmbda)
                        # print(distortion * self.lmbda)
                        # print(torch.mean(distortion * self.lmbda))
                        # exit()
                        distortion = torch.mean(distortion * self.lmbda)
                        loss = distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        train_i_psnr.update(i_psnr, self.batch_size)
                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.2f}|{:.2f}|{:.2f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   i_psnr,
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _mse, _bpp, _aux_loss = torch.zeros(size=(len(self.lambda_list), )).cuda(), \
                                            torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, self.l, 1, feature)
                        # distortion = mse_loss + warp_weight * mc_loss
                        _mse += mse_loss * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        _loss = torch.mean(mse_loss) + bpp

                        train_i_psnr.update(i_psnr, self.batch_size)
                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                        self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                        self.writer.add_scalar('train_psnr', psrn, self.global_step)
                        self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_loss', _loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T{} [{}|{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.2f}|{:.2f}|{:.2f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   i_psnr,
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                    # print('distortion', _mse)
                    # print('lambda', self.lmbda)
                    # print(_mse[0] * self.lmbda[0], _mse[1] * self.lmbda[1])
                    # print(_mse * self.lmbda)
                    # print(_mse[0] * self.lmbda[0] + _mse[1] * self.lmbda[1] +
                    #       _mse[2] * self.lmbda[2] + _mse[3] * self.lmbda[3])
                    # print(torch.mean(_mse * self.lmbda))
                    # exit()
                    distortion = torch.mean(_mse * self.lmbda)
                    # print(distortion)
                    num = f * (f + 1) // 2
                    loss = distortion.div(num) + _bpp.div(num)
                    # print('===loss', loss)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    aux_loss = _aux_loss.div(num)
                    aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 20:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.2f}|{:.2f}|{:.2f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_psnr.avg,
                                    train_warp_psnr.avg,
                                    train_mc_psnr.avg,
                                    train_psnr.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            if epoch % self.args.val_freq == 0:
                self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()

        psnr_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        bpp_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        msssim_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                # ref_frame = []
                # for q in self.key_frame_models.keys():
                #     _ref_frame = self.key_frame_models[q](frames[0][q].unsqueeze(0))['x_hat']
                #     ref_frame.append(_ref_frame)
                # ref_frame = torch.cat(ref_frame, dim=0)
                # i_mse = torch.mean((frames[0] - ref_frame).pow(2))
                # i_psnr = 10 * np.log10(1.0 / i_mse.detach().cpu())
                # val_i_psnr.update(i_psnr, self.batch_size)

                f = self.get_f()
                for s in range(0, len(self.lambda_list)):
                    feature = None
                    ref_frame = self.key_frame_models[s](frames[0])['x_hat']
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()
                        distortion = torch.mean(mse_loss * self.lmbda)
                        loss = distortion + bpp
                        self.optimizer.zero_grad()

                        msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        mv_aux = self.graph.mv_aux_loss()
                        res_aux = self.graph.res_aux_loss()
                        aux = self.graph.aux_loss()

                        psnr_dict[s].update(psrn, self.batch_size)
                        bpp_dict[s].update(bpp.mean().detach().item(), self.batch_size)
                        msssim_dict[s].update(msssim.mean().detach().item(), self.batch_size)

                        # val_i_psnr.update(i_psnr, self.batch_size)
                        # val_loss.update(loss.mean().detach().item(), self.batch_size)
                        # val_warp_psnr.update(warp_psrn, self.batch_size)
                        # val_mc_psnr.update(mc_psrn, self.batch_size)
                        # val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        # val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        # val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        # val_msssim.update(msssim, self.batch_size)
                        # val_psnr.update(psrn, self.batch_size)
                        #
                        # val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                        # val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                        # val_aux.update(aux.mean().detach().item(), self.batch_size)
                        # self.writer.add_scalar('val_mv_aux', mv_aux.detach().item(), self.global_step)
                        # self.writer.add_scalar('val_res_aux', res_aux.detach().item(), self.global_step)
                        # self.writer.add_scalar('val_aux', aux.detach().item(), self.global_step)
                        # self.writer.add_scalar('val_psnr', psrn, self.global_step)
                        # self.writer.add_scalar('val_loss', loss.mean().detach().item(), self.global_eval_step)
                        # self.writer.add_scalar('val_warp_psnr', warp_psrn, self.global_eval_step)
                        # self.writer.add_scalar('val_mc_psnr', mc_psrn, self.global_eval_step)
                        # self.writer.add_scalar('val_bpp', bpp.mean().detach().item(), self.global_eval_step)
                        # self.writer.add_scalar('val_res_bpp', bpp_res.mean().detach().item(), self.global_eval_step)
                        # self.writer.add_scalar('val_mv_bpp', bpp_mv.mean().detach().item(), self.global_eval_step)
                        # self.writer.add_scalar('val_msssim', msssim, self.global_eval_step)
                        # self.writer.add_scalar('val_psnr', psrn, self.global_eval_step)

                        valid_bar.desc = "V{} [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                         "AUX[{:.2f}|{:.2f}|{:.2f}]".format(
                            f,
                            self.global_epoch + 1,
                            self.num_epochs,
                            loss.mean().detach().item(),
                            # i_psnr,
                            warp_psrn,
                            mc_psrn,
                            psrn,
                            bpp_mv.mean().detach().item(),
                            bpp_res.mean().detach().item(),
                            bpp.mean().detach().item(),
                            mv_aux.detach().item(),
                            res_aux.detach().item(),
                            aux.detach().item(),
                        )
                self.global_eval_step += 1

                if k > 800:
                    break

        # self.logger.info("VALID [{}|{}] LOSS[{:.4f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
        #                  "AUX[{:.3f}|{:.3f}|{:.3f}]". \
        #                  format(self.global_epoch + 1,
        #                         self.num_epochs,
        #                         val_loss.avg,
        #                         val_i_psnr.avg,
        #                         val_warp_psnr.avg,
        #                         val_mc_psnr.avg,
        #                         val_psnr.avg,
        #                         val_mv_bpp.avg,
        #                         val_res_bpp.avg,
        #                         val_bpp.avg,
        #                         val_mv_aux.avg,
        #                         val_res_aux.avg,
        #                         val_aux.avg
        #                         ))

        self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
        # for s in range(len(self.lambda_list)):
        #     self.logger.info(f"Val [{s}], PSNR [{psnr_dict[s].avg:.4f}], Bpp [{bpp_dict[s].avg:.4f}], "
        #                      f"MS-SSIM [{msssim_dict[s].avg:.4f}]")
        for s in range(len(self.lambda_list)):
            self.logger.info(
                f"Val [{self.lambda_list[s]:4d}],\tPSNR [{psnr_dict[s].avg:.4f}],\tBpp [{bpp_dict[s].avg:.4f}],\t"
                f"MS-SSIM [{msssim_dict[s].avg:.4f}]")

        is_best = bool(val_loss.avg < self.lowest_val_loss)
        self.lowest_val_loss = min(self.lowest_val_loss, val_loss.avg)
        self.save_checkpoint(val_loss.avg, "checkpoint.pth", is_best)
        self.graph.train()

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        # self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        # self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        if self.global_step >= int(546000 - 10):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        if self.global_step > int(546000 + 8e4):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        if self.global_step > int(546000 + 10e4):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        # elif self.stage2_step <= self.global_step <= self.stage3_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        # else:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = set(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 3
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 4
        else:
            f = 4
        return f


# sample single point
class GainedVideoTrainer0(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)
        # self.lambda_list = [128, 256, 512, 1024, 2048]
        # self.lambda_list = [256, 512, 1024, 2048]
        # self.lambda_list = [75, 200, 480, 1100, 2700]
        # self.lambda_list = [160, 320, 640, 1280, 2560]

        # self.lambda_list = [60, 180, 420, 920, 2260]
        # self.lambda_list = [60, 165, 370, 880, 1920]

        # self.lambda_list = [60, 140, 320, 720, 1790]
        self.lambda_list = [60, 140, 320, 720, 1800]
        self.stage1_step = 2e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 3frames
        self.stage3_step = self.stage2_step + 1e5  # 4frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        self.grad_clip = 1.0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        # self.log_dir = os.path.join(args.log_root, f"WOSM_VB_mbt2018_160_3840_{date}")
        self.log_dir = os.path.join(args.log_root, f"VB_icip_loadmbt192_{min(self.lambda_list)}_{max(self.lambda_list)}_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')
        self.logger.info(f'[*] Training Lambdas {self.lambda_list}')

        # data
        self.frames = 5
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        training_set, valid_set = get_dataset11(args, mf=5, crop=True, worgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = int(self.stage4_step)
        self.stop_count = 0

        self.key_frame_models = {}
        self.logger.info(f"[*] Try Load Pretrained Image Codec Model...")
        # for q in [4, 5, 6, 7, 8]:
        #     self.key_frame_models[q - 4] = mbt2018(q, 'mse', True)

        for i, I_lambda in enumerate([0.0067, 0.013, 0.025, 0.0483, 0.0932]):
            codec = ICIP2020ResB()
            # /tdx/LHB/pretrained/ICIP2020ResB/msssim_from0
            ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/mse/lambda_{I_lambda}.pth'
            state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
            state_dict = load_pretrained(state_dict)
            codec.load_state_dict(state_dict)
            self.key_frame_models[i] = codec

        for q in self.key_frame_models.keys():
            for param in self.key_frame_models[q].parameters():
                param.requires_grad = False
            self.key_frame_models[q] = self.key_frame_models[q].eval().cuda()

        self.mode_type = args.mode_type
        self.graph = LHB_DVC_WOSM_VB().cuda()
        self.logger.info(f"[*] Try Load Pretrained Video Codec Model...")
        ckpt = '/tdx/LHB/code/torch/LHBDVC/checkpoints/LHB_DVC_WOSM_bpg2048.pth'
        tgt_model_dict = self.graph.state_dict()
        src_pretrained_dict = torch.load(ckpt)['state_dict']
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        # for k, v in src_pretrained_dict.items():
        #     if k not in tgt_model_dict:
        #         print(k)
        tgt_model_dict.update(_pretrained_dict)
        self.graph.load_state_dict(tgt_model_dict)

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        # self.validate()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_psnr, train_mc_psnr = AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp = AverageMeter(), AverageMeter()
            train_psnr, train_msssim, train_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                if self.global_step > 0 and self.global_step % 4e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                f = self.get_f()

                if self.global_step > self.stage1_step + self.stage2_step // 2:
                    s = random_index([19, 16, 15, 16, 34])
                    # s = random_index([19, 20, 27, 34])  # 25,25,25,25
                else:
                    s = random.randint(0, len(self.lambda_list) - 1)

                feature = None
                with torch.no_grad():
                    ref_frame = self.key_frame_models[s](frames[0])['x_hat']
                # if s == 4:
                #     ref_frame = frames[0]
                # else:
                #     with torch.no_grad():
                #         ref_frame = self.key_frame_models[s](frames[1])['x_hat']

                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature, avg_dim=(0, 1, 2, 3))
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step // 2:
                            distortion = mse_loss + 0.1 * warp_loss + 0.2 * mc_loss
                        elif self.stage1_step // 2 <= self.global_epoch < self.stage1_step:
                            distortion = mse_loss + 0.2 * mc_loss
                        else:
                            distortion = mse_loss
                        loss = distortion * self.lambda_list[s] + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T{} [{}|{}|{}] [{}|{:4d}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   s,
                                   self.lambda_list[s],
                                   loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _mse, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature, avg_dim=(0, 1, 2, 3))
                        # distortion = mse_loss + warp_weight * mc_loss
                        _mse += mse_loss * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        _loss = torch.mean(mse_loss) + bpp

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                        self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                        self.writer.add_scalar('train_psnr', psrn, self.global_step)
                        self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_loss', _loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "Final{} [{}|{}|{}] [{}|{:4d}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   s,
                                   self.lambda_list[s],
                                   _loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                    distortion = _mse * self.lambda_list[s]
                    # print(distortion)
                    num = f * (f + 1) // 2
                    loss = distortion.div(num) + _bpp.div(num)
                    # print('===loss', loss)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    aux_loss = _aux_loss.div(num)
                    aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 20:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.2f}|{:.2f}|{:.2f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_psnr.avg,
                                    train_warp_psnr.avg,
                                    train_mc_psnr.avg,
                                    train_psnr.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            # if epoch % self.args.val_freq == 0:
            #     self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        psnr_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        bpp_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        msssim_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]

                f = self.get_f()
                for s in range(0, len(self.lambda_list)):
                    feature = None
                    # if s == 4:
                    #     ref_frame = frames[0]
                    # else:
                    ref_frame = self.key_frame_models[s](frames[0])['x_hat']
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()
                        distortion = mse_loss * self.lambda_list[s]
                        loss = distortion + bpp
                        self.optimizer.zero_grad()

                        msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        mv_aux = self.graph.mv_aux_loss()
                        res_aux = self.graph.res_aux_loss()
                        aux = self.graph.aux_loss()

                        psnr_dict[s].update(psrn, self.batch_size)
                        bpp_dict[s].update(bpp.mean().detach().item(), self.batch_size)
                        msssim_dict[s].update(msssim.mean().detach().item(), self.batch_size)

                        valid_bar.desc = "V{} {:4d} [{}|{}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                         "AUX[{:.1f}|{:.1f}|{:1f}]".format(
                            f,
                            self.lambda_list[s],
                            self.global_epoch + 1,
                            self.num_epochs,
                            loss.mean().detach().item(),
                            warp_psrn,
                            mc_psrn,
                            psrn,
                            bpp_mv.mean().detach().item(),
                            bpp_res.mean().detach().item(),
                            bpp.mean().detach().item(),
                            mv_aux.detach().item(),
                            res_aux.detach().item(),
                            aux.detach().item(),
                        )
                self.global_eval_step += 1

                # if k > 20:
                #     break

                if k > 1000:
                    break

        self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
        for s in range(len(self.lambda_list)):
            self.logger.info(
                f"Val [{self.lambda_list[s]:4d}],\tPSNR [{psnr_dict[s].avg:.4f}],\tBpp [{bpp_dict[s].avg:.4f}],\t"
                f"MS-SSIM [{msssim_dict[s].avg:.4f}]")

        self.save_checkpoint(0.0, "checkpoint.pth", False)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 3
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 5
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        last_step = checkpoint["global_step"]
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")
        self.logger.info(f"[*] Load Pretrained Model From Step {last_step}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        # self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # if self.global_step > self.stage2_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        # if self.global_step > self.stage3_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        # if self.global_step > self.stage4_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0

        if self.global_step > int(864000 - 5):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        if self.global_step > int(864000 + 3e4):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        if self.global_step > int(864000 + 5e4):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = set(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class GainedVideoTrainer0_SSIM(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)
        # self.lambda_list = [128, 256, 512, 1024, 2048]
        # self.lambda_list = [256, 512, 1024, 2048]

        factor = 50
        # self.lambda_list = [60 / 50, 140 / 50, 320 / 50, 720 / 50, 1800 / 50]
        self.lambda_list = [60 / factor, 125 / factor, 320 / factor, 720 / factor, 1550 / factor]
        # self.lambda_list = [80 / 50, 160 / 50, 320 / 50, 640 / 50]

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 2frames
        self.stage3_step = self.stage2_step + 1e5  # 3frames
        self.stage4_step = self.stage3_step + 1e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        # self.stage1_step = 10  # 2frames
        # self.stage2_step = self.stage1_step + 10  # 3frames
        # self.stage3_step = self.stage2_step + 10  # 4frames
        # self.stage4_step = self.stage3_step + 10  # 5frames
        # self.stage5_step = self.stage4_step + 10  # 5frames

        self.grad_clip = 1.0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        # self.log_dir = os.path.join(args.log_root, f"WOSM_VB_mbt2018_{date}")
        self.log_dir = os.path.join(args.log_root, f"VB_loadMSE_MSSSIM_{min(self.lambda_list):.1f}_{max(self.lambda_list):.1f}_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')
        self.logger.info(f'[*] Training Lambda {self.lambda_list}')

        # data
        self.frames = 5
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        training_set, valid_set = get_dataset11(args, mf=5, crop=True, worgi=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        self.key_frame_models = {}
        self.logger.info(f"[*] Try Load Pretrained Image Codec Model...")

        for i, I_lambda in enumerate([8.73, 16.64, 31.73, 60.5, 115.37]):
            codec = ICIP2020ResB()
            # /tdx/LHB/pretrained/ICIP2020ResB/msssim_from0
            ckpt = f'/tdx/LHB/pretrained/ICIP2020ResB/msssim_from0/lambda_{I_lambda}.pth'
            state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
            state_dict = load_pretrained(state_dict)
            codec.load_state_dict(state_dict)
            self.key_frame_models[i] = codec

        for q in self.key_frame_models.keys():
            for param in self.key_frame_models[q].parameters():
                param.requires_grad = False
            self.key_frame_models[q] = self.key_frame_models[q].eval().cuda()

        self.mode_type = args.mode_type
        self.graph = LHB_DVC_WOSM_VB().cuda()

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        # self.validate()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_msssim, train_mc_msssim, train_msssim = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp, train_i_msssim = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                if self.global_step > 0 and self.global_step % 4e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                f = self.get_f()

                if self.global_step > self.stage1_step + self.stage2_step // 2:
                    # s = random_index([17, 17, 19, 21, 26])
                    # s = random_index([20, 20, 27, 33])  # 25,25,25,25
                    # s = random_index([19, 16, 15, 16, 34])
                    s = random_index([31, 11, 11, 11, 36])
                    # s = random_index([20, 23, 27, 30])  # 25,25,25,25
                else:
                    s = random.randint(0, len(self.lambda_list) - 1)

                feature = None
                with torch.no_grad():
                    ref_frame = self.key_frame_models[s](frames[0])['x_hat']
                    i_msssim = ms_ssim(ref_frame, frames[0], 1.0)
                # if s == 4:
                #     ref_frame = frames[0]
                # else:
                #     with torch.no_grad():
                #         ref_frame = self.key_frame_models[s](frames[1])['x_hat']

                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        with torch.no_grad():
                            sm = self.sm_p(self.process(curr_frame.mul(255)), self.supp, mean=True)
                            self.supp = torch.cat([self.supp, sm[0]], 0)[self.batch_size:]

                        decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, sm[0], feature=feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step:
                            warp_weight = 0.2
                        else:
                            warp_weight = 0
                        distortion = (1 - msssim) + warp_weight * (2 - warp_msssim - mc_msssim)
                        loss = self.l_MSSSIM * distortion + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        train_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        train_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        train_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        train_msssim.update(msssim.mean().detach().item(), self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)

                        train_bar.desc = "T-ALL{} [{}|{}|{}] LOSS[{:.2f}], MS-SSIM[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   loss.mean().detach().item(),
                                   i_msssim.mean().detach().item(),
                                   warp_msssim.mean().detach().item(),
                                   mc_msssim.mean().detach().item(),
                                   msssim.mean().detach().item(),
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _msssim, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        # decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                        #     self.graph(ref_frame, curr_frame, [s], 1, feature)
                        ref_frame, feature, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, [s], 1, feature=feature)
                        # mu = torch.ones_like(i_msssim.mean().detach()) + \
                        #      (i_msssim.mean().detach() - msssim.mean().detach()) / i_msssim.mean().detach() * index
                        _msssim += (1 - msssim) * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss()

                        # print(msssim)
                        # print(warp_msssim)
                        # print(mc_msssim)
                        # exit()

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        _loss = self.l_MSSSIM * (1 - msssim) + bpp

                        train_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        train_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        train_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        train_msssim.update(msssim.mean().detach().item(), self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)

                        train_bar.desc = "Final{} [{}] LOSS[{:.1f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   # epoch + 1,
                                   # self.num_epochs,
                                   self.global_step,
                                   _loss.mean().detach().item(),
                                   i_msssim.mean().detach().item(),
                                   warp_msssim.mean().detach().item(),
                                   mc_msssim.mean().detach().item(),
                                   msssim.mean().detach().item(),
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )
                    # num = f * (f - 1) // 2
                    # loss = self.l_MSSSIM * _msssim.div(num) + _bpp.div(num)
                    # loss = self.l_MSSSIM * _msssim + _bpp
                    distortion = _msssim * self.lambda_list[s]
                    # print(distortion)
                    num = f * (f + 1) // 2
                    loss = distortion.div(num) + _bpp.div(num)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    _aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 10:
                #     break

            self.logger.info("T-ALL [{}|{}] LOSS[{:.4f}], MS-SSIM[{:.3f}|{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_i_msssim.avg,
                                    train_warp_msssim.avg,
                                    train_mc_msssim.avg,
                                    train_msssim.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            if epoch % self.args.val_freq == 0:
                self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        val_bpp, val_loss = AverageMeter(), AverageMeter()
        val_warp_msssim, val_mc_msssim = AverageMeter(), AverageMeter()
        val_res_bpp, val_mv_bpp = AverageMeter(), AverageMeter()
        val_msssim, val_i_msssim = AverageMeter(), AverageMeter()
        val_res_aux, val_mv_aux, val_aux = AverageMeter(), AverageMeter(), AverageMeter()

        psnr_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        bpp_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        msssim_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]

                f = self.get_f()
                for s in range(0, len(self.lambda_list)):
                    feature = None
                    ref_frame = self.key_frame_models[s](frames[0])['x_hat']
                    i_msssim = ms_ssim(ref_frame, frames[0], 1.0)

                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, msssim, warp_msssim, mc_msssim, bpp_res, bpp_mv, bpp = \
                            self.graph.forward_msssim(ref_frame, curr_frame, [s], 1, feature)
                        # decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                        #     self.graph(ref_frame, curr_frame, [s], 1, feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()
                        loss = self.lambda_list[s] * (1 - msssim) + bpp
                        mse_loss = torch.mean((decoded_frame - curr_frame).pow(2))

                        msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())

                        mv_aux = self.graph.mv_aux_loss()
                        res_aux = self.graph.res_aux_loss()
                        aux = self.graph.aux_loss()

                        psnr_dict[s].update(psrn, self.batch_size)
                        bpp_dict[s].update(bpp.mean().detach().item(), self.batch_size)
                        msssim_dict[s].update(msssim.mean().detach().item(), self.batch_size)

                        val_i_msssim.update(i_msssim.mean().detach().item(), self.batch_size)
                        val_loss.update(loss.mean().detach().item(), self.batch_size)
                        val_warp_msssim.update(warp_msssim.mean().detach().item(), self.batch_size)
                        val_mc_msssim.update(mc_msssim.mean().detach().item(), self.batch_size)
                        val_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        val_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        val_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        val_msssim.update(msssim.mean().detach().item(), self.batch_size)

                        val_mv_aux.update(mv_aux.mean().detach().item(), self.batch_size)
                        val_res_aux.update(res_aux.mean().detach().item(), self.batch_size)
                        val_aux.update(aux.mean().detach().item(), self.batch_size)

                        valid_bar.desc = "VALID [{}|{}] LOSS[{:.2f}], MS-SSIM[{:.2f}|{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}] AUX[{:.1f}|{:.1f}|{:.1f}]".format(
                            self.global_epoch + 1,
                            self.num_epochs,
                            loss.mean().detach().item(),
                            i_msssim.mean().detach().item(),
                            warp_msssim.mean().detach().item(),
                            mc_msssim.mean().detach().item(),
                            msssim.mean().detach().item(),
                            bpp_mv.mean().detach().item(),
                            bpp_res.mean().detach().item(),
                            bpp.mean().detach().item(),
                            mv_aux.detach().item(),
                            res_aux.detach().item(),
                            aux.detach().item(),
                        )

                self.global_eval_step += 1

                # if k > 10:
                #     break

                if k > 1000:
                    break

        self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
        for s in range(len(self.lambda_list)):
            self.logger.info(
                f"Val [{self.lambda_list[s]}],\tPSNR [{psnr_dict[s].avg:.4f}],\tBpp [{bpp_dict[s].avg:.4f}],\t"
                f"MS-SSIM [{msssim_dict[s].avg:.4f}]")

        self.save_checkpoint(0.0, "checkpoint.pth", False)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 3
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 5
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        last_step = checkpoint["global_step"]
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")
        self.logger.info(f"[*] Load Pretrained Model From Step {last_step}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        # self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        # if self.global_step > self.stage2_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        # if self.global_step > self.stage3_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        # if self.global_step > self.stage4_step:
        #     self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
        #     self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0

        if self.global_step > int(1516000 - 5):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        if self.global_step > int(1516000 + 5e4):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 5.0
        if self.global_step > int(1516000 + 8e4):
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 20.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 20.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        # if is_best:
        #     torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = set(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class GainedVideoTrainer1(object):
    def __init__(self, args):
        # args
        args.cuda = torch.cuda.is_available()
        self.args = args
        utils.fix_random_seed(args.seed)
        self.lambda_list = [128, 256, 512, 1024, 2048]

        self.stage1_step = 3e5  # 2frames
        self.stage2_step = self.stage1_step + 1e5  # 2frames
        self.stage3_step = self.stage2_step + 1.5e5  # 3frames
        self.stage4_step = self.stage3_step + 1.5e5  # 5frames
        self.stage5_step = self.stage4_step + 1e5  # 5frames

        self.grad_clip = 1.0

        # logs
        date = str(datetime.datetime.now())
        date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, f"WOSM_VB_bpg_{date}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.log_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.summary_dir, comment='info')

        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
        os.makedirs(os.path.join(self.log_dir, 'codes'), exist_ok=True)
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            os.makedirs(os.path.join(self.log_dir, 'codes', to_make))

        pyfiles = glob("./*.py")
        for py in pyfiles:
            shutil.copyfile(py, os.path.join(self.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                shutil.copyfile(py, os.path.join(self.log_dir, 'codes', py[2:]))

        # logger
        utils.setup_logger('base', self.log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'[*] Using GPU = {args.cuda}')
        self.logger.info(f'[*] Start Log To {self.log_dir}')
        self.logger.info(f"[*] Lambda_list {self.lambda_list}")

        # data
        self.frames = 5
        self.batch_size = args.batch_size
        self.Height, self.Width, self.Channel = self.args.image_size
        self.l_PSNR = args.l_PSNR
        self.l_MSSSIM = args.l_MSSSIM

        # training_set, valid_set = get_dataset(args)
        training_set, valid_set = get_dataset11(args, mf=5, crop=True)
        self.training_set_loader = DataLoader(training_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              )

        self.valid_set_loader = DataLoader(valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           )
        self.logger.info(f'[*] Train File Account For {len(training_set)}, val {len(valid_set)}')

        # epoch
        self.num_epochs = args.epochs
        self.start_epoch = 0
        self.global_step = 0  # int(self.stage2_step)
        self.global_eval_step = 0
        self.global_epoch = 0
        self.stop_count = 0

        self.mode_type = args.mode_type
        self.graph = LHB_DVC_WOSM_VB(len(self.lambda_list)).cuda()
        self.logger.info(f"[*] Try Load Pretrained Video Codec Model...")
        ckpt = './checkpoint/LHB_DVC_WOSM_bpg2048.pth'
        tgt_model_dict = self.graph.state_dict()
        src_pretrained_dict = torch.load(ckpt)['state_dict']
        _pretrained_dict = {k: v for k, v in src_pretrained_dict.items() if k in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.graph.load_state_dict(tgt_model_dict)

        # device
        self.cuda = args.use_gpu
        self.device = next(self.graph.parameters()).device
        self.logger.info(
            f'[*] Total Parameters = {sum(p.numel() for p in self.graph.parameters() if p.requires_grad)}')

        self.configure_optimizers(args)

        self.lowest_val_loss = float("inf")

        if args.load_pretrained:
            self.resume()
        else:
            self.logger.info("[*] Train From Scratch")

        with open(os.path.join(self.log_dir, 'setting.json'), 'w') as f:
            flags_dict = {k: vars(args)[k] for k in vars(args)}
            json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.global_epoch = epoch
            train_bpp, train_loss = AverageMeter(), AverageMeter()
            train_warp_psnr, train_mc_psnr = AverageMeter(), AverageMeter()
            train_res_bpp, train_mv_bpp = AverageMeter(), AverageMeter()
            train_psnr, train_msssim, train_i_psnr = AverageMeter(), AverageMeter(), AverageMeter()
            train_res_aux, train_mv_aux, train_aux = AverageMeter(), AverageMeter(), AverageMeter()

            self.adjust_lr()
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[*] lr = {lr}')
            self.graph.train()
            train_bar = tqdm(self.training_set_loader)
            for kk, batch in enumerate(train_bar):
                if self.global_step > 0 and self.global_step % 4e3 == 0:
                    self.save_checkpoint(train_loss.avg, f"step_{self.global_step}.pth", is_best=False)
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = frames[0]
                f = self.get_f()
                feature = None
                if self.global_step > self.stage1_step + self.stage2_step // 2:
                    s = random_index([17, 17, 19, 21, 26])
                else:
                    s = random.randint(0, len(self.lambda_list) - 1)

                if 0 <= self.global_step < self.stage4_step:
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature, avg_dim=(0, 1, 2, 3))
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()

                        if self.global_epoch < self.stage1_step // 2:
                            distortion = mse_loss + 0.1 * warp_loss + 0.2 * mc_loss
                        elif self.stage1_step // 2 <= self.global_epoch < self.stage1_step:
                            distortion = mse_loss + 0.2 * mc_loss
                        else:
                            distortion = mse_loss
                        loss = distortion * self.lambda_list[s] + bpp
                        self.optimizer.zero_grad()
                        self.aux_optimizer.zero_grad()
                        loss.backward()
                        self.clip_gradient(self.optimizer, self.grad_clip)
                        self.optimizer.step()
                        aux_loss = self.graph.aux_loss()
                        aux_loss.backward()
                        self.aux_optimizer.step()

                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        if self.global_step % 300 == 0:
                            self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                            self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                            self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                            self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                            self.writer.add_scalar('train_loss', loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "T{} [{}|{}|{}] [{}|{:4d}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   s,
                                   self.lambda_list[s],
                                   loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                        self.global_step += 1
                else:
                    _mse, _bpp, _aux_loss = torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()
                    for index in range(1, f):
                        curr_frame = frames[index]
                        ref_frame, feature, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature, avg_dim=(0, 1, 2, 3))
                        # distortion = mse_loss + warp_weight * mc_loss
                        _mse += mse_loss * index
                        _bpp += bpp * index
                        _aux_loss += self.graph.aux_loss() * index

                        aux_loss = self.graph.aux_loss()
                        res_aux_loss = self.graph.res_aux_loss()
                        mv_aux_loss = self.graph.mv_aux_loss()

                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        _loss = torch.mean(mse_loss) + bpp

                        train_mc_psnr.update(mc_psrn, self.batch_size)
                        train_warp_psnr.update(warp_psrn, self.batch_size)
                        train_psnr.update(psrn, self.batch_size)
                        train_mv_bpp.update(bpp_mv.mean().detach().item(), self.batch_size)
                        train_res_bpp.update(bpp_res.mean().detach().item(), self.batch_size)
                        train_bpp.update(bpp.mean().detach().item(), self.batch_size)
                        train_loss.update(_loss.mean().detach().item(), self.batch_size)
                        train_mv_aux.update(mv_aux_loss.mean().detach().item(), self.batch_size)
                        train_res_aux.update(res_aux_loss.mean().detach().item(), self.batch_size)
                        train_aux.update(aux_loss.mean().detach().item(), self.batch_size)
                        self.writer.add_scalar('train_mv_aux', mv_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_aux', res_aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_aux', aux_loss.detach().item(), self.global_step)
                        self.writer.add_scalar('train_mc_psnr', mc_psrn, self.global_step)
                        self.writer.add_scalar('train_warp_psnr', warp_psrn, self.global_step)
                        self.writer.add_scalar('train_psnr', psrn, self.global_step)
                        self.writer.add_scalar('train_mv_bpp', bpp_mv.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_res_bpp', bpp_res.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_bpp', bpp.mean().detach().item(), self.global_step)
                        self.writer.add_scalar('train_loss', _loss.mean().detach().item(), self.global_step)

                        train_bar.desc = "Final{} [{}|{}|{}] [{}|{:4d}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                                         "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.1f}|{:.1f}|{:.1f}]". \
                            format(f,
                                   epoch + 1,
                                   self.num_epochs,
                                   self.global_step,
                                   s,
                                   self.lambda_list[s],
                                   _loss.mean().detach().item(),
                                   warp_psrn,
                                   mc_psrn,
                                   psrn,
                                   bpp_mv.mean().detach().item(),
                                   bpp_res.mean().detach().item(),
                                   bpp.mean().detach().item(),
                                   mv_aux_loss.mean().detach().item(),
                                   res_aux_loss.mean().detach().item(),
                                   aux_loss.mean().detach().item()
                                   )

                    distortion = _mse * self.lambda_list[s]
                    # print(distortion)
                    num = f * (f + 1) // 2
                    loss = distortion.div(num) + _bpp.div(num)
                    # print('===loss', loss)

                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    self.clip_gradient(self.optimizer, self.grad_clip)
                    self.optimizer.step()
                    aux_loss = _aux_loss.div(num)
                    aux_loss.backward()
                    self.aux_optimizer.step()
                    self.global_step += 1

                # if kk > 20:
                #     break

            self.logger.info("TALL [{}|{}] LOSS[{:.2f}], PSNR[{:.3f}|{:.3f}|{:.3f}], " \
                             "BPP[{:.3f}|{:.3f}|{:.3f}], AUX[{:.2f}|{:.2f}|{:.2f}]". \
                             format(epoch + 1,
                                    self.num_epochs,
                                    train_loss.avg,
                                    train_warp_psnr.avg,
                                    train_mc_psnr.avg,
                                    train_psnr.avg,
                                    train_mv_bpp.avg,
                                    train_res_bpp.avg,
                                    train_bpp.avg,
                                    train_mv_aux.avg,
                                    train_res_aux.avg,
                                    train_aux.avg
                                    ))

            # Needs to be called once after training
            self.graph.update()
            self.save_checkpoint(train_loss.avg, f"checkpoint_{epoch}.pth", is_best=False)
            if epoch % self.args.val_freq == 0:
                self.validate()
        # Needs to be called once after training
        self.graph.update()

    def validate(self):
        self.graph.eval()
        psnr_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        bpp_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}
        msssim_dict = {k: AverageMeter() for k in range(len(self.lambda_list))}

        with torch.no_grad():
            valid_bar = tqdm(self.valid_set_loader)
            for k, batch in enumerate(valid_bar):
                frames = [frame.to(self.device) for frame in batch]
                ref_frame = frames[0]

                f = self.get_f()
                for s in range(0, len(self.lambda_list)):
                    feature = None
                    for frame_index in range(1, f):
                        curr_frame = frames[frame_index]
                        decoded_frame, feature1, mse_loss, warp_loss, mc_loss, bpp_res, bpp_mv, bpp = \
                            self.graph(ref_frame, curr_frame, [s], 1, feature)
                        ref_frame = decoded_frame.detach().clone()
                        feature = feature1.detach().clone()
                        distortion = mse_loss * self.lambda_list[s]
                        loss = distortion + bpp
                        self.optimizer.zero_grad()

                        msssim = ms_ssim(curr_frame.detach(), decoded_frame.detach(), data_range=1.0)
                        psrn = 10 * np.log10(1.0 / torch.mean(mse_loss).detach().cpu())
                        mc_psrn = 10 * np.log10(1.0 / torch.mean(mc_loss).detach().cpu())
                        warp_psrn = 10 * np.log10(1.0 / torch.mean(warp_loss).detach().cpu())

                        mv_aux = self.graph.mv_aux_loss()
                        res_aux = self.graph.res_aux_loss()
                        aux = self.graph.aux_loss()

                        psnr_dict[s].update(psrn, self.batch_size)
                        bpp_dict[s].update(bpp.mean().detach().item(), self.batch_size)
                        msssim_dict[s].update(msssim.mean().detach().item(), self.batch_size)

                        valid_bar.desc = "V{} {:4d} [{}|{}] LOSS[{:.1f}], PSNR[{:.3f}|{:.3f}|{:.3f}], BPP[{:.3f}|{:.3f}|{:.3f}] " \
                                         "AUX[{:.1f}|{:.1f}|{:1f}]".format(
                            f,
                            self.lambda_list[s],
                            self.global_epoch + 1,
                            self.num_epochs,
                            loss.mean().detach().item(),
                            warp_psrn,
                            mc_psrn,
                            psrn,
                            bpp_mv.mean().detach().item(),
                            bpp_res.mean().detach().item(),
                            bpp.mean().detach().item(),
                            mv_aux.detach().item(),
                            res_aux.detach().item(),
                            aux.detach().item(),
                        )
                self.global_eval_step += 1

                # if k > 10:
                #     break

                if k > 600:
                    break

        self.logger.info(f"VALID [{self.global_epoch + 1}|{self.num_epochs}]")
        for s in range(len(self.lambda_list)):
            self.logger.info(f"Val [{self.lambda_list[s]:4d}],\tPSNR [{psnr_dict[s].avg:.4f}],\tBpp [{bpp_dict[s].avg:.4f}],\t"
                             f"MS-SSIM [{msssim_dict[s].avg:.4f}]")

        self.save_checkpoint(0.0, "checkpoint.pth", False)
        self.graph.train()

    def get_f(self):
        if self.global_step < self.stage2_step:
            f = 2
        elif self.stage2_step < self.global_step < self.stage3_step:
            f = 3
        elif self.stage3_step < self.global_step < self.stage4_step:
            f = 5
        else:
            f = 5
        return f

    def resume(self):
        self.logger.info(f"[*] Try Load Pretrained Model From {self.args.model_restore_path}...")
        checkpoint = torch.load(self.args.model_restore_path, map_location='cpu')
        last_epoch = checkpoint["epoch"] + 1
        last_step = checkpoint["global_step"]
        self.logger.info(f"[*] Load Pretrained Model From Epoch {last_epoch}...")
        self.logger.info(f"[*] Load Pretrained Model From Step {last_step}...")

        self.graph.load_state_dict(checkpoint["state_dict"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = last_epoch
        self.global_step = checkpoint["global_step"] + 1
        del checkpoint

    def adjust_lr(self):
        if self.global_step > self.stage2_step:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 2.0
        if self.global_step > self.stage3_step:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 10.0
        if self.global_step > self.stage4_step:
            self.optimizer.param_groups[0]['lr'] = self.args.lr / 50.0
            self.aux_optimizer.param_groups[0]['lr'] = self.args.lr / 50.0

    def save_checkpoint(self, loss, name, is_best):
        state = {
            "epoch": self.global_epoch,
            "global_step": self.global_step,
            "state_dict": self.graph.state_dict(),
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.checkpoints_dir, name))
        if is_best:
            torch.save(state, os.path.join(self.checkpoints_dir, "checkpoint_best_loss.pth"))

    def configure_optimizers(self, args):
        bp_parameters = set(p for n, p in self.graph.named_parameters() if not n.endswith(".quantiles"))
        aux_parameters = set(p for n, p in self.graph.named_parameters() if n.endswith(".quantiles"))
        self.optimizer = torch.optim.Adam(bp_parameters, lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(aux_parameters, lr=args.aux_lr)
        return None

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


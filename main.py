import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import utils
import Learner

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# tensorboard  --logdir test --samples_per_plugin scalars=999999999


def main():
    args = utils.get_args()
    # Learner.GainedImageCodecTrainer0(args).train()
    # Learner.GainedVideoTrainer(args).train()
    # Learner.GainedVideoTrainer0(args).train()
    Learner.GainedVideoTrainer0_SSIM(args).train()

    # For video
    # SSIM load  MSE10
    # Learner.GainedVideoTrainer0_SSIM(args).train()
    # MSE1
    # Learner.GainedVideoTrainer(args).train()
    # MSE0
    # Learner.GainedVideoTrainer0(args).train()
    return 0


if __name__ == "__main__":
    main()

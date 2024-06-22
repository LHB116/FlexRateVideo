import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch
from glob import glob
from torchvision.transforms.functional import hflip, to_tensor
from torch.distributions.multivariate_normal import MultivariateNormal


class VimeoDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        assert split == 'train' or 'test'
        if transform is None:
            raise Exception("Transform must be applied")

        self.max_frames = 5  # for Vimeo DataSet
        self.transform = transform
        self.file_name_list = os.path.join(root, f'sep_{split}list.txt')
        self.frames_dir = [os.path.join(root, 'sequences', x.strip())
                           for x in open(self.file_name_list, "r").readlines()]

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index]
        frame_paths = []
        for i in range(self.max_frames):
            frame_paths.append(os.path.join(sample_folder, f'im{i + 1}.png'))

        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        frames = self.transform(frames)
        frames = torch.chunk(frames, chunks=self.max_frames, dim=0)
        return frames

    def __len__(self):
        return len(self.frames_dir)


class VimeoImageDataset(Dataset):
    def __init__(self, root, split="train", image_size=256):
        assert split == 'train' or 'test'
        self.image_size = image_size
        self.file_name_list = os.path.join(root, f'sep_{split}list.txt')
        self.frames_dir = [os.path.join(root, 'sequences', x.strip())
                           for x in open(self.file_name_list, "r").readlines()]

    def __getitem__(self, index):
        sample_folder = self.frames_dir[index]
        index = np.random.randint(1, 7)
        image = Image.open(os.path.join(sample_folder, f'im{index}.png')).convert("RGB")
        transform = transforms.Compose([
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)
        return image

    def __len__(self):
        return len(self.frames_dir)


class ImageDatasets(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


class ImageDatasets1(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size, num_workers):
    train_dataset1 = ImageDatasets('//home/tdx/桌面/Project/LHB/data/vimeo_interp_train', image_size)
    train_dataset2 = ImageDatasets('/home/tdx/桌面/Project/LHB/data/ICLC2020/train', image_size)
    train_dataset3 = ImageDatasets('/home/tdx/桌面/Project/LHB/data/flicker_2W_images', image_size)
    train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])
    val_dataset = ImageDatasets1('/tdx/LHB/code/torch/IMC/data/image/kodim')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             )

    return train_loader, val_loader


def get_dataset(args, part='all'):
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(args.image_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(args.image_size[0]),
        ]
    )

    training_set = VimeoDataset(root=args.dataset_root,
                                transform=train_transforms,
                                split="train",
                                )
    valid_set = VimeoDataset(root=args.dataset_root,
                             transform=test_transforms,
                             split="test",
                             )
    if part == 'all':
        return training_set, valid_set
    elif part == 'train':
        return training_set
    elif part == 'valid':
        return valid_set


class QualityMapDataset(Dataset):
    def __init__(self, path, cropsize=256, mode='train', level_range=(0, 100), level=0, p=0.3, logger_handle=None):
        self.data_dir = path
        self.paths = sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.cropsize = cropsize
        self.mode = mode
        self.level_range = level_range
        self.level = level
        self.p = p
        self.grid = self._get_grid((self.cropsize, cropsize))
        self.qlevel_init = np.zeros_like(np.zeros((cropsize, cropsize), dtype=np.uint8), dtype=float)
        self.logger_handle = logger_handle
        if self.mode == 'train':
            # print(f'[{mode}set] {len(self.paths)} images')
            if self.logger_handle is not None:
                self.logger_handle.info(f'[{mode}-set] {len(self.paths)} images')
            else:
                print(f'[{mode}set] {len(self.paths)} images')
        elif self.mode == 'test':
            # print(f'[{mode}set] {len(self.paths)} images for quality {level / level_range[1]}')
            if self.logger_handle is not None:
                self.logger_handle.info(f'[{mode}-set] {len(self.paths)} images for quality {level / 100}')
            else:
                print(f'[{mode}set] {len(self.paths)} images for quality {level / 100}')

    def __len__(self):
        return len(self.paths)

    def _get_crop_params(self, img):
        w, h = img.size
        if w == self.cropsize and h == self.cropsize:
            return 0, 0, h, w

        if self.mode == 'train':
            top = random.randint(0, h - self.cropsize)
            left = random.randint(0, w - self.cropsize)
        else:
            # center
            top = int(round((h - self.cropsize) / 2.))
            left = int(round((w - self.cropsize) / 2.))
        return top, left

    def _get_grid(self, size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)

        grid1 = grid_x1.view(size[0], size[1], 1)
        grid2 = grid_x2.view(size[0], size[1], 1)
        grid = torch.cat([grid1, grid2], dim=-1)
        return grid

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')

        # crop if training
        if self.mode == 'train':
            top, left = self._get_crop_params(img)
            region = (left, top, left + self.cropsize, top + self.cropsize)
            img = img.crop(region)
        # horizontal flip
        if random.random() < 0.5 and self.mode == 'train':
            img = hflip(img)
        if self.mode == "train":
            qlevel = self.qlevel_init
        else:
            segqlevel = np.zeros(img.size[::-1], dtype=np.uint8)
            qlevel = np.zeros_like(segqlevel, dtype=float)
        if self.mode == 'train':
            sample = random.random()
            if sample < self.p:
                # uniform
                if random.random() < 0.01:
                    qlevel[:] = 0
                else:
                    qlevel[:] = (self.level_range[1] + 1) * random.random()
            elif sample < 2 * self.p:
                # gradation between two levels
                v1 = random.random() * self.level_range[1]
                v2 = random.random() * self.level_range[1]
                qlevel = np.tile(np.linspace(v1, v2, self.cropsize), (self.cropsize, 1)).astype(float)
                if random.random() < 0.5:
                    qlevel = qlevel.T
            else:
                # gaussian kernel
                gaussian_num = int(1 + random.random() * 20)
                for i in range(gaussian_num):
                    mu_x = self.cropsize * random.random()
                    mu_y = self.cropsize * random.random()
                    var_x = 2000 * random.random() + 1000
                    var_y = 2000 * random.random() + 1000

                    m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                    p = m.log_prob(self.grid)
                    kernel = torch.exp(p).numpy()
                    qlevel += kernel
                qlevel *= 100 / qlevel.max() * (0.5 * random.random() + 0.5)
        else:
            # uniques.sort()
            if self.level == -100:
                w, h = img.size
                # gradation
                if idx % 3 == 0:
                    v1 = idx / len(self.paths) * self.level_range[1]
                    v2 = (1 - idx / len(self.paths)) * self.level_range[1]
                    qlevel = np.tile(np.linspace(v1, v2, w), (h, 1)).astype(float)
                # gaussian kernel
                else:
                    gaussian_num = 1
                    for i in range(gaussian_num):
                        mu_x = h / 4 + (h / 2) * idx / len(self.paths)
                        mu_y = w / 4 + (w / 2) * (1 - idx / len(self.paths))
                        var_x = 20000 * (1 - idx / len(self.paths)) + 5000
                        var_y = 20000 * idx / len(self.paths) + 5000

                        m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                        grid = self._get_grid((h, w))
                        p = m.log_prob(grid)
                        kernel = torch.exp(p).numpy()
                        qlevel += kernel
                    qlevel *= 100 / qlevel.max() * (0.4 * idx / len(self.paths) + 0.6)
            else:
                # uniform level
                qlevel[:] = self.level

        # to tensor
        img = to_tensor(img)
        qlevel = torch.FloatTensor(qlevel).unsqueeze(dim=0)
        qlevel *= 1 / self.level_range[1]  # 0~100 -> 0~1
        return img, qlevel


def get_vb_dataloader(path1, path2, batch_size=16, L=10, logger_handle=None):
    #     train_dataset1 = Datasets1('/home/user/桌面/LHB/vimeo_interp_train', train_transforms)
    #     train_dataset2 = Datasets1('/home/user/桌面/zzc/CLIC+Flicker/flicker_2W_images', train_transforms)
    #     train_dataset3 = Datasets1('/home/user/桌面/LHB/ICLC/train', train_transforms)
    #     train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])
    train_dataset1 = QualityMapDataset('/home/user/桌面/LHB/vimeo_interp_train', mode='train',
                                       logger_handle=logger_handle)
    train_dataset2 = QualityMapDataset('/home/user/桌面/zzc/CLIC+Flicker/flicker_2W_images', mode='train',
                                       logger_handle=logger_handle)
    train_dataset3 = QualityMapDataset('/home/user/桌面/LHB/ICLC/train', mode='train',
                                       logger_handle=logger_handle)
    train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True)
    levels = [-100] + [int(100 * (i / L)) for i in range(L + 1)]
    test_dataloaders = []
    for level in levels:
        test_dataset = QualityMapDataset(path2, mode='test', level=level, logger_handle=logger_handle)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        test_dataloaders.append(test_dataloader)
    return train_dataloader, test_dataloaders


if __name__ == "__main__":
    pass
    size = 256

    transforms1 = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(size)]
    )

    train, test = get_vb_dataloader('D:/DataSet/Flicker/train', 'D:/DataSet/Flicker/val')
    print(len(train), len(test[0]))

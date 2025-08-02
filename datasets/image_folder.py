import os
import json
from PIL import Image
import pdb
import pickle
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register

from utils import make_coord


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache=None, rank=0, ngpu=1):

        self.repeat = repeat
        self.cache = cache
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        if ngpu != 1:
            chop = len(filenames) // ngpu
            filenames = filenames[rank * chop:(rank + 1) * chop]

        self.files = []
        for filename in filenames:
            if filename == 'bin':
                continue
            file = os.path.join(root_path, filename)

            if cache is None:
                self.files.append(file)
            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.cache is None:
            return transforms.ToTensor()(Image.open(x))
        elif self.cache == 'in_memory':
            return x


@register('image-volume')
class ImageVolume(Dataset):
    def __init__(self, root_path, sdf_path=None, first_k=None, repeat=1, rank=0, ngpu=1):
        self.image = io.imread(root_path)
        self.repeat = repeat
        self.sdf = None

        if sdf_path is not None:
            print(f"正在加载SDF数据从: {sdf_path}")
            self.sdf = io.imread(sdf_path)
            # 关键检查：确保图像和SDF体数据的形状匹配
            assert self.image.shape == self.sdf.shape, \
                f"图像形状 {self.image.shape} 和 SDF形状 {self.sdf.shape} 必须一致。"

        self.length, _, _ = self.image.shape

        if first_k is not None:
            self.image = self.image[:first_k]
            if self.sdf is not None:
                self.sdf = self.sdf[:first_k]

    def __len__(self):
        return self.length * self.repeat

    def __getitem__(self, idx):
        img_slice = self.image[idx % self.length]

        item = {
            'img': transforms.ToTensor()(Image.fromarray(img_slice, mode='L'))
        }

        if self.sdf is not None:
            sdf_slice = self.sdf[idx % self.length]
            # SDF值是浮点数，直接转换为Tensor
            item['sdf'] = torch.from_numpy(sdf_slice.astype(np.float32)).unsqueeze(0)

        return item  # 返回一个字典，包含图像和SDF数据（如果存在）


@register('image-test')
class ImageTest(Dataset):
    def __init__(self, root_path, direction, scale, rank=0, ngpu=1):
        self.image = io.imread(root_path)
        self.shape = self.image.shape
        self.direction = direction
        self.scale = scale

        if direction not in ['xy', 'xz', 'yz']:
            raise ValueError("direction should be xz or yz or xy")

        if self.direction == 'xz':
            self.length = self.shape[1]
        elif self.direction == 'yz':
            self.length = self.shape[2]
        else:
            self.length = self.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.direction == 'xz':
            x = self.image[:, idx % self.length, :]
        elif self.direction == 'yz':
            x = self.image[:, :, idx % self.length]
        else:
            x = self.image[idx % self.length]

        h, w = x.shape
        h = round(h * self.scale)
        coord = make_coord((h, w))
        cell = torch.ones_like(coord)

        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        return {
            'inp': transforms.ToTensor()(Image.fromarray(x, mode='L')),
            'coord': coord,
            'cell': cell,
            'shape':[h, w]
        }


@register('paired-image-folders')
class PairedImageFolders(Dataset):
    def __init__(self, root_path_1, root_path_2, rank=0, ngpu=1, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, rank=rank, ngpu=ngpu, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, rank=rank, ngpu=ngpu, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

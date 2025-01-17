import os
import json
from PIL import Image
from skimage import io
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
    def __init__(self, root_path, first_k=None, repeat=1, rank=0, ngpu=1):
        self.image = io.imread(root_path)
        self.repeat = repeat

        self.length, _, _ = self.image.shape

        if first_k is not None:
            self.image = self.image[:first_k]

    def __len__(self):
        return self.length * self.repeat

    def __getitem__(self, idx):
        x = self.image[idx % self.length]
        return transforms.ToTensor()(Image.fromarray(x, mode='L'))


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
            'h': h,
            'w': w
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

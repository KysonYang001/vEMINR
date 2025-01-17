import argparse

import numpy as np
import yaml
import pdb
from tqdm import tqdm

import datasets
import models
import utils
from models.controller import *
from torchvision import transforms
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage import io


def batched_predict(model, inp, coord, cell, bsize):
    n = coord.shape[1]
    ql = 0
    preds = []

    # w = utils.wave(inp, model.wav)
    w = inp
    feature = model.encoder(w)  # fix
    model.SR.gen_feat(inp)

    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.SR.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :], feature)
        preds.append(pred)
        ql = qr
    pred = torch.cat(preds, dim=1)
    return pred




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', help='model config file path', default='')
    parser.add_argument('--model_weight', default='your path to model_weight.pth')
    parser.add_argument('--save', help='your path to save reconstructed image', default='')
    parser.add_argument('--test_config', help='test config file path', default='')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.test_config, 'r') as f:
        testconfig = yaml.load(f, Loader=yaml.FullLoader)

    direct = testconfig['test_dataset']['dataset']['args']['direction']
    scale = testconfig['test_dataset']['dataset']['args']['scale']

    loader = datasets.make_data_loaders(testconfig, DDP=False, state='test')

    if direct not in ['xz', 'yz']:
        raise ValueError('direction should be xz or yz')

    with open(args.model_config, 'r') as f:
        modelconfig = yaml.load(f, Loader=yaml.FullLoader)

    model = models.make(modelconfig['model'], args={'config': modelconfig}, load_sd=args.model_weight).to(device)

    length = len(loader)
    volume = None
    idx = 0
    for batch in tqdm(loader, leave=False, desc='train'):
        inp, coord, cell, h, w = batch['inp'], batch['coord'], batch['cell'], batch['h'][0], batch['w'][0]
        inp, coord, cell = inp.to(device), coord.to(device), cell.to(device)

        model.eval()
        with torch.no_grad():
            pred = batched_predict(model, ((inp - 0.5) / 0.5), coord, cell, bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 1).permute(2, 0, 1).cpu().numpy() * 255
        pred = pred.astype(np.uint8)

        if volume is None:
            if direct == 'yz':
                volume = np.zeros((h, w, length), dtype=np.uint8)
            else:
                volume = np.zeros((h, length, w), dtype=np.uint8)

        if direct == 'yz':
            volume[:, :, idx] = pred
        else:
            volume[:, idx, :] = pred

        idx = idx+1

    io.imsave(args.save, volume)
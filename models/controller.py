import pdb

import torch
import torch.nn as nn

import models
from models import register
from utils import wave, make_coord
from datasets import degrade


@register('models')
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        spec = config['model']
        self.encoder = models.make(spec['degrade'], load_sd=spec['path'], freeze=True, key='degrade').to(self.device)
        self.SR = models.make(spec['SR']).to(self.device)

        if config.get('data_norm') is None:
            config['data_norm'] = {
                'inp': {'sub': [0], 'div': [1]},
                'gt': {'sub': [0], 'div': [1]}
            }
        # data normalize
        data_norm = config['data_norm']
        t = data_norm['inp']
        self.inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(self.device)
        self.inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(self.device)
        t = data_norm['gt']
        self.gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(self.device)
        self.gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(self.device)
        if config.get('sdf_norm'):
            t = config['sdf_norm']
            self.sdf_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(self.device)
            self.sdf_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(self.device)

    def forward(self, lr, coord=None, cell=None, scale=1, kernel=None, state='test'):
        with torch.no_grad():
            feature = self.encoder(lr)  # fix

        inp = (lr - self.inp_sub) / self.inp_div
        pred_rgb = self.SR(inp, coord, cell, feature) # self.SR(LIIF)现在只返回RGB
        pred_rgb = pred_rgb * self.gt_div + self.gt_sub
        pred_rgb.clamp_(0, 1)

        return pred_rgb
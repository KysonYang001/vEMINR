import argparse
import os
import yaml
from tqdm import tqdm
from show_utils import show_imgs
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import test
from datasets.degrade import SRMDPreprocessing
import models
import utils
from models.controller import *
import warnings
warnings.filterwarnings("ignore", message="FALLBACK path has been taken")

class Trainer():
    def __init__(self, config, args):
        #### device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        #### dataloader
        self.train_loader1, self.val_loader1 = datasets.make_data_loaders(config, False, 0, 1, state='SR')

        #### prepair training
        # model/optimzer/lr sceduler
        self.model = models.make(config['model'], args={'config': config}).to(self.device)

        print('model: #params={}'.format(utils.compute_num_params(self.model, text=True)))

        self.criterion = nn.L1Loss()
        self.optimizer = utils.make_optimizer(self.model.parameters(), config['optimizer'])
        # infer learning rate before changing batch size
        if config.get('multi_step_lr') is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = MultiStepLR(self.optimizer, **config['multi_step_lr'])

        # else

        self.args = args
        self.config = config
        self.epoch_save = config.get('epoch_save')
        self.epoch_val = config.get('epoch_val')
        self.max_val_v = -1e18
        self.bs = config['batch_size']
        self.degrade = SRMDPreprocessing()
        self.min_loss = 1e18
        self.scale = config['downsample_scale']
        self.inp_size = config['inp_size']

    def preprocess(self, batch, scale):
        # show_imgs(batch['inp'].cpu())
        lr = self.degrade(batch['inp'], scale, norm=True)  # bn, c, h, w
        # show_imgs(lr.cpu())
        return lr

    def data(self, batch1, scale):
        lr = self.preprocess(batch1, scale)
        gt = batch1['gt']
        cell = batch1['cell']
        coord = batch1['coord']
        return lr, gt, cell, coord

    def val(self, eval_bsize=None):
        self.model.eval()
        metric_fn = utils.calc_psnr
        val_res = utils.Averager()
        for batch1 in tqdm(self.val_loader1, leave=False, desc='val'):
            for k1, v1 in batch1.items():
                batch1[k1] = v1.to(self.device)

            lr, gt, cell, coord= self.data(batch1, self.scale)
            with torch.no_grad():
                if eval_bsize is None:
                    pred = self.model(lr, coord, cell)
                else:
                    inp = (lr - 0.5) / 0.5
                    pred = test.batched_predict(self.model, inp, coord, cell, eval_bsize)
                    pred = pred * 0.5 + 0.5
                    pred.clamp_(0, 1)
            res = metric_fn(pred, gt)
            val_res.add(res.item(), self.bs)
        return val_res.item()

    def train(self, epoch, timer):
        # initial
        t_epoch_start = timer.t()
        losses = utils.Averager()

        # train model
        self.model.train()

        for batch1 in tqdm(self.train_loader1, leave=False, desc='train'):

            for k1, v1 in batch1.items():
                batch1[k1] = v1.to(self.device)

            lr, gt, cell, coord = self.data(batch1, self.scale)

            # [0, 1] to [-1, 1]
            lr = (lr - 0.5) / 0.5
            gt = (gt - 0.5) / 0.5

            self.optimizer.zero_grad()
            pred_rgb = self.model(lr, coord, cell, state='train')

            loss = self.criterion(pred_rgb, gt)
            losses.add(loss.item())
            loss.backward()
            self.optimizer.step()

        # lr_scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # timer stop
        t = timer.t()
        prog = (epoch - 1 + 1) / (self.config['epoch_max'] - 1 + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        # validation
        if (self.epoch_val is not None) and (epoch % self.epoch_val == 0):
            val_res = self.val(self.config.get('eval_bsize'))

            res = 'epoch {}/{}, loss:{:.4f}, val:{:.4f} , {} {}/{}\n' \
                .format(epoch, self.config['epoch_max'], losses.item(), val_res, t_epoch, t_elapsed, t_all)
            tqdm.write(res)

            if val_res > self.max_val_v:
                self.max_val_v = val_res
                utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, False,
                           self.args.savedir, 'best')

            # if losses.item() < self.min_loss:
            #     self.min_loss = losses.item()
            #     utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP,
            #                self.args.savepath, 'best-loss')
            # write log


        # save file
        utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, False, self.args.savedir,
                   'last')
        if (self.epoch_save is not None) and (epoch % self.epoch_save == 0):
            utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, False,
                       self.args.savedir, 'hun')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path', default='configs/train_INR-rdn-liif-volume.yaml')
    parser.add_argument('--savedir', help='your path to save model directory', default="checkpoints/SR")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    #### read config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
        save_name = args.savedir
        if save_name is None:
            save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
        if os.path.exists(args.savedir) is False:
            os.makedirs(args.savedir)
            print('{} succeed'.format(args.savedir))
    torch.manual_seed(config['seed'])

    #### log file
    with open(os.path.join(args.savedir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    t = Trainer(config, args)
    # train epochs
    timer = utils.Timer()
    for epoch in range(1, config['epoch_max'] + 1):
        t.train(epoch, timer)
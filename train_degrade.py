import os
import pdb

import argparse
import yaml
from tqdm import tqdm

import utils
import datasets
from datasets.degrade import SRMDPreprocessing
import models
from datasets.queue import dequeue_and_enqueue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR


class Trainer():
    def __init__(self, config, args):
        #### distributed data parallel
        if args.DDP:
            dist.init_process_group(backend='nccl')
            dist.barrier()
            self.local_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
            ngpu = dist.get_world_size()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.local_rank = 0  # just for dataloader
            ngpu = 1

        #### dataloader
        self.loader1 = datasets.make_data_loaders(config, args.DDP, self.local_rank, ngpu, state='degrade')
        if args.queue:
            self.pool = dequeue_and_enqueue(config, 'degrade').to(self.device)

        #### prepair training
        # model/optimzer/lr sceduler
        self.model = models.make(config['degrade']).to(self.device)
        if not args.DDP or dist.get_rank() == 0:
            print('model: #params={}'.format(utils.compute_num_params(self.model, text=True)))

        if args.DDP:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            # Apply SyncBN
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.L1 = nn.L1Loss()

        self.weight = torch.tensor(
            [[2, 2, 2, 2, 2], [2, 3, 3, 3, 2], [2, 3, 4, 3, 2], [2, 3, 3, 3, 2], [2, 2, 2, 2, 2]])
        self.weight = F.pad(self.weight, (8, 8, 8, 8), "constant", 1).to(self.device)
        self.weight *= 100

        data_norm = config['data_norm']
        t = data_norm['inp']
        self.inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(self.device)
        self.inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(self.device)

        self.criterion = nn.CosineSimilarity(dim=1)

        if args.DDP and config['degrade']['name'] == 'simsiam':
            optim_params = [{'params': self.model.module.encoder.parameters(), 'fix_lr': False},
                            {'params': self.model.module.predictor.parameters(), 'fix_lr': True}]
        elif config['degrade']['name'] == 'simsiam':
            optim_params = [{'params': self.model.encoder.parameters(), 'fix_lr': False},
                            {'params': self.model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = [{'params': self.model.parameters()}]
        self.optimizer = utils.make_optimizer(optim_params, config['optimizer'])

        # else
        self.args = args
        self.config = config
        self.data_norm = config['data_norm']
        self.epoch_save = config.get('epoch_save')
        self.min_loss, self.cl = 1e5, 1
        self.bs = config['total_batch_size']
        self.degrade = SRMDPreprocessing()

    def preprocess(self, batch):
        scale = batch['scale'][0].item()
        inp = torch.cat((batch['query'].unsqueeze(1), batch['key'].unsqueeze(1)), dim=1)  # b, n, c, h, w
        lr = self.degrade(inp, scale, norm=True)  # bn, c, h, w
        q_degrade, k_degrade = torch.split(lr, [1, 1], dim=1)
        q_degrade = q_degrade.squeeze(1)
        k_degrade = k_degrade.squeeze(1)

        q_degrade = (q_degrade - self.inp_sub) / self.inp_div
        k_degrade = (k_degrade - self.inp_sub) / self.inp_div



        return q_degrade, k_degrade

    def train(self, epoch, timer):
        # for DDP, epoch will go random
        if self.args.DDP:
            self.sampler1.sampler.set_epoch(epoch)

        # initial
        if not self.args.DDP or self.local_rank == 0:
            t_epoch_start = timer.t()
        losses = utils.Averager()

        # train model
        self.model.train()

        for batch1 in tqdm(self.loader1, leave=False, desc='train'):
            # 0-1
            for k1, v1 in batch1.items():
                batch1[k1] = v1.to(self.device)
            # online preprocess
            q, k = self.preprocess(batch1)

            if self.args.queue:
                p = {'query': q, 'key': k}
                pq, pk = self.pool(p)
            if epoch > 20:
                q = pq
                k = pk


            H = torch.cat((q, k), dim=0)

            self.optimizer.zero_grad()

            p1, p2, z1, z2, fea = self.model(x1=H[:self.bs, ...], x2=H[self.bs:, ...])
            loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

            losses.add(loss.item())
            loss.backward()
            self.optimizer.step()

        if not self.args.DDP or self.local_rank == 0:
            # timer stop
            t = timer.t()
            prog = (epoch - 1 + 1) / (self.config['epoch_max'] - 1 + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

            # write log
            res = 'epoch {}/{}, loss:{:.4f},  {} {}/{}\n' \
                .format(epoch, self.config['epoch_max'], losses.item(), t_epoch, t_elapsed, t_all)
            tqdm.write(res)

            # save file
            model = {'degrade': self.model}
            utils.save(model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP, self.args.savepath,
                       'last')
            if (self.epoch_save is not None) and (epoch % self.epoch_save == 0):
                utils.save(model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP, self.args.savepath,
                           'hun')
            if losses.item() < self.min_loss:
                self.min_loss = losses.item()
                utils.save(model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP, self.args.savepath,
                           'best')


# python train_degrade.py --tag 2 --queue
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path', default='configs/degradation.yaml')
    parser.add_argument('--savedir', help='your path to save model directory', default="checkpoints/degrade")
    parser.add_argument('--savepath', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--DDP', action='store_true')
    parser.add_argument('--queue', action='store_true')
    args = parser.parse_args()
    args.queue = True

    #### read config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        save_name = args.savedir
        if save_name is None:
            save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
        if args.tag is not None:
            args.savepath = os.path.join(save_name, args.tag)
        if os.path.exists(args.savepath) is False:
            os.makedirs(args.savepath)
            print('{} succeed'.format(args.savepath))
    torch.manual_seed(config['seed'])

    #### log file
    with open(os.path.join(args.savepath, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    t = Trainer(config, args)
    # train epochs
    timer = utils.Timer()
    for epoch in range(1, config['epoch_max'] + 1):
        t.train(epoch, timer)

import os

import argparse
import yaml
from tqdm import tqdm
import random
import torch
import torch.nn as nn

import utils
import datasets
from datasets.degrade import SRMDPreprocessing
import models
from show_utils import show_imgs, norm

class Trainer():
    def __init__(self, config, args):

        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.loader1 = datasets.make_data_loaders(config, DDP=False, rank=0, ngpu=1, state='degrade')

        self.model = models.make(config['degrade']).to(self.device)
        print('model: #params={}'.format(utils.compute_num_params(self.model, text=True)))


        self.L1 = nn.L1Loss()


        self.criterion = nn.CosineSimilarity(dim=1)

        if config['degrade']['name'] == 'simsiam':
            optim_params = [{'params': self.model.encoder.parameters(), 'fix_lr': False},
                            {'params': self.model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = [{'params': self.model.parameters()}]
        self.optimizer = utils.make_optimizer(optim_params, config['optimizer'])


        self.args = args
        self.config = config
        self.epoch_save = config.get('epoch_save')
        self.min_loss = 1e5
        self.bs = config['batch_size']
        self.degrade = SRMDPreprocessing()

    def preprocess(self, batch,  scale):

        inp = torch.cat((batch['query'].unsqueeze(1), batch['key'].unsqueeze(1)), dim=1)
        lr = self.degrade(inp, scale, norm=True)
        q_degrade, k_degrade = torch.split(lr, [1, 1], dim=1)
        q_degrade = q_degrade.squeeze(1)
        k_degrade = k_degrade.squeeze(1)

        # [0, 1] to [-1, 1]
        q_degrade = (q_degrade - 0.5) / 0.5
        k_degrade = (k_degrade - 0.5) / 0.5
        return q_degrade, k_degrade

    def train(self, epoch, timer):

        t_epoch_start = timer.t()
        losses = utils.Averager()

        self.model.train()

        for batch1 in tqdm(self.loader1, leave=False, desc=f'Epoch {epoch} Training'):
            for k1, v1 in batch1.items():
                batch1[k1] = v1.to(self.device)


            if self.config['fixed_downsample_scale']:
                scale = self.config['downsample_scale']
            else:
                scale = random.randint(2, 8)
                print('scale:', scale)
            q, k = self.preprocess(batch1, scale)

            # show_imgs(torch.cat([norm(q.cpu()),norm(k.cpu())], dim=0))

            H = torch.cat((q, k), dim=0)

            self.optimizer.zero_grad()

            p1, p2, z1, z2, fea = self.model(x1=H[:self.bs, ...], x2=H[self.bs:, ...])
            loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

            losses.add(loss.item())
            loss.backward()
            self.optimizer.step()

        t = timer.t()
        prog = epoch / self.config['epoch_max']
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        res = 'epoch {}/{}, loss:{:.4f}, {} {}/{}' \
            .format(epoch, self.config['epoch_max'], losses.item(), t_epoch, t_elapsed, t_all)
        tqdm.write(res)

        model_dict = {'degrade': self.model}
        utils.save(model_dict, self.optimizer, self.config['optimizer'], epoch, DDP=False,
                   savepath=self.args.savedir, state='last')
        if (self.epoch_save is not None) and (epoch % self.epoch_save == 0):
            utils.save(model_dict, self.optimizer, self.config['optimizer'], epoch, DDP=False,
                       savepath=self.args.savedir, state=f'epoch_{epoch}')
        if losses.item() < self.min_loss:
            self.min_loss = losses.item()
            utils.save(model_dict, self.optimizer, self.config['optimizer'], epoch, DDP=False,
                       savepath=self.args.savedir, state='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path', default='configs/degradation.yaml')
    parser.add_argument('--savedir', help='your path to save model directory', default="checkpoints/degrade")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--debug', action='store_true')
    # 移除了 --DDP 和 --queue 参数
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        save_name = args.savedir
        if args.tag is not None:
            args.savedir = os.path.join(save_name, args.tag)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
            print(f'Created save directory: {args.savedir}')

    torch.manual_seed(config['seed'])

    with open(os.path.join(args.savedir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # 初始化并开始训练
    trainer = Trainer(config, args)
    timer = utils.Timer()
    for epoch in range(1, config['epoch_max'] + 1):
        trainer.train(epoch, timer)
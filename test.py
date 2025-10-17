import argparse
import os.path

import yaml
from PIL import Image
from tqdm import tqdm
import numpy as np
import datasets
import models
import utils
from models.controller import *
import torch
from skimage import io
from show_utils import show_imgs, visualize_orthogonal_views


def batched_predict(model, inp, coord, cell, bsize):
	n = coord.shape[1]
	ql = 0
	preds = []

	feature = model.encoder(inp) # fix
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
	parser.add_argument('--model_weight', help='your path to model_weight.pth', default='checkpoints/sr/EPFL/epoch-best.pth')
	parser.add_argument('--save', help='your path to save reconstructed image', default='output')
	parser.add_argument('--test_config', help='test config file path', default='configs/test_yz.yaml')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	with open(args.test_config, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	direct = config['test_dataset']['dataset']['args']['direction']
	scale = config['test_dataset']['dataset']['args']['scale']

	save_dir = os.path.join(args.save, direct)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	loader = datasets.make_data_loaders(config, DDP=False, state='test')

	if direct not in ['xz', 'yz']:
		raise ValueError('direction should be xz or yz')


	model = models.make(config['model'], args={'config': config}, load_sd=args.model_weight).to(device)


	bs = config['batch_size']
	length = len(loader) * bs

	volume = None
	idx = 0

	for batch in tqdm(loader, leave=False, desc='test'):
		inp, coord, cell, shape = batch['inp'], batch['coord'], batch['cell'], batch['shape']
		inp, coord, cell = inp.to(device), coord.to(device), cell.to(device)

		# show_imgs(inp.cpu())
		inp = (inp - 0.5) / 0.5
		h = shape[0][0]
		w = shape[1][0]
		model.eval()
		with torch.no_grad():
			pred = batched_predict(model, inp, coord, cell, bsize=30000)

		pred = pred * 0.5 + 0.5
		pred = pred.detach().view(-1, h, w).cpu()  # [batch, h, w]
		show_imgs(pred.unsqueeze(1))
		pred = pred.numpy()
		for i in range(pred.shape[0]):
			img_array = np.clip(pred[i] * 255, 0, 255).astype(np.uint8)
			io.imsave(f'{args.save}/{direct}/{idx}.png', img_array)

			if volume is None:
				if direct == 'yz':
					volume = np.zeros((h, w, length), dtype=np.uint8)
				else:
					volume = np.zeros((h, length, w), dtype=np.uint8)

			if direct == 'yz':
				volume[:, :, idx] = img_array.reshape(h, w)
			else:
				volume[:, idx, :] = img_array.reshape(h, w)

			idx += 1
	if direct == 'yz':
		volume = volume[:, :, :idx]
	else:
		volume = volume[:, :idx, :]
	io.imsave(f'{args.save}/{direct}.tif', volume)
	visualize_orthogonal_views(volume)











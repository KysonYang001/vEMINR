import torch
import torch.nn as nn

@torch.no_grad()
class dequeue_and_enqueue(nn.Module):
    """It is the training pair pool for increasing the diversity in a batch.
    Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
    batch could not have different resize scaling factors. Therefore, we employ this training pair pool
    to increase the degradation diversity in a batch.
    """
    def __init__(self, config, state):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.b, c, h, w = config['total_batch_size'], 1, config['inp_size'], config['inp_size']*config['train_dataset1']['wrapper']['args']['scale']
        self.queue_size = config['queue_size']
        self.state = state
        
        # 添加SDF归一化参数
        self.sdf_norm = None
        if config.get('sdf_norm'):
            sdf_norm = config['sdf_norm']
            self.sdf_sub = torch.FloatTensor(sdf_norm['sub']).to(self.device)
            self.sdf_div = torch.FloatTensor(sdf_norm['div']).to(self.device)
            self.sdf_norm = True

        # initialize
        if state == 'degrade':
            assert self.queue_size % self.b == 0, f'queue size {self.queue_size} should be divisible by batch size {self.b}'
            self.queue_q = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_k = torch.zeros(self.queue_size, c, h, w).to(self.device)
            # self.queue_ker = torch.zeros(self.queue_size, 1, k, k).to(self.device)
            self.queue_ptr = 0
        elif state =='SR':
            assert self.queue_size % self.b == 0, f'queue size {self.queue_size} should be divisible by batch size {self.b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_gt = torch.zeros(self.queue_size, config['sample_q'], c).to(self.device)
            self.queue_cell = torch.zeros(self.queue_size, config['sample_q'], 2).to(self.device)
            self.queue_coord = torch.zeros(self.queue_size, config['sample_q'], 2).to(self.device)
            self.queue_scale = torch.zeros(self.queue_size).to(self.device)
            self.queue_sdf = torch.zeros(self.queue_size, config['sample_q'], c).to(self.device)
            self.queue_ptr = 0
    
    def normalize_sdf(self, sdf_tensor):
        """对SDF张量应用归一化: (x - mean) / std"""
        if self.sdf_norm and sdf_tensor is not None:
            return (sdf_tensor - self.sdf_sub) / self.sdf_div
        return sdf_tensor
        
    def forward(self, inp):
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            if self.state == 'degrade':
                self.queue_q = self.queue_q[idx]
                self.queue_k = self.queue_k[idx]
                # get first b samples
                q_dequeue = self.queue_q[0:self.b, :, :, :].clone()
                k_dequeue = self.queue_k[0:self.b, :, :, :].clone()
                # update the queue
                self.queue_q[0:self.b, :, :, :] = inp['query'].clone()
                self.queue_k[0:self.b, :, :, :] = inp['key'].clone()
    
                return q_dequeue, k_dequeue


            
            elif self.state == 'SR':
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_cell = self.queue_cell[idx]
                self.queue_coord = self.queue_coord[idx]
                self.queue_scale = self.queue_scale[idx]
                self.queue_sdf = self.queue_sdf[idx]
                # get first b samples
                lr_dequeue = self.queue_lr[0:self.b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:self.b, :, :].clone()
                cell_dequeue = self.queue_cell[0:self.b, :, :].clone()
                coord_dequeue = self.queue_coord[0:self.b, :, :].clone()
                scale_dequeue = self.queue_scale[0:self.b].clone()
                # SDF数据已经在队列中归一化了，直接取出即可
                sdf_dequeue = self.queue_sdf[0:self.b, :, :].clone()
                
                # update the queue
                self.queue_lr[0:self.b, :, :, :] = inp['lr'].clone()
                self.queue_gt[0:self.b, :, :] = inp['gt'].clone()
                self.queue_cell[0:self.b, :, :] = inp['cell'].clone()
                self.queue_coord[0:self.b, :, :] = inp['coord'].clone()
                self.queue_scale[0:self.b] = inp['scale'].clone()
                # 对SDF应用归一化
                gt_sdf_normalized = self.normalize_sdf(inp.get('gt_sdf'))
                if gt_sdf_normalized is not None:
                    self.queue_sdf[0:self.b, :, :] = gt_sdf_normalized.clone()
                else:
                    # 如果没有SDF数据，设置为零
                    self.queue_sdf[0:self.b, :, :] = torch.zeros_like(self.queue_sdf[0:self.b, :, :])

                return lr_dequeue, gt_dequeue, cell_dequeue, coord_dequeue, scale_dequeue.unsqueeze(-1), sdf_dequeue

        else:
            # pool isn't full
            if self.state == 'degrade':
                self.queue_q[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['query'].clone()
                self.queue_k[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['key'].clone()
                self.queue_ptr = self.queue_ptr + self.b
                return inp['query'], inp['key']

            
            elif self.state == 'SR':
                self.queue_lr[self.queue_ptr:self.queue_ptr + self.b, :, :, :] = inp['lr'].clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + self.b, :, :] = inp['gt'].clone()
                self.queue_cell[self.queue_ptr:self.queue_ptr + self.b, :, :] = inp['cell'].clone()
                self.queue_coord[self.queue_ptr:self.queue_ptr + self.b, :, :] = inp['coord'].clone()
                self.queue_scale[self.queue_ptr:self.queue_ptr + self.b] = inp['scale'].clone()
                # 对SDF应用归一化
                gt_sdf_normalized = self.normalize_sdf(inp.get('gt_sdf'))
                if gt_sdf_normalized is not None:
                    self.queue_sdf[self.queue_ptr:self.queue_ptr + self.b, :, :] = gt_sdf_normalized.clone()
                else:
                    # 如果没有SDF数据，设置为零
                    self.queue_sdf[self.queue_ptr:self.queue_ptr + self.b, :, :] = torch.zeros_like(self.queue_sdf[self.queue_ptr:self.queue_ptr + self.b, :, :])
                self.queue_ptr = self.queue_ptr + self.b
                return inp['lr'], inp['gt'], inp['cell'], inp['coord'], inp['scale'].unsqueeze(-1), gt_sdf_normalized
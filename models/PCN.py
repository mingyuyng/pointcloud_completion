import numpy as np
import torch
import os
from torch.autograd import Variable
from models.networks import *
import scipy.io as sio
import random
import pytorch3d
from models.extensions.chamfer_dist import ChamferDistance
from torch.optim import lr_scheduler
from models.extensions.emd_dist.emd import earth_mover_distance
from config import cfg

class PCN(torch.nn.Module):

    def __init__(self):

        super(PCN, self).__init__()

        # self.opt = opt
        self.device = torch.device('cuda:{}'.format(cfg.TRAIN.GPU_IDS[0])) if cfg.TRAIN.GPU_IDS else torch.device('cpu')

        self.encoder = define_E(len_global=cfg.TRAIN.LEN_GLOBAL, init_type='kaiming', gpu_ids=cfg.TRAIN.GPU_IDS)
        self.decoder = define_D(num_coarse=cfg.TRAIN.NUM_COARSE, grid_size=cfg.TRAIN.GRID_SIZE, grid_scale=cfg.TRAIN.GRID_SCALE, 
                                 len_global=cfg.TRAIN.LEN_GLOBAL, init_type='kaiming', gpu_ids=cfg.TRAIN.GPU_IDS)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.LEARNING_RATE, betas=(cfg.TRAIN.BETA1, 0.999))
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_ITERS, gamma=cfg.TRAIN.GAMMA)
        self.chamfer_dist = ChamferDistance().to(self.device)

    def name(self):
        return 'PCN_Model'

    def set_input(self, pc, pc_true_coarse, pc_true_fine, labels):
        self.input_pc = pc.clone().to(self.device)
        self.target_pc_coarse = pc_true_coarse.clone().to(self.device)
        self.target_pc_fine = pc_true_fine.clone().to(self.device)

    def forward(self):
        self.coarse, self.fine = self.decoder(self.encoder(self.input_pc))

    def backward(self):
        
        self.loss_coarse = cfg.TRAIN.LAM_COARSE*(self.chamfer_dist(self.coarse, self.target_pc_coarse))
        self.loss_fine = cfg.TRAIN.LAM_FINE*self.chamfer_dist(self.fine, self.target_pc_fine)

        self.loss = self.loss_coarse + self.loss_fine
        self.loss.backward()

    def optimize_parameters(self):

        self.forward()                  
        self.optimizer.zero_grad()        
        self.backward()                   
        self.optimizer.step()             

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def save_network(self, path, name):
        torch.save(self.encoder.state_dict(), os.path.join(path, name + '_E.w'))
        torch.save(self.decoder.state_dict(), os.path.join(path, name + '_D.w'))

    def load_network(self, path, name):
        self.encoder.load_state_dict(torch.load(os.path.join(path, name + '_E.w')))
        self.decoder.load_state_dict(torch.load(os.path.join(path, name + '_D.w')))

    def print_loss_message(self):
        message = f'Chamfer (coarse): {self.loss_coarse.item():.3f}, Chamfer (fine): {self.loss_fine.item():.3f}'
        return message
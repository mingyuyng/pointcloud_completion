import numpy as np
import torch
import os
from torch.autograd import Variable
from models.networks import *
import scipy.io as sio
import random
import pytorch3d
from chamferdist import ChamferDistance
from torch.optim import lr_scheduler
from models.extensions.emd_dist.emd import earth_mover_distance
from config import cfg

class Trans(torch.nn.Module):

    def __init__(self):

        super(Trans, self).__init__()

        # self.opt = opt
        self.device = torch.device('cuda:{}'.format(cfg.TRAIN.GPU_IDS[0])) if cfg.TRAIN.GPU_IDS else torch.device('cpu')

        self.encoder = define_E_trans(len_global=cfg.TRAIN.LEN_GLOBAL, init_type='xavier', gpu_ids=cfg.TRAIN.GPU_IDS)
        self.decoder = define_D(num_coarse=cfg.TRAIN.NUM_COARSE, grid_size=cfg.TRAIN.GRID_SIZE, grid_scale=cfg.TRAIN.GRID_SCALE, 
                                len_global=cfg.TRAIN.LEN_GLOBAL, init_type='kaiming', gpu_ids=cfg.TRAIN.GPU_IDS)
        self.classifier = define_C(len_global=cfg.TRAIN.LEN_GLOBAL, init_type='kaiming', gpu_ids=cfg.TRAIN.GPU_IDS)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.LEARNING_RATE, betas=(cfg.TRAIN.BETA1, 0.999))
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_ITERS, gamma=cfg.TRAIN.GAMMA)

        self.chamfer_dist = ChamferDistance()
        self.loss_c = nn.CrossEntropyLoss()

    def name(self):
        return 'Trans_Model'

    def set_input(self, pc, pc_true_coarse, pc_true_fine, gt_labels):
        self.input_pc = pc.clone().to(self.device)
        self.target_pc_coarse = pc_true_coarse.clone().to(self.device)
        self.target_pc_fine = pc_true_fine.clone().to(self.device)
        if gt_labels is not None:
            self.gt_labels = gt_labels.clone().to(self.device)

    def forward(self):
        global_feat = self.encoder(self.input_pc)
        coarse, fine = self.decoder(global_feat)

        pred = self.classifier(global_feat)
        return coarse, fine, pred
        

    def backward(self, coarse, fine, pred):
         
        self.loss_coarse = cfg.TRAIN.LAM_COARSE*(self.chamfer_dist(coarse, self.target_pc_coarse, bidirectional=True))/512
        self.loss_fine = cfg.TRAIN.LAM_FINE*self.chamfer_dist(fine, self.target_pc_fine, bidirectional=True)/2048
        self.loss_cla = cfg.TRAIN.LAM_CLA*self.loss_c(pred, self.gt_labels)

        self.loss = self.loss_coarse + self.loss_fine + self.loss_cla
        self.loss.backward()

    def optimize_parameters(self):

        coarse, fine, pred = self.forward()                  
        self.optimizer.zero_grad()        
        self.backward(coarse, fine, pred)                   
        self.optimizer.step()             

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()

    def save_network(self, path, name):
        torch.save(self.encoder.state_dict(), os.path.join(path, name + '_E.w'))
        torch.save(self.decoder.state_dict(), os.path.join(path, name + '_D.w'))

    def load_network(self, path, name):
        self.encoder.load_state_dict(torch.load(os.path.join(path, name + '_E.w')))
        self.decoder.load_state_dict(torch.load(os.path.join(path, name + '_D.w')))

    def print_loss_message(self):
        message = f'Chamfer (coarse): {self.loss_coarse.item():.3f}, Chamfer (fine): {self.loss_fine.item():.3f}'
        message += f', Classification loss: {self.loss_cla.item():.3f}' if cfg.TRAIN.LAM_CLA != 0 else ''
        return message
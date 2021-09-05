import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

    
def define_E(len_global, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = pcn_encoder(len_global=len_global)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(num_coarse, grid_size, grid_scale, len_global, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = pcn_decoder(num_coarse=num_coarse, grid_size=grid_size, grid_scale=grid_scale, len_global=len_global)
    return init_net(net, init_type, init_gain, gpu_ids)


class mlp(torch.nn.Module):
    def __init__(self, layer_dims, layer_out, bn=None):
        
        super(mlp, self).__init__()

        model = []
        for i in range(len(layer_dims)-1):
            model += [nn.Linear(layer_dims[i], layer_dims[i+1]),
                      nn.ReLU()]
            if bn is not None:
                model += [nn.BatchNorm1d(layer_dims[i])]
        
        model += [nn.Linear(layer_dims[-1], layer_out)]        
        self.model = nn.Sequential(*model)

    def forward(self, inputs):
        return self.model(inputs)


class mlp_conv(torch.nn.Module):
    def __init__(self, layer_dims, layer_out, bn=None):
        
        super(mlp_conv, self).__init__()

        model = []
        for i in range(len(layer_dims)-1):
            model += [nn.Conv1d(layer_dims[i], layer_dims[i+1], 1),
                      nn.ReLU()]
            if bn is not None:
                model += [nn.BatchNorm1d(layer_dims[i+1])]
        
        model += [nn.Conv1d(layer_dims[-1], layer_out, 1)]        
        self.model = nn.Sequential(*model)

    def forward(self, inputs):

        return self.model(inputs)

# Encoder of PCN
class pcn_encoder(nn.Module):
    def __init__(self, len_global):
        
        super(pcn_encoder, self).__init__()

        self.mlp1 = mlp_conv([3, 128], 256)
        self.mlp2 = mlp_conv([512, 512], len_global)

    def forward(self, inputs):

        # input dimention: NxMx3
        inputs = torch.transpose(inputs, 1, 2)
        
        local_f = self.mlp1(inputs)                                # Nx3xM -> NxfxM
        global_f, _ = torch.max(local_f, -1, keepdim=True)         # NxfxM -> Nxfx1
        
        total_f = torch.cat((local_f, global_f.repeat(1,1,local_f.shape[-1])), 1)  # NxfxM -> Nx2fxM
        
        features = self.mlp2(total_f)
        global_f, _ = torch.max(features, -1)

        return global_f

# Decoder of PCN
class pcn_decoder(nn.Module):
    def __init__(self, num_coarse, grid_size, grid_scale, len_global):
        
        super(pcn_decoder, self).__init__()
        
        self.coarse = num_coarse
        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.num_fine = self.coarse * self.grid_size**2
        self.mlp = mlp([len_global, len_global, len_global], self.coarse*3)
        self.mlp_conv = mlp_conv([len_global+5, len_global, len_global], 3)

    def forward(self, inputs):

        # input dimention: Nx1024
        # Coarse estimation        
        coarse = self.mlp(inputs)
        coarse = coarse.view(-1, self.coarse, 3)    # N x c x 3
        
        # Folding
        with torch.no_grad():
            axis = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)   
            grid_x, grid_y = torch.meshgrid(axis, axis)                                          
            grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), -1)         # d x d x 2
            grid = grid.view(-1, 2).unsqueeze(0).unsqueeze(0)                          # 1 x 1 x d^2 x 2
            grid_feat = grid.repeat(inputs.shape[0], self.coarse, 1, 1)                # N x c x d^2 x 2
            grid_feat = grid_feat.view(-1, self.num_fine, 2).to(coarse.device)         # N x c*d^2 x 2

        point_feat = coarse.unsqueeze(2).repeat(1, 1, self.grid_size**2, 1)            # N x c x d^2 x 3 
        point_feat = point_feat.view(-1, self.num_fine, 3)                             # N x c*d^2 x 3
        
        global_feat = inputs.unsqueeze(1).repeat(1, self.num_fine, 1)                  # N x c*d^2 x 1024
        
        feat = torch.cat((grid_feat, point_feat, global_feat), 2)

        center = coarse.unsqueeze(2).repeat(1, 1, self.grid_size**2, 1)
        center = center.view(-1, self.num_fine, 3)
        
        fine = self.mlp_conv(feat.transpose(1, 2)).transpose(1, 2) + center

        return coarse, fine

# Self-attention layer
class SA_Layer(nn.Module):
    def __init__(self, channels, dropout=0):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xyz):
        x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, self.dropout(attention)) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x-x_r)))
        x = x + self.dropout(x_r)
        return x

class trans_encoder(nn.Module):
    def __init__(self, len_global):
        
        super(trans_encoder, self).__init__()

        self.mlp1 = mlp_conv([3, 128], 128)
        self.mlp2 = mlp_conv([256, 128], 128)
        self.mlp3 = mlp_conv([384, 512], 1024)

        self.sa1  = SA_Layer(128, 0)
        self.sa2  = SA_Layer(128, 0)
        self.sa3  = SA_Layer(128, 0)

    def forward(self, inputs):

        # input dimention: NxMx3
        inputs = torch.transpose(inputs, 1, 2)

        # embedding for each point
        local_f = self.mlp1(inputs)                               # Nx3xM -> NxfxM
        global_f, _ = torch.max(local_f, -1, keepdim=True)        # NxfxM -> Nxfx1

        total_f = torch.cat((local_f, global_f.repeat(1,1,local_f.shape[-1])), 1)  # NxfxM -> Nx2fxM
        features = self.mlp2(total_f)                             # Nx2fxM -> NxfxM

        # Point Transformer
        local_1 = self.sa1(features, local_f)
        local_2 = self.sa2(local_1, local_f)
        local_3 = self.sa3(local_2, local_f)

        # Concatenation
        local_cat  = torch.cat((local_1, local_2, local_3), 1)   # NxfxM -> Nx4fxM
        local_fuse = self.mlp3(local_cat)
        global_f, _ = torch.max(local_fuse, -1)                  # NxfxM -> Nxfx1
        return global_f


def define_E_trans(len_global, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = trans_encoder(len_global=len_global)
    return init_net(net, init_type, init_gain, gpu_ids)


class pcn_classifier(nn.Module):
    def __init__(self, len_global):
        
        super(pcn_classifier, self).__init__()

        self.mlp1 = nn.Linear(len_global, 512)
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, 8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):

        x = self.dropout(self.mlp1(inputs))
        x = self.relu(x)
        x = self.dropout(self.mlp2(x))
        x = self.relu(x)
        pred = self.mlp3(x)

        return pred

def define_C(len_global, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = pcn_classifier(len_global=len_global)
    return init_net(net, init_type, init_gain, gpu_ids)


if __name__ == '__main__':

    points = torch.randn((100, 200, 3))
    encoder = pcn_encoder()
    decoder = pcn_decoder(1024, 4, 0.05)

    features =encoder(points)
    coarse, fine = decoder(features)

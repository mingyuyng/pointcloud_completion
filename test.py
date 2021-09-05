import numpy as np
import torch
import os
import pytorch3d
from easydict import EasyDict
from chamferdist import ChamferDistance
import torch
import data_utils.data_loader
from config import cfg
import time
import open3d as o3d

torch.manual_seed(0)

''' To do list
Algoritihm:
    discriminator loss or multi-view loss
    add transformer layers
'''

# Create the dataloader 
test_dataset_loader = data_utils.data_loader.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(data_utils.data_loader.DatasetSubset.VAL),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=data_utils.data_loader.collate_fn,
                                                pin_memory=True,
                                                shuffle=False)

train_dataset_loader = data_utils.data_loader.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(data_utils.data_loader.DatasetSubset.TRAIN),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=data_utils.data_loader.collate_fn,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True)

print(f'Dataloader built!! Number: {len(val_data_loader)}')

if cfg.MODEL == 'PCN_att':
    from models.Trans import Trans
    net = Trans()
elif cfg.MODEL == 'PCN':
    from models.PCN import PCN
    net = PCN()

print(cfg.TRAIN.SAVE_PATH)
net.load_network(cfg.TRAIN.SAVE_PATH, 'latest')
net.eval()

if not os.path.exists(cfg.TEST.SAVE_PATH):
    os.mkdir(cfg.TEST.SAVE_PATH)

chamfer_dist = ChamferDistance()

coarse_chamfer_dist = []
fine_chamfer_dist = []

for batch_i, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
    
    net.set_input(data['partial_cloud'], data['gtcloud'][:cfg.TEST.NUM_COARSE], data['gtcloud'], None)
    coarse, fine, pred = net.forward()
  
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coarse[0].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(cfg.TEST.SAVE_PATH, str(batch_i)+'_coarse_est.pcd'), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fine[0].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(cfg.TEST.SAVE_PATH, str(batch_i)+'_fine_est.pcd'), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['partial_cloud'][0].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(cfg.TEST.SAVE_PATH, str(batch_i)+'_partial.pcd'), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['gtcloud'][0][:cfg.TEST.NUM_COARSE].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(cfg.TEST.SAVE_PATH, str(batch_i)+'_coarse_gt.pcd'), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['gtcloud'][0].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(cfg.TEST.SAVE_PATH, str(batch_i)+'_fine_gt.pcd'), pcd)

    coarse_chamfer = 10000*chamfer_dist(coarse.cuda(), data['gtcloud'][:cfg.TEST.NUM_COARSE].cuda(), bidirectional=True).item()/512
    fine_chamfer = 10000*chamfer_dist(fine.cuda(), data['gtcloud'].cuda(), bidirectional=True).item()/2048

    coarse_chamfer_dist.append(coarse_chamfer)
    fine_chamfer_dist.append(fine_chamfer)

    print(f'Coarse chamfer: {coarse_chamfer:.3f}, Fine chamfer: {fine_chamfer:.3f}')
    print('Processed ' + str(batch_i))

coarse_chamfer_np = np.asarray(coarse_chamfer_dist).reshape((8, 100)).mean(axis=-1)
fine_chamfer_np = np.asarray(fine_chamfer_dist).reshape((8, 100)).mean(axis=-1)

print(coarse_chamfer_np)
print(fine_chamfer_np)

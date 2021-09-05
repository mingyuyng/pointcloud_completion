import numpy as np
import torch
import os
import pytorch3d
import data_utils.data_loader
from config import cfg
import time
import shutil


torch.manual_seed(0)

# Mapping for each category id
dic = {
    '03001627': 0,
    '02958343': 1,
    '02691156': 2,
    '03636649': 3,
    '04256520': 4,
    '04379243': 5,
    '04530566': 6,
    '02933112': 7
}

# Create the dataloader 
train_dataset_loader = data_utils.data_loader.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
test_dataset_loader = data_utils.data_loader.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(data_utils.data_loader.DatasetSubset.TRAIN),
                                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=data_utils.data_loader.collate_fn,
                                                pin_memory=True,
                                                shuffle=True,
                                                drop_last=True)
val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(data_utils.data_loader.DatasetSubset.VAL),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=data_utils.data_loader.collate_fn,
                                                pin_memory=True,
                                                shuffle=False)

print(f'Dataloader built!! Number: {len(train_data_loader)}')


# Choose the method ["PCN_att", "PCN"]
if cfg.MODEL == 'PCN_att':
    from models.Trans import Trans
    net = Trans()
elif cfg.MODEL == 'PCN':
    from models.PCN import PCN
    net = PCN()


if os.path.exists(cfg.TRAIN.LOG_PATH):
	os.remove(cfg.TRAIN.LOG_PATH)

if not os.path.exists(cfg.TRAIN.SAVE_PATH):
    os.mkdir(cfg.TRAIN.SAVE_PATH)
else:
    shutil.rmtree(cfg.TRAIN.SAVE_PATH)
    os.mkdir(cfg.TRAIN.SAVE_PATH)

count = 0
for epoch in range(cfg.TRAIN.N_EPOCHS):
    epoch_start = time.time()
    for batch_i, (taxonomy_id, model_id, data) in enumerate(train_data_loader):
        
        labels = torch.tensor([dic[x] for x in taxonomy_id])
        net.set_input(data['partial_cloud'], data['gtcloud'][:cfg.TRAIN.NUM_COARSE], data['gtcloud'], labels)
        net.optimize_parameters()
        end = time.time()
        if count % cfg.TRAIN.PRINT_ITER == 0:
            with open(cfg.TRAIN.LOG_PATH, "a") as log_file:
                message = f'Epoch: {epoch}, Iter: {batch_i}, time: {end-epoch_start:.3f}, {net.print_loss_message()}'
                print(message)
                log_file.write('%s\n' % message)
        count += 1

    if epoch % cfg.TRAIN.SAVE_FREQ == 0:
        net.save_network(cfg.TRAIN.SAVE_PATH, 'epoch_' + str(epoch))
        end = time.time()
        message = f'Saved network at epoch {epoch}, time: {end-epoch_start}'
        print(message)
        with open(cfg.TRAIN.LOG_PATH, "a") as log_file:
            log_file.write('%s\n' % message)
        net.save_network(cfg.TRAIN.SAVE_PATH, 'latest')

    net.update_learning_rate()

net.save_network(cfg.TRAIN.SAVE_PATH, 'latest')
with open(cfg.TRAIN.LOG_PATH, "a") as log_file:
    log_file.write('%s\n' % 'Finished!!!')


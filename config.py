from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = '/home/mingyuy/pointcloud_completion/dataset/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/home/mingyuy/pointcloud_completion/dataset/shapenet/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/home/mingyuy/pointcloud_completion/dataset/shapenet/%s/gt/%s/%s.h5'

#
# Dataset Selection
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTI
__C.DATASET.TRAIN_DATASET                        = 'Completion3D'
__C.DATASET.TEST_DATASET                         = 'Completion3D'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048

__C.MODEL                                        = 'PCN_att'   # 'PCN', 'PCN_att'

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.EXPERIMENT_NAME                        = 'experiment1'
__C.TRAIN.BATCH_SIZE                             = 32
__C.TRAIN.N_EPOCHS                               = 200
__C.TRAIN.SAVE_FREQ                              = 5
__C.TRAIN.LEARNING_RATE                          = 5e-4
__C.TRAIN.LR_DECAY_ITERS                         = 40
__C.TRAIN.BETA1                                  = 0.9
__C.TRAIN.BETA2                                  = 0.999
__C.TRAIN.LEN_GLOBAL                             = 1024
__C.TRAIN.NUM_COARSE                             = 512
__C.TRAIN.GPU_IDS                                = [0]
__C.TRAIN.LAM_COARSE                             = 5000
__C.TRAIN.LAM_FINE                               = 20000
__C.TRAIN.LAM_CLA                                = 1
__C.TRAIN.GAMMA                                  = .3
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.PRINT_ITER                             = 25
__C.TRAIN.GRID_SIZE                              = 2
__C.TRAIN.GRID_SCALE                             = 0.05
__C.TRAIN.SAVE_PATH                              = 'network/'+__C.TRAIN.EXPERIMENT_NAME 
__C.TRAIN.LOG_PATH                               = 'logs/'+__C.TRAIN.EXPERIMENT_NAME+'.txt'

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.NUM_COARSE                              = 512
__C.TEST.SAVE_PATH                               = 'results/'+__C.TRAIN.EXPERIMENT_NAME



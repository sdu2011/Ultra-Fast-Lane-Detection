# DATA
# dataset='Tusimple'
dataset='Autocore'
data_root = '/home/train/hdd/sc/data/lane/autocore'

# TRAIN
epoch = 500
batch_size = 8
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
# gamma = 0 #
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '34'
griding_num = 100
# griding_num = 200
use_aux = False

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/train/hdd/sc/work/logs'

# FINETUNE or RESUME MODEL PATH
finetune = None
# finetune = '/home/train/hdd/sc/work/logs/20210611_175335_lr_4e-04_b_16/ep097.pth'
# finetune = '/home/train/hdd/sc/work/logs/20210617_103517_lr_4e-04_b_12/ep079.pth'
# finetune = '/home/train/hdd/sc/work/logs/20210617_164050_lr_4e-04_b_8/ep038.pth'
# finetune = '/home/train/hdd/sc/work/logs/20210618_103634_lr_4e-04_b_8/ep030.pth'

resume = None

# TEST
# test_model = '/home/train/hdd/sc/work/logs/20210611_181305_lr_4e-04_b_16/ep499.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210616_150204_lr_4e-04_b_8/ep102.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210616_155137_lr_4e-04_b_8/ep499.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210616_182559_lr_4e-04_b_12/ep499.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210617_104415_lr_4e-04_b_12/ep333.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210617_114804_lr_4e-04_b_12/ep499.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210617_154004_lr_4e-04_b_12/ep119.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210617_155303_lr_4e-04_b_12/ep067.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210617_164050_lr_4e-04_b_8/ep038.pth'
# test_model = '/home/train/hdd/sc/work/logs/20210617_183715_lr_4e-04_b_8/ep013.pth'
test_model = '/home/train/hdd/sc/work/logs/20210618_103634_lr_4e-04_b_8/ep028.pth'

# test_work_dir = '/home/train/hdd/sc/data/lane/rosbag0610'
test_work_dir = '/home/train/hdd/sc/data/lane/autocore_test'

num_lanes = 4

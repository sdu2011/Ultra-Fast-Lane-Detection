# DATA
# dataset='Tusimple'
dataset='Autocore'
data_root = '/home/train/hdd/sc/data/lane/autocore'

# TRAIN
epoch = 500
batch_size = 16
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = False

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/train/hdd/sc/work/logs'

# FINETUNE or RESUME MODEL PATH
finetune = '/home/train/hdd/sc/work/logs/20210611_175335_lr_4e-04_b_16/ep097.pth'
resume = None

# TEST
test_model = '/home/train/hdd/sc/work/logs/20210611_181305_lr_4e-04_b_16/ep499.pth'
test_work_dir = '/home/train/hdd/sc/data/lane/rosbag0610'

num_lanes = 4

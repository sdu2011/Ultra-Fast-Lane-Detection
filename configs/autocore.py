# DATA
# dataset='Tusimple'
dataset='Autocore'
data_root = '/home/suchang/data/lane/autocore_0622'

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
# backbone = '34'
backbone = '34'
griding_num = 100
# griding_num = 200
use_aux = False

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/suchang/work/logs'

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
# test_model = '/home/train/hdd/sc/work/logs/20210618_103634_lr_4e-04_b_8/ep028.pth'
# test_model = '/home/suchang/work/logs/20210630_181641_lr_4e-04_b_8/ep037.pth'
# test_model = '/home/suchang/work/logs/20210630_183412_lr_4e-04_b_8/ep490.pth'
# test_model = '/home/suchang/work/logs/20210701_111648_lr_4e-04_b_8/ep050.pth'
# test_model = '/home/suchang/work/logs/20210701_145246_lr_4e-04_b_8/ep180.pth'
test_model = '/home/suchang/work/logs/20210701_173126_lr_4e-04_b_1/ep010.pth'

# test_work_dir = '/home/train/hdd/sc/data/lane/rosbag0610'
test_work_dir = '/home/suchang/data/lane/autocore_0622_test'

num_lanes = 4

import torch, os
from model.model import parsingNet
# from utils.common import merge_config
# from utils.dist_utils import dist_print
# from evaluation.eval_wrapper import eval_lane
import torch
import scipy.special
import numpy as np
import cv2
from utils.tool_func import get_linenumber

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
row_anchor = culane_row_anchor
# row_anchor = tusimple_row_anchor

model_type='culane'

if __name__ == "__main__":

    net = parsingNet(backbone='18',cls_dim=(201, 18, 4))
    test_model = './download/culane_18.pth'
    if model_type == 'tusimple':
        net = parsingNet(backbone='18',cls_dim=(100, 56, 4))
        test_model = './download/tusimple_18.pth'

    # input=torch.randn((1,3,288,800))

    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    imPath= './download/lishui_tl.png'
    img = cv2.imread(imPath)
    img_w,img_h = img.shape[1],img.shape[0]
    print('line{},img_w={},img_h={}'.format(get_linenumber(),img_w,img_h))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(800,288))
    img = img / 255.
    img = (img - image_mean) / image_std
    
    input = np.zeros([1, 3, 288, 800], dtype=np.float32)
    input[0,:,:,:] = img.transpose(2,0,1)
    input = torch.from_numpy(input)

    #加载cfg.test_model
    state_dict = torch.load(test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    out = net(input)
    print(out.shape) 

    # out_j = out[0].data.cpu().numpy()  
    # out_j = out_j[:, ::-1, :] #为啥在h这个维度要逆序?
    # prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) #在grid这一维度上做softmax
    # idx = np.arange(cfg.griding_num) + 1
    # idx = idx.reshape(-1, 1, 1)
    # loc = np.sum(prob * idx, axis=0) #这里为什么要做点乘?
    # out_j = np.argmax(out_j, axis=0) #在grid这一维度上求argmax
    # loc[out_j == cfg.griding_num] = 0
    # out_j = loc

    out_j = out[0].data.cpu().numpy() #[grid+1,anchor_lane_number,max_lane_num]
    out_j = out_j[:, ::-1, :] #h这个维度倒序?
    # print(''out_j.shape)
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) #grid这个维度只对前grid个值求概率 最后一个值用来表示是否存在车道线的点
    print('prob shape={}'.format(prob.shape))
    print(prob[5,7,1]) #第7个参考行的第五个grid是第1条车道线的概率

    griding_num=200
    idx = np.arange(griding_num) + 1
    # print('idx shape={}'.format(idx.shape))
    # print('idx={}'.format(idx))
    # print(type(idx))
    idx = idx.reshape(-1, 1, 1)
    # print('idx shape={}'.format(idx.shape))
    # print('idx={}'.format(idx))

    #这里作者没有直接用loc=argmax(prob,axis=0)求最大概率的位置. 原因如下:https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/99
    loc = np.sum(prob * idx, axis=0)   #loc为18 * 4矩阵,值为该位置的grid的期望.
    print('line{},out_j shape={}'.format(get_linenumber(),out_j.shape))
    out_j = np.argmax(out_j, axis=0) # 求出概率最大的下标
    print('line{},out_j shape={}'.format(get_linenumber(),out_j.shape))
    loc[out_j == griding_num] = 0 #如果概率最大的下标为gridding_num的话说明是grid+1中的那个1. 则把loc相应位置的值置为0.表示在这个位置无车道线点.?
    out_j = loc
    # print('out_j shape={}'.format(out_j.shape))
    # print('out_j={}'.format(out_j))

    col_sample = np.linspace(0, 800 - 1, 200) # 0-799均匀分出200个点
    col_sample_w = col_sample[1] - col_sample[0] # 每个grid的像素数目

    #out_j.shape=(18,4)
    vis = cv2.imread(imPath)
    for i in range(out_j.shape[1]):  #遍历每条车道线
        if np.sum(out_j[:, i] != 0) > 2: #对车道线i来说,要至少两个点才处理
            for k in range(out_j.shape[0]): #遍历每个参考行
                if out_j[k, i] > 0:  #k代表行 i代表grid
                    #图中的车道点位置
                    point_w =  int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                    point_h = int(img_h * (row_anchor[18-1-k]/288)) - 1
                    ppp = (point_w,  point_h)
                    # print(out_j[k, i])

                    cv2.circle(vis,ppp,5,(0,255,0),-1)
    
    cv2.imshow('lane',vis)
    cv2.waitKey(0)

from model.model import AutocoreNet
from data.constant import autocore_row_anchor
from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger
from utils.tool_func import get_linenumber

import configs.autocore as autocore_cfg

import torch, os
import scipy.special
import numpy as np
import cv2
import argparse

import time

def load_model(cfg):
    print(autocore_row_anchor)
    cls_num_per_lane=len(autocore_row_anchor)
        
    ## !!注意net是.cuda()的话 input也要相应的拷贝到gpu
    net = AutocoreNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()
    test_model = cfg.test_model
    print('test_model={}'.format(test_model))

    # 加载model
    state_dict = torch.load(test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        print(k)
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    print('model load success')

    return net

# return hxwxc ndarray
def img_resize(img,width,height):
    origin_w,origin_h = img.shape[1],img.shape[0]
    origin_r = 1.0*origin_w/origin_h

    empty_img=np.zeros([height,width,3],dtype=np.uint8)
    dim = None
    resized = None
    r = 1.0*width/height
    if origin_r < r:
        #新的图片的h = height
        r = height / float(origin_h)
        dim = (int(origin_w * r), height)
        # print('dim={}'.format(dim))
        resized = cv2.resize(img, dim)
        print(resized.shape)
        w_diff = int(width - int(origin_w * r))/2
        w_start,w_end = int(w_diff),int(width - w_diff)
        print(w_start,w_end)
        empty_img[:,w_start:w_end,:] = resized
    else:
        # 新的图片的w = width
        r = width / float(origin_w)
        dim = (width, int(origin_h * r))
        # print('dim={}'.format(dim))
        resized = cv2.resize(img, dim)

        h_diff = int(height - int(origin_h * r))/2
        h_start,h_end = int(h_diff),int(height - h_diff)
        empty_img[h_start:h_end,:,:] = resized

    return empty_img

low_prob_imgs=[]
def lane_detect(imPath,net,export_onnx=False,onnx_model_name='./lane.onnx',downsample_dim=(1440,1080)):
    global low_prob_imgs

    # print('detect {} begin'.format(imPath))
    new_w,new_h =  downsample_dim[0],downsample_dim[1]

    img = cv2.imread(imPath)
    img = img_resize(img,width=new_w,height=new_h)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img / 255.
    input = np.zeros([1, 3, new_h, new_w], dtype=np.float32)
    input[0,:,:,:] = img.transpose(2,0,1)
    input = torch.from_numpy(input)

    # 推理
    input=input.cuda()
    with torch.no_grad(): #reduce memory
        out = net(input)
        # print(out.shape) 

    if export_onnx:
        # 保存为onnx格式
        # onnx_model_name = test_model[-9:-4] + '.onnx'
        torch.onnx.export(net,               # model being run
                    input,                         # model input (or a tuple for multiple inputs)
                    onnx_model_name,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #                 'output' : {0 : 'batch_size'}}
                                    )

    # 后处理
    out_j = out[0].data.cpu().numpy() #[grid+1,anchor_lane_number,max_lane_num]
    # print('out_j[3,2,1]={}'.format(out_j[0,0,0]))

    # out_j = out_j[:, ::-1, :] #h这个维度倒序?
    
    debug_row,debug_lane = 1,0
    truth_grid=76
    # print('out_j[:,{},{}]={}'.format(debug_row,debug_lane,out_j[:,debug_row,debug_lane]))

    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) #grid这个维度只对前grid个值求概率 最后一个值用来表示是否存在车道线的点
    # print('prob shape={}'.format(prob.shape))
    
    debug_row,debug_lane = 0,0
    # print('prob[:,{},{}]={}'.format(debug_row,debug_lane,prob[truth_grid,debug_row,debug_lane]))
    # print('prob[:,{},{}]={}'.format(debug_row,debug_lane,prob[74:85,debug_row,debug_lane]))
    
    idx = np.arange(cfg.griding_num) + 1
    # print('idx shape={}'.format(idx.shape))
    # print('idx={}'.format(idx))
    # print(type(idx))
    idx = idx.reshape(-1, 1, 1)
    # print('idx shape={}'.format(idx.shape))
    # print('idx={}'.format(idx))

    solve_loc = 'argmax' #argmax or expection
    if solve_loc == 'expection':
        #这里作者没有直接用loc=argmax(prob,axis=0)求最大概率的位置. 原因如下:https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/99
        # print('prob*idx shape={}'.format((prob*idx).shape))

        loc_exp = prob * idx
        debug_row,debug_lane = 1,0
        # print('loc_exp[:,{},{}]={}'.format(debug_row,debug_lane,loc_exp[:,debug_row,debug_lane]))
        # print('loc_exp[:,{},{}]={}'.format(debug_row,debug_lane,np.sum(loc_exp[:,debug_row,debug_lane])))
        
        loc = np.sum(prob * idx, axis=0)   #loc为18 * 4矩阵,值为该位置的grid的期望.
        # print('line{},out_j shape={}'.format(get_linenumber(),out_j.shape))
        # print('line{},loc shape={}'.format(get_linenumber(),loc.shape))
        # print('loc[0,0]={}'.format(loc[1,0]))
    elif solve_loc == 'argmax':
        loc = np.argmax(prob,axis=0)

    out_j = np.argmax(out_j, axis=0) # 求出概率最大的下标
    # print('line{},out_j shape={}'.format(get_linenumber(),out_j.shape))
    loc[out_j == cfg.griding_num] = 0 #如果概率最大的下标为gridding_num的话说明是grid+1中的那个1. 则把loc相应位置的值置为0.表示在这个位置无车道线点.?
    out_j = loc
    # print('out_j shape={}'.format(out_j.shape))
    # print('out_j={}'.format(out_j))grid 
    col_sample = np.linspace(0, 1440 - 1, cfg.griding_num) 
    col_sample_w = col_sample[1] - col_sample[0] # 每个grid的像素数目

    #out_j.shape=(18,4)
    vis = cv2.imread(imPath)

    low_prob = False
    for i in range(out_j.shape[1]):  #遍历每条车道线 4
        #每条车道线有一个flag
        if np.sum(out_j[:, i] != 0) > 2: #对车道线i来说,要至少两个点才处理
            for k in range(out_j.shape[0]): #遍历每个参考行 9
                grid = out_j[k, i]
                if grid > 0: #k代表行 i代表grid  =0表示无车道线
                    #图中的车道点位置
                    max_prob = prob[grid,k,i]
                    if max_prob > 0.:
                        # print('max peob={}'.format(max_prob))
                        point_w =  int(out_j[k, i] * col_sample_w) - 1
                        point_h = int(autocore_row_anchor[k]) - 1
                        ppp = (point_w,  point_h)
                        # print(out_j[k, i])

                        #不同的线用不同颜色点标识
                        colors = [(0,0,255),(255,0,0),(255,255,255),(0,255,0)]
                        cv2.circle(vis,ppp,5,colors[i],-1)
                    if max_prob < 0.5:
                        low_prob = True
                        # print('max peob={}'.format(max_prob))
                        # print('near grid prob={}'.format(prob[:,k,i]))
    if low_prob:
        print('low prob,img={}'.format(imPath))
        low_prob_imgs.append(imPath)
    
    # cv2.imshow('lane',vis)
    # cv2.waitKey(0)

    cv2.imwrite('./lane.jpg',vis)

    return vis


if __name__ == "__main__":
    """
    python debug_inference2.py
    """
    args, cfg = merge_config()

    # cfg = autocore_cfg
 
    net = load_model(cfg)

    test_imgs = []
    print(cfg.test_work_dir)
    for e in os.listdir(cfg.test_work_dir):
        # print(e)
        if e[-3:] == 'jpg':
            #img_name = e[:-4] + 'jpg'
            # print(e)
            full_path = '{}/{}'.format(cfg.test_work_dir,e)
            # json_path = full_path[:-3] + 'json'
            # if os.path.isfile(json_path):
            #     test_imgs.append(full_path)

            test_imgs.append(full_path)
    
    #排序,key为x[-8:-4]
    test_imgs.sort(key=lambda x:int(x[-8:-4]))
    # print(test_imgs)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter('lane_detect.avi', fourcc , 10.0, (1440, 1080))
    begin = time.time()
    for i,img in enumerate(test_imgs):
        # imPath= '/home/train/hdd/sc/data/lane/autocore/frame0450.jpg'   
        # onnx_model_name = cfg.test_model[-9:-4] + '.onnx' 
        # print(i,':',img)
        vis = lane_detect(img,net,export_onnx=False,downsample_dim=(720,540))
        vout.write(vis)

        if i % 100 == 0:
            end = time.time()
            print('{}image,duration={}'.format(i+1,end-begin))
            print('average={}'.format((end-begin)/(i+1)))
    vout.release()
    
    # global low_prob_imgs
    print('low prob,low_prob_imgs={}'.format(low_prob_imgs))
    print(' len(low_prob_imgs)={}'.format(len(low_prob_imgs) ))
    # print(autocore_row_anchor)
    # cls_num_per_lane=len(autocore_row_anchor)
        
    # ## !!注意net是.cuda()的话 input也要相应的拷贝到gpu
    # net = AutocoreNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()
    # # test_model = '/home/train/hdd/sc/work/logs/20210513_172314_lr_4e-04_b_16/ep153.pth'
    # test_model = cfg.test_model
    # print('test_model={}'.format(test_model))

    # labels = []
    # for e in os.listdir(cfg.test_work_dir):
    #     if e[-4:] == 'json':
    #         #img_name = e[:-4] + 'jpg'
    #         print(e)
    #         labels.append(e)
    # print(labels)
    # # 准备模型输入
    # # imPath= './download/lishui_tl.png'
    # imPath= '/home/train/hdd/sc/data/lane/autocore/frame0450.jpg'

    # img = cv2.imread(imPath)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img = img / 255.
    # input = np.zeros([1, 3, 1080, 1440], dtype=np.float32)
    # input[0,:,:,:] = img.transpose(2,0,1)
    # input = torch.from_numpy(input)

    # # 加载model
    # state_dict = torch.load(test_model, map_location='cpu')['model']
    # compatible_state_dict = {}
    # for k, v in state_dict.items():
    #     # print(k)
    #     if 'module.' in k:
    #         compatible_state_dict[k[7:]] = v
    #     else:
    #         compatible_state_dict[k] = v

    # net.load_state_dict(compatible_state_dict, strict=False)
    # net.eval()
    # print('model load success')

    # # 推理
    # input=input.cuda()
    # out = net(input)
    # print(out.shape) 

    # # 保存为onnx格式
    # onnx_model_name = test_model[-9:-4] + '.onnx'
    # torch.onnx.export(net,               # model being run
    #               input,                         # model input (or a tuple for multiple inputs)
    #               onnx_model_name,   # where to save the model (can be a file or file-like object)
    #               export_params=True,        # store the trained parameter weights inside the model file
    #               opset_version=10,          # the ONNX version to export the model to
    #               do_constant_folding=True,  # whether to execute constant folding for optimization
    #               input_names = ['input'],   # the model's input names
    #               output_names = ['output'], # the model's output names
    #             #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #             #                 'output' : {0 : 'batch_size'}}
    #                             )

    # # 后处理
    # out_j = out[0].data.cpu().numpy() #[grid+1,anchor_lane_number,max_lane_num]
    # print('out_j[3,2,1]={}'.format(out_j[0,0,0]))

    # # out_j = out_j[:, ::-1, :] #h这个维度倒序?
    
    # debug_row,debug_lane = 1,0
    # print('out_j[:,{},{}]={}'.format(debug_row,debug_lane,out_j[:,debug_row,debug_lane]))


    # prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) #grid这个维度只对前grid个值求概率 最后一个值用来表示是否存在车道线的点
    # print('prob shape={}'.format(prob.shape))
    
    # debug_row,debug_lane = 1,0
    # print('prob[:,{},{}]={}'.format(debug_row,debug_lane,prob[:,debug_row,debug_lane]))
    
    # idx = np.arange(cfg.griding_num) + 1
    # # print('idx shape={}'.format(idx.shape))
    # # print('idx={}'.format(idx))
    # # print(type(idx))
    # idx = idx.reshape(-1, 1, 1)
    # print('idx shape={}'.format(idx.shape))
    # # print('idx={}'.format(idx))

    # #这里作者没有直接用loc=argmax(prob,axis=0)求最大概率的位置. 原因如下:https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/99
    # print('prob*idx shape={}'.format((prob*idx).shape))

    # loc_exp = prob * idx
    # debug_row,debug_lane = 1,0
    # print('loc_exp[:,{},{}]={}'.format(debug_row,debug_lane,loc_exp[:,debug_row,debug_lane]))
    # print('loc_exp[:,{},{}]={}'.format(debug_row,debug_lane,np.sum(loc_exp[:,debug_row,debug_lane])))
    
    # loc = np.sum(prob * idx, axis=0)   #loc为18 * 4矩阵,值为该位置的grid的期望.
    # # print('line{},out_j shape={}'.format(get_linenumber(),out_j.shape))
    # print('line{},loc shape={}'.format(get_linenumber(),loc.shape))
    # # print('loc[0,0]={}'.format(loc[1,0]))

    # out_j = np.argmax(out_j, axis=0) # 求出概率最大的下标
    # print('line{},out_j shape={}'.format(get_linenumber(),out_j.shape))
    # loc[out_j == cfg.griding_num] = 0 #如果概率最大的下标为gridding_num的话说明是grid+1中的那个1. 则把loc相应位置的值置为0.表示在这个位置无车道线点.?
    # out_j = loc
    # # print('out_j shape={}'.format(out_j.shape))
    # print('out_j={}'.format(out_j))

    # col_sample = np.linspace(0, 1440 - 1, cfg.griding_num) 
    # col_sample_w = col_sample[1] - col_sample[0] # 每个grid的像素数目

    # #out_j.shape=(18,4)
    # vis = cv2.imread(imPath)
    # for i in range(out_j.shape[1]):  #遍历每条车道线
    #     if np.sum(out_j[:, i] != 0) > 2: #对车道线i来说,要至少两个点才处理
    #         for k in range(out_j.shape[0]): #遍历每个参考行
    #             if out_j[k, i] > 0:  #k代表行 i代表grid
    #                 #图中的车道点位置
    #                 point_w =  int(out_j[k, i] * col_sample_w) - 1
    #                 point_h = int(autocore_row_anchor[k]) - 1
    #                 ppp = (point_w,  point_h)
    #                 # print(out_j[k, i])

    #                 cv2.circle(vis,ppp,5,(0,255,0),-1)
    
    # # cv2.imshow('lane',vis)
    # # cv2.waitKey(0)

    # cv2.imwrite('./lane.jpg',vis)
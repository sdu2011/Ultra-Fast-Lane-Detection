import os
import cv2
import tqdm
import numpy as np
import pdb
import json, argparse

def calc_k(line):
    '''
    Calculate the direction of lanes
    line=[567.0, 270.0, 532.0, 280.0, 496.0, 290.0, 461.0, 300.0, 425.0, 310.0, 390.0, 320.0, 355.0, 330.0, 319.0, 340.0, 284.0, 350.0, 248.0, 360.0, 213.0, 370.0, 177.0, 380.0, 142.0, 390.0, 106.0, 400.0, 71.0, 410.0, 35.0, 420.0]
    '''
    # print('line={}'.format(line))
    line_x = line[::2]
    line_y = line[1::2]
    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2) 
    if length < 90:
        return -10                                          # if the lane is too short, it will be skipped

    p = np.polyfit(line_x, line_y,deg = 1) #多项式拟合 deg=1代表拟合直线
    p1 = np.poly1d(p) #得到多项式系数，按照阶数从高到低排列
    # print('y={}'.format(p1))
    rad = np.arctan(p[0])
    
    # print('p={}'.format(p))
    # print('rad={}'.format(rad))

    return rad

def draw(im,line,idx,show = True):
    '''
    Generate the segmentation label according to json annotation
    '''
    # print('ssssssssss')
    line_x = line[::2]
    # print('line_x={}'.format(line_x))
    line_y = line[1::2]
    pt0 = (int(line_x[0]),int(line_y[0]))
    if show:
        cv2.putText(im,str(idx),(int(line_x[len(line_x) // 2]),int(line_y[len(line_x) // 2]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60 #下面cv2.line使用,像素亮度调到60倍
        
    for i in range(len(line_x)-1):
        # print(idx)
        #绘制的时候为啥亮度值搞这么低?  答:这里的亮度值和dataset.py中确认车道线位置时np.where(label_r == lane_idx)是相互统一,一一对应的.
        cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),(idx,),thickness = 16) 
        # cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),(255,0,0),thickness = 16)        
        pt0 = (int(line_x[i+1]),int(line_y[i+1]))


# raw_file : 每一个数据段的第20帧图像的的 path 路径
# lanes 和 h_samples 是数据具体的标注内容，为了压缩，h_sample 是纵坐标（等分确定），lanes 是每个车道的横坐标，是个二维数组。-2 表示这个点是无效的点。
# {"lanes": [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 567, 532, 496, 461, 425, 390, 355, 319, 284, 248, 213, 177, 142, 106, 71, 35, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], 
#     [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 646, 633, 621, 609, 596, 584, 571, 559, 546, 534, 522, 509, 497, 484, 472, 460, 447, 435, 422, 410, 397, 385, 373, 360, 348, 335, 323, 311, 298, 286, 273, 261, 248, 236, 224, 211, 199, 186, 174, 162, 149, 137, 124, 112, 100, 87], 
#     [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 690, 702, 713, 724, 736, 747, 758, 770, 781, 792, 804, 815, 827, 838, 849, 861, 872, 883, 895, 906, 917, 929, 940, 951, 963, 974, 985, 997, 1008, 1019, 1031, 1042, 1053, 1065, 1076, 1087, 1099, 1110, 1122, 1133, 1144, 1156, 1167, 1178, -2], 
#     [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 721, 754, 788, 821, 855, 888, 922, 955, 989, 1022, 1056, 1089, 1123, 1156, 1190, 1223, 1257, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]], 
# "h_samples": [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710], 
# "raw_file": "clips/0601/1494453497604532231/20.jpg"}

# 目的:提取出每张图里的有效车道线点的位置. 生成[ [l1 points] , [l2 points],[l3 points],[l3 points]]这种格式
def get_tusimple_list(root, label_list):
    '''
    Get all the files' names from the json annotation
    '''
    label_json_all = []
    for l in label_list:
        l = os.path.join(root,l)
        label_json = [json.loads(line) for line in open(l).readlines()]
        label_json_all += label_json
    names = [l['raw_file'] for l in label_json_all]
    h_samples = [np.array(l['h_samples']) for l in label_json_all]
    # print(h_samples)
    lanes = [np.array(l['lanes']) for l in label_json_all]

    line_txt = []
    for i in range(len(lanes)):
        line_txt_i = []
        for j in range(len(lanes[i])):
            if np.all(lanes[i][j] == -2): #-2代表这一行没有车道线点
                continue
            valid = lanes[i][j] != -2
            line_txt_tmp = [None]*(len(h_samples[i][valid])+len(lanes[i][j][valid]))
            line_txt_tmp[::2] = list(map(str,lanes[i][j][valid]))
            line_txt_tmp[1::2] = list(map(str,h_samples[i][valid]))
            line_txt_i.append(line_txt_tmp)
        line_txt.append(line_txt_i)

    return names,line_txt

# tusimple数据集标注顺序是无序的,脚本通过斜率来判断车道线的最左,次左,右,最右...
def generate_segmentation_and_train_list(root, line_txt, names):
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    train_gt_fp = open(os.path.join(root,'train_gt.txt'),'w')
    
    for i in tqdm.tqdm(range(len(line_txt))): #遍历每一张图

        tmp_line = line_txt[i]
        lines = []
        for j in range(len(tmp_line)): #遍历每一条车道线
            lines.append(list(map(float,tmp_line[j])))
        
        ks = np.array([calc_k(line) for line in lines])             # get the direction of each lane
        print('ks={}'.format(ks.shape))     #每条车道线的斜率

        k_neg = ks[ks<0].copy()       
        k_pos = ks[ks>0].copy()
        k_neg = k_neg[k_neg != -10]                                      # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()  #这里为什么要sort?
        k_pos.sort()

        label_path = names[i][:-3]+'png'
        print('label_path={}'.format(label_path))
        print('k_neg={}'.format(k_neg))
        print('k_pos={}'.format(k_pos))

        label = np.zeros((720,1280),dtype=np.uint8)
        bin_label = [0,0,0,0]
        if len(k_neg) == 1:                                           # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[1] = 1
        elif len(k_neg) == 2:                                         # for two lanes in the left
            print('********')
            which_lane = np.where(ks == k_neg[1])[0][0]
            print(which_lane)
            draw(label,lines[which_lane],1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[0] = 1
            bin_label[1] = 1
        elif len(k_neg) > 2:                                           # for more than two lanes in the left, 
            which_lane = np.where(ks == k_neg[1])[0][0]                # we only choose the two lanes that are closest to the center
            draw(label,lines[which_lane],1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[0] = 1
            bin_label[1] = 1

        if len(k_pos) == 1:                                            # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],3)
            bin_label[2] = 1
        elif len(k_pos) == 2:
            which_lane = np.where(ks == k_pos[1])[0][0]
            draw(label,lines[which_lane],3)
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],4)
            bin_label[2] = 1
            bin_label[3] = 1
        elif len(k_pos) > 2:
            which_lane = np.where(ks == k_pos[-1])[0][0]
            draw(label,lines[which_lane],3)
            which_lane = np.where(ks == k_pos[-2])[0][0]
            draw(label,lines[which_lane],4)
            bin_label[2] = 1
            bin_label[3] = 1

        cv2.imwrite(os.path.join(root,label_path),label)

        train_gt_fp.write(names[i] + ' ' + label_path + ' '+' '.join(list(map(str,bin_label))) + '\n')
    train_gt_fp.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Tusimple dataset')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()

    # training set
    # label_list = ['label_data_0601.json','label_data_0531.json','label_data_0313.json']
    label_list = ['test.json']
    names,line_txt = get_tusimple_list(args.root, label_list )
    print('names={},line_txt={}'.format(names,line_txt))
    # generate segmentation and training list for training
    generate_segmentation_and_train_list(args.root, line_txt, names)

    # testing set
    names,line_txt = get_tusimple_list(args.root, ['test_tasks_0627.json'])
    # generate testing set for testing
    with open(os.path.join(args.root,'test.txt'),'w') as fp:
        for name in names:
            fp.write(name + '\n')


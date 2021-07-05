import os
import cv2
import tqdm
import numpy as np
import pdb
import json, argparse

# def calc_k(line):
#     '''
#     Calculate the direction of lanes
#     '''
#     line_x = line[::2]
#     line_y = line[1::2]
#     length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
#     if length < 90:
#         return -10                                          # if the lane is too short, it will be skipped

#     p = np.polyfit(line_x, line_y,deg = 1)
#     print('p={}'.format(p))
#     rad = np.arctan(p[0])
    
#     return rad

def draw(im,line_points,idx,show = True):
    """
        line_points中点的顺序由labelme标注顺序决定
    """
    print('idx={}'.format(idx))
    
    if show:
        points_num = len(line_points)
        pos = line_points[points_num//2]
        print('pos={}'.format(pos))

        # 这里要注意,putText会使得非车道线像素点的像素值不为0.
        # cv2.putText(im,str(idx),(int(pos[0]),int(pos[1])-20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 30 #下面cv2.line使用,像素亮度调到30倍
    
    pt_pre = ( int(line_points[0][0]),int(line_points[0][1]) )
    
    for i in range(len(line_points) - 1):
        point_next = ( int(line_points[i + 1][0]), int(line_points[i + 1][1]) ) 
        # print(type(point_next),point_next)
        #第idx条车道线点的亮度为idx  这里的亮度值要和dataset.py中_get_index中的np.where(img_r == lane_idx)相匹配
        cv2.line(im,pt_pre,(int(point_next[0]),int(point_next[1])),(idx,),thickness = 16)  
        pt_pre = point_next


def get_autocore_list(root,label):
    # Opening JSON file
    f = open(os.path.join(root,label),)
    data = json.load(f)

    img_path = data['imagePath']

    label_info = {}
    label_info['imagePath'] = img_path 
    for l in data['shapes']:
        # print(l['label'])
        # print(l['points'])

        lane = l['label']
        points = l['points']
        if lane in label_info.keys():
            label_info[lane] = label_info[lane] + points
        else:
            label_info[lane] = points

    return label_info

def generate_segmentation_and_train_list(root, label_info,train_gt_fp):
    # print(label_info)
    
    label_img = np.zeros((1080,1440),dtype=np.uint8)
    
    lane_num = len(label_info.keys()) - 1
    
    #绘制label  记录下车道线id
    lane_ids=[]
    print(label_info.keys())
    for key in label_info.keys():
        if key.startswith( 'lane' ):
            lane_points = label_info[key]
            
            lane_idx = 0
            if key == 'lane_l':
                lane_idx = 1
            elif key == 'lane_r':
                lane_idx = 4
            else:
                lane_idx = 1 + int(key[-1])
            
            lane_ids.append(lane_idx)

            draw(label_img,lane_points,lane_idx)
    
    # for i in range(1,lane_num + 1):
    #     # key = 'lane{}'.format(i)

    #     if key in label_info.keys():
    #         lane_points = label_info['lane{}'.format(i)]
    #         draw(label_img,lane_points,i)

    #         lane_ids.append(i)

    # cv2.imshow("label_img",label_img)
    label_img_path = label_info['imagePath'][:-3] + 'png'
    cv2.imwrite(os.path.join(root,label_img_path),label_img)

    origin_img_path = label_info['imagePath']
    lane_ids = map(str,lane_ids)
    train_gt_fp.write(origin_img_path + ' ' + label_img_path + ' '+ ' '.join(lane_ids) + '\n') 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Tusimple dataset')
    return parser

if __name__ == "__main__":
    """usage: python convert_autocore.py --root /home/suchang/data/lane/autocore_0622"""
    args = get_args().parse_args()

    # training set
    labels = []
    for e in os.listdir(args.root):
        if e[-4:] == 'json':
            #img_name = e[:-4] + 'jpg'
            # print(e)
            labels.append(e)
    # print(labels)

    train_gt_fp = open(os.path.join(args.root,'train_gt.txt'),'w')
    for label in labels:
        label_info = get_autocore_list(args.root,label)
        # print(label_info)
        # # generate segmentation and training list for training
        generate_segmentation_and_train_list(args.root, label_info,train_gt_fp)

    train_gt_fp.close()


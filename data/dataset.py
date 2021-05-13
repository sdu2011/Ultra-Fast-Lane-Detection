import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=50, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()
            # print('self.list={}'.format(self.list))

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        print('img_name={},label_name={}'.format(img_name,label_name))
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
    
        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)  
        print('lane_pts={}'.format(lane_pts))
        # get the coordinates of lanes at row anchors

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size
        print('label size={}'.format(label.size))

        #这里为何要这么做? self.row_anchor不是已经区分culane还是tusimple了吗 800*288是culane的数据集的图片大小 1280*720是tusimple数据集大小
        #答:因为constant.py中定义的tusimple_row_anchor是以h为288作为参考来定义的.
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))

        #第三个维度存的数据的含义1:原图中的参考行的行数r 2.第r行对应的第x条车道线像素的均值
        #举个具体例子,有10个参考行,其中第5个参考行为原图中的第50行,有两条车道线,lane1的像素在本行的列数为30,31,32,lane2的像素在本行的列数为70,71,72的话,
        #则all_idx[0,5,:]=50,31   all_idx[1,5,:]=50,71
        all_idx = np.zeros((self.num_lanes,len(sample_tmp),2))  # 
        # print('all_idx shape={}'.format(all_idx.shape))
        # print('sample_tmp={}'.format(sample_tmp))
        for i,r in enumerate(sample_tmp):
            # print('i={},r={}'.format(i,r))
            label_r = np.asarray(label)[int(round(r))]  # np.asarray(label): 720x1280 ndarray
            # print('label_r shape={}'.format(label_r.shape))
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0] #np.where
                if len(pos) == 0: #说明第r行没有车道线lane_idx
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1 
                    continue

                pos = np.mean(pos)  #第r行,第lane_idx条车道线的平均列位置.
                all_idx[lane_idx - 1, i, 0] = r  # 是哪一个参考行
                all_idx[lane_idx - 1, i, 1] = pos # 第lane_idx-1个车道线 第r行车道线的平均列位置.

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i,:,1] != -1 
            # print('valid shape={}'.format(valid.shape)) #(len(sample_tmp),)
            # get all valid lane points' index
            valid_idx = all_idx_cp[i,valid,:]
            # get all valid lane points
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp



class AutocoreLaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=50, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(AutocoreLaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()
            # print('self.list={}'.format(self.list))

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        # print('img_name={},label_name={}'.format(img_name,label_name))
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        # 加载gt图
        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        # 加载原图
        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
    
        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)  
        # print('lane_pts={}'.format(lane_pts))
        # get the coordinates of lanes at row anchors

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size
        # print('label size={}'.format(label.size))

        #第三个维度存的数据的含义1:原图中的参考行的行数r 2.第r行对应的第x条车道线像素的列的均值
        #举个具体例子,有10个参考行,其中第5个参考行为原图中的第50行,有两条车道线,lane1的像素在本行的列数为30,31,32,lane2的像素在本行的列数为70,71,72的话,
        #则all_idx[0,5,:]=50,31   all_idx[1,5,:]=50,71
        all_idx = np.zeros((self.num_lanes,len(self.row_anchor),2))  # 
        # print('all_idx shape={}'.format(all_idx.shape))
   
        for i,r in enumerate(self.row_anchor):  #遍历每一个参考行  
            # print('i={},r={}'.format(i,r))
            img = np.asarray(label) # 1080 * 1440

            img_r = img[r]  # 第r行的img 1440

            for lane_idx in range(1, self.num_lanes + 1):
                pixel = 30 * lane_idx  #数据集制作时当前lane的像素点的亮度值. 参考脚本convert_autocore.py
                pos = np.where(img_r == pixel)[0] # 第lane_idx条车道线的列数 
                # print('lane{} in row{} cols={}'.format(lane_idx,r,pos))
                # print('pos shape={}'.format(pos.shape))
                if len(pos) == 0: #说明第r行没有车道线lane_idx
                    # print('no lane{} point in row{}'.format(lane_idx,r))
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1 
                    continue

                pos = np.mean(pos)  #第r行,第lane_idx条车道线的平均列位置.
                all_idx[lane_idx - 1, i, 0] = r  # 是哪一个参考行
                all_idx[lane_idx - 1, i, 1] = pos # 第lane_idx-1个车道线 第r行车道线的平均列位置.

        # 这一部分可以通过标注数据集的时候标注仔细一点来避免.
        # 防止数据集标注的点太少,通过代码逻辑延长车道线至图像底部
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i,:,1] != -1  # 有效参考行. 即该行有车道线.
            # print('valid shape={}'.format(valid.shape)) 
            
            # 获取有效参考行的车道线点
            valid_idx = all_idx_cp[i,valid,:]
            # print('valid_idx shape={}'.format(valid_idx.shape))  #(n,2) 

            # get all valid lane points
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:  # if 最后一个有效行的y = 倒数第一个参考行的y
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue

            # 有效参考行太少
            if len(valid_idx) < 6:
                continue
                
            # 用后面一半的有效参考行的车道线点做直线拟合
            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp

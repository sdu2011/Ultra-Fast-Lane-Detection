import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1) #转成概率
        # print('scores={}'.format(scores[0,]))
        factor = torch.pow(1.0-scores, self.gamma)#求每个概率的权重w=(1-p)的gamma次方
        log_score = F.log_softmax(logits, dim=1)#求log(p)
        log_score = factor * log_score#每个元素都变成w*log(p)
        loss = self.nll(log_score, labels)#取labels标识的log_score求平均
        # print('loss={}'.format(loss))

        #debug code 
        # print('scores={}'.format(scores.shape)) #lane1 [1,101,9,4]
        # print('labels={}'.format(labels.shape) )#lane1 [1,9,4]
        # lane1_pre= scores[0,:,:,0].cpu() #[101,9]
        # print('lane1_pre shape={}'.format(lane1_pre.shape))
        # lane1_loc = labels[0,:,0] 
        # print(lane1_loc)
        
        # scores = scores.cpu()
        # labels = labels.cpu() #cpu上的错误提示更友好一些
        # lane_num = labels.shape[2]
        # anchor_row_num = labels.shape[1]
        # for i in range(anchor_row_num):
        #     for j in range(lane_num):
        #         if j == 0:  #debug lane1
        #             truth_grid = labels[0,i,j] #acnhor_row_i truth grid
        #             # true_grid_preprob = lane1_pre.index_select(0,lane1_loc.cpu()) #在维度1上选择特定下标
        #             pre_prob = scores[0,truth_grid,i,j]
        #             print('lane{},anchor{},truth_grid={},prob={}'.format(j,i,truth_grid,pre_prob))

        return loss

class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
    def forward(self,logits):
        n,c,h,w = logits.shape
        loss_all = []
        for i in range(0,h-1):
            loss_all.append(logits[:,:,i,:] - logits[:,:,i+1,:])#一阶差分　保证平滑
        #loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))



class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        pos = torch.sum(x*embedding,dim = 1)

        diff_list1 = []
        for i in range(0,num_rows // 2):
            diff_list1.append(pos[:,i,:] - pos[:,i+1,:])#

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l1(diff_list1[i],diff_list1[i+1]) #二阶差分　保证是直线
        loss /= len(diff_list1) - 1
        return loss


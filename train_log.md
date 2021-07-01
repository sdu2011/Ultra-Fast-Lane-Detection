# 20210616
每隔15帧标注一张图.共计80张图.

batch=8. 再多显存不够. 多gpu训练.

## test1
``` python
        train_dataset = AutocoreLaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=None,
                                           simu_transform = None,  #不做数据增强
                                           griding_num=griding_num, 
                                           row_anchor = autocore_row_anchor,
                                           segment_transform=None,use_aux=use_aux, num_lanes = num_lanes)
```

网络配置:
```
backbone = '18'
griding_num = 100
use_aux = False

autocore_row_anchor = [421,444,475,507,535,570,618,667,712]
```

在epoch=100时,loss=0.045左右.下降缓慢.

## test2
``` python
        simu_transform = mytransforms.Compose2([
                mytransforms.RandomRotate(6), #随机旋转+-6度.
                mytransforms.RandomUDoffsetLABEL(100),#随机上下移动x个像素
                mytransforms.RandomLROffsetLABEL(200) #随机左右平移x个像素,x<200.
            ])
        train_dataset = AutocoreLaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=None,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = autocore_row_anchor,
                                           segment_transform=None,use_aux=use_aux, num_lanes = num_lanes)
```

网络配置:
``` python
backbone = '18'
griding_num = 100
use_aux = False

autocore_row_anchor = [421,444,475,507,535,570,618,667,712]
```

在epoch=100时,loss=0.13左右.跳变严重. epoch=500时,loss=0.04,基本稳定. 在未标注数据上实际测试,效果糟糕.


## test3
``` python
        simu_transform = mytransforms.Compose2([
                mytransforms.RandomRotate(6),
                mytransforms.RandomColorBright(2)  #亮度随机调整
                # mytransforms.RandomUDoffsetLABEL(100),
                # mytransforms.RandomLROffsetLABEL(200)
            ])
        train_dataset = AutocoreLaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=None,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = autocore_row_anchor,
                                           segment_transform=None,use_aux=use_aux, num_lanes = num_lanes)
```

网络配置:
``` python
backbone = '18'
griding_num = 100
use_aux = False

autocore_row_anchor = [421,444,475,507,535,570,618,667,712]
```

在epoch=100时,loss=0.13左右. epoch=335时,loss=0.05,基本稳定. 在未标注数据上实际测试,效果糟糕.


## test4
``` python
        simu_transform = mytransforms.Compose2([
                mytransforms.RandomRotate(6),
                mytransforms.RandomColorBright(2),
                mytransforms.RandomColorContrast(3) #对比度随机调整
            ])
        train_dataset = AutocoreLaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=None,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = autocore_row_anchor,
                                           segment_transform=None,use_aux=use_aux, num_lanes = num_lanes)
```

网络配置:
``` python
batch_size = 12

backbone = '18'
griding_num = 100
use_aux = False

autocore_row_anchor = [421,444,475,507,535,570,618,667,712]
```

在epoch=100时,loss=0.15左右,不稳定,跳变严重. epoch=500时,loss基本稳定在0.045左右. 在未标注数据上实际测试,效果糟糕.

# 20210617
通过查看标注数据,在车道线远端,各个车道线已经非常接近,难以区分. 所以考虑
1. 增大每一个参考行的grid的数目
2. 改变参考行,使其不要位于较为远端的行
3. 重新标注数据集

## test1
相比20160616 test4更改anchor_row.
``` python
# autocore_row_anchor = [421,444,475,507,535,570,618,667,712]
autocore_row_anchor = [475,507,535,570,618,667,712,756,800]
```

epoch=300时,loss=0.035左右,基本稳定.  实测效果差.


## test2
更改grid_num
``` python
# griding_num = 100
griding_num = 200
```
epoch=500,loss稳定在0.024. 实测效果差.

## test3
**训练集里镜头上有雨水的画面过多,甚至造成车道线扭曲.重新选取图片质量还行的图片.总计10张.**
``` python
gamma=0 #不使用focalloss
```
epoch=100,loss在0.2波动.top1=1,top2=1,top3=1.

## test4
``` python
gamma = 0.2  #w=(1-p)^gamma
```
epoch=100,loss在0.18左右波动.相对于普通交叉熵loss的优势不明显.top1=1,top2=1,top3=1.

## test5
针对frame0000.jpg. 只训练一张图片.在epoch=66,loss=0.095停止.此时的模型对frame0000.jpg的检测效果依然很差.
需要思考:为什么此时的loss却很低?
![avatar](./lane_fail.jpg)


修改backbone,搞成强力backbone,先冲着过拟合去,看看loss能优化到什么程度.
``` python
# NETWORK
backbone = '101'
# griding_num = 100
griding_num = 200
use_aux = False
```
不做任何数据增强
``` python
        simu_transform = mytransforms.Compose2([
                # mytransforms.RandomRotate(6),
                # mytransforms.RandomColorBright(2),
                # mytransforms.RandomColorContrast(3)
                # mytransforms.RandomUDoffsetLABEL(100),
                # mytransforms.RandomLROffsetLABEL(200)
            ])
```

epoch=38,loss=0.047.测试.
![lane1](./docs/lane1.jpg)


通过对'SoftmaxFocalLoss'的调试,查看输出概率和真值概率,发现输出概率正常.  检查推理代码,grid计算算的是期望,改为由argmax计算.结果正常.  

# 20210618
标注数据集增大到184张.

## test1
``` python
# NETWORK
backbone = '18'
griding_num = 100
use_aux = False
```

做数据增强
``` python
        simu_transform = mytransforms.Compose2([
                mytransforms.RandomRotate(6),
                mytransforms.RandomColorBright(2),
                mytransforms.RandomColorContrast(3)
            ])
```
epoch=150时,loss=0.13左右跳变.




---------
# 调参技巧
## 大方向
1. 刚开始，先上小规模数据，模型往大了放（能用256个filter就别用128个），直接奔着过拟合去（此时都可不用测试集验证集）验证训练脚本的流程。小数据量，速度快，便于测试。  
如果小数据、大网络，还不能过拟合，需要检查输入输出、代码是否错误、模型定义是否恰当、应用场景是否正确理解。比较神经网络没法拟合的问题，这种概率太小了。

2. loss设计

--------
# 20210701
溧水日间数据. 标注总计144张.




import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


# 短路连接Block
class ShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortcutBlock, self).__init__()
        self.layer1 = self.conv_layer(in_channels, out_channels, kernel_size=1, stride=1)
        self.layer2 = self.conv_layer(out_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output += x
        return output

    def conv_layer(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )
        return layer


# Yolo层Block
class YoloBlock(nn.Module):
    def __init__(self, anchor_wh, _dataset='voc'):
        super(YoloBlock, self).__init__()
        self.dataset = _dataset
        self.num_classes = 20 if self.dataset == 'voc' else 80
        self.anchor_wh = anchor_wh
        self.img_dim = 416
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, targets=None):
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.batch_size = x.shape[0]
        self.grid_size = x.shape[2]                   # 特征图宽高尺寸
        self.stride = self.img_dim / self.grid_size   # 相比原图缩放倍数

        self.gen_anchors() 
        x = x.view(self.batch_size, 3, self.num_classes+5, self.grid_size, self.grid_size).permute(0,1,3,4,2).contiguous()

        shift_x = torch.sigmoid(x[..., 0])    # center x
        shift_y = torch.sigmoid(x[..., 1])    # center y
        shift_w = x[..., 2]                   # width 
        shift_h = x[..., 3]                   # height
        pred_conf = torch.sigmoid(x[..., 4])  # conf
        pred_cls = torch.sigmoid(x[..., 5:])  # cls

        pred_boxes = FloatTensor(x[..., :4].shape)
        pred_boxes[..., 0] = shift_x.detach() + self.grid_x
        pred_boxes[..., 1] = shift_y.detach() + self.grid_y
        pred_boxes[..., 2] = torch.exp(shift_w.detach()) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(shift_h.detach()) * self.anchor_h

        output = torch.cat((pred_boxes.view(self.batch_size, -1, 4) * self.stride, 
                            pred_conf.view(self.batch_size, -1, 1),
                            pred_cls.view(self.batch_size, -1, self.num_classes)), dim=-1)       # (1,507,85)

        if targets is None:
            return output, 0
        else:
            self.gen_targets(targets)

            loss_x = self.mse_loss(shift_x[self.obj_mask], self.tx[self.obj_mask])
            loss_y = self.mse_loss(shift_y[self.obj_mask], self.ty[self.obj_mask])
            loss_w = self.mse_loss(shift_w[self.obj_mask], self.tw[self.obj_mask])
            loss_h = self.mse_loss(shift_h[self.obj_mask], self.th[self.obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[self.obj_mask], self.tconf[self.obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[self.noobj_mask], self.tconf[self.noobj_mask])
            loss_conf = loss_conf_obj + 100*loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[self.obj_mask], self.tcls[self.obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return output, total_loss
        
    # 生成anchors
    def gen_anchors(self):
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        g = self.grid_size
        self.grid_x = torch.arange(g).repeat(g,1).view(1,1,g,g).type(FloatTensor)       # (1,1,13,13)
        self.grid_y = torch.arange(g).repeat(g,1).t().view(1,1,g,g).type(FloatTensor)   # (1,1,13,13)
        self.anchor_wh = FloatTensor([(wi/self.stride,hi/self.stride) for wi,hi in self.anchor_wh]) # (3,2)
        self.anchor_w = self.anchor_wh[:,0].view(1,3,1,1)
        self.anchor_h = self.anchor_wh[:,1].view(1,3,1,1)

    # 生成训练样本
    def gen_targets(self, targets):
        # targets:第一列表示属于第几个img，后面依次为label,center-x,center-y,w,h
        b = self.batch_size
        g = self.grid_size

        BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        self.obj_mask = BoolTensor(b,3,g,g).fill_(0)
        self.noobj_mask = BoolTensor(b,3,g,g).fill_(1)
        self.tx = FloatTensor(b,3,g,g).fill_(0) 
        self.ty = FloatTensor(b,3,g,g).fill_(0)
        self.tw = FloatTensor(b,3,g,g).fill_(0)
        self.th = FloatTensor(b,3,g,g).fill_(0)
        self.tconf = FloatTensor(b,3,g,g).fill_(0)
        self.tcls = FloatTensor(b,3,g,g,self.num_classes).fill_(0)

        batch, target_labels = targets[:, 0:2].long().t()
        targets_boxes = targets[:, 2:6] * g  # 还原到13x13的尺寸
        gx = targets_boxes[:, 0]
        gy = targets_boxes[:, 1]
        gw = targets_boxes[:, 2]
        gh = targets_boxes[:, 3]
        gi = targets_boxes[:, 0].long()  # 化为整型的坐标
        gj = targets_boxes[:, 1].long()

        ious = torch.stack([cal_wh_iou(anchor, gw, gh) for anchor in self.anchor_wh]) # (3,n)
        best_ious, best_n = ious.max(0)

        self.obj_mask[batch, best_n, gj, gi] = 1
        self.noobj_mask[batch, best_n, gj, gi] = 0

        for i,each_ious in enumerate(ious.t()):  # 将iou>0.5的预测点都视为有目标
            self.noobj_mask[batch[i], each_ious>0.5, gj[i], gi[i]] = 0

        self.tx[batch, best_n, gj, gi] = gx - gx.floor()
        self.ty[batch, best_n, gj, gi] = gy - gy.floor()
        self.tw[batch, best_n, gj, gi] = torch.log(gw / self.anchor_wh[best_n, 0] + 1e-16)
        self.th[batch, best_n, gj, gi] = torch.log(gh / self.anchor_wh[best_n, 1] + 1e-16)
        self.tcls[batch, best_n, gj, gi, target_labels] = 1
        self.tconf = self.obj_mask.float()


class YoloNet(nn.Module):
    def __init__(self, _dataset='voc'):
        super(YoloNet, self).__init__()
        self.dataset = _dataset

        self.conv1 = self.conv_layer( 3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = self.conv_layer(32, 64, kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(1):
            layers.append(ShortcutBlock(64, 32))
        self.shortcut1 = nn.Sequential(*layers)
        self.pool1 = self.conv_layer(64, 128, kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(2):
            layers.append(ShortcutBlock(128, 64))
        self.shortcut2 = nn.Sequential(*layers)    
        self.pool2 = self.conv_layer(128, 256, kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(8):
            layers.append(ShortcutBlock(256, 128))
        self.shortcut3 = nn.Sequential(*layers)    
        self.pool3 = self.conv_layer(256, 512, kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(8):
            layers.append(ShortcutBlock(512, 256))
        self.shortcut4 = nn.Sequential(*layers)
        self.pool4 = self.conv_layer(512, 1024, kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(4):
            layers.append(ShortcutBlock(1024, 512))
        self.shortcut5 = nn.Sequential(*layers)
            
        self.conv5x1 = self.conv5x_layer(1024, 512)
        self.conv2x1 = self.conv2x_layer(512, 1024)
        self.conv1x1 = self.conv_layer(512, 256, kernel_size=1, stride=1)
      
        self.conv5x2 = self.conv5x_layer(768, 256, flag=True)
        self.conv2x2 = self.conv2x_layer(256, 512)
        self.conv1x2 = self.conv_layer(256, 128, kernel_size=1, stride=1)

        self.conv5x3 = self.conv5x_layer(384, 128, flag=True)
        self.conv2x3 = self.conv2x_layer(128, 256)

        self.yolo1 = YoloBlock(anchor_wh=[(116,90), (156,198), (373,326)])
        self.yolo2 = YoloBlock(anchor_wh=[(30,61), (62,45), (59,119)])
        self.yolo3 = YoloBlock(anchor_wh=[(10,13), (16,30), (33,23)])

    def forward(self, x, targets=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.shortcut1(x)
        x = self.pool1(x)
        x = self.shortcut2(x)
        x = self.pool2(x)
        x = self.shortcut3(x)
        feature_map3 = x
        x = self.pool3(x)
        x = self.shortcut4(x)
        feature_map2 = x
        x = self.pool4(x)
        x = self.shortcut5(x)
        x = self.conv5x1(x)
        feature_map1 = self.conv2x1(x)
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, feature_map2], 1)
        x = self.conv5x2(x)
        feature_map2 = self.conv2x2(x)
        x = self.conv1x2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, feature_map3], 1)
        x = self.conv5x3(x)
        feature_map3 = self.conv2x3(x)

        output1, loss1 = self.yolo1(feature_map1, targets)
        output2, loss2 = self.yolo2(feature_map2, targets)
        output3, loss3 = self.yolo3(feature_map3, targets)
        output = torch.cat((output1,output2,output3), 1)
        loss = loss1 + loss2 + loss3

        return output, loss

    # 基础卷积pack
    def conv_layer(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )
        return layer

    # 5个连续卷积pack
    def conv5x_layer(self, in_channels, out_channels, flag=False):
        layer1 = self.conv_layer(in_channels, out_channels, kernel_size=1, stride=1)
        if flag:  # 由于cat连接导致通道数的变化需要额外处理
            in_channels = out_channels*2
        layer2 = self.conv_layer(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        layer3 = self.conv_layer(in_channels, out_channels, kernel_size=1, stride=1)
        layer4 = self.conv_layer(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        layer5 = self.conv_layer(in_channels, out_channels, kernel_size=1, stride=1)
        return nn.Sequential(*[layer1, layer2, layer3, layer4, layer5])

    # 2个连续卷积pack
    def conv2x_layer(self, in_channels, out_channels):
        layer1 = self.conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.dataset == 'coco':
            layer2 = nn.Conv2d(out_channels, 255, kernel_size=1, stride=1)
        elif self.dataset == 'voc':
            layer2 = nn.Conv2d(out_channels, 75, kernel_size=1, stride=1)
        return nn.Sequential(*[layer1, layer2])

import torch
import numpy as np
from tqdm import tqdm


# 网络权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 中心宽高转化为左上右下
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# 只根据宽高计算可能的最大IOU值
def cal_wh_iou(anchor, gw, gh):
    # anchor:[anchor_w,anchor_h]  gw:[gw1,gw2,...] gh:[gh1,gh2,...]
    inter_area = torch.min(anchor[0], gw) * torch.min(anchor[1], gh)
    union_area = anchor[0] * anchor[1] + gw * gh - inter_area
    return inter_area / union_area  # [a1,a2,...]


# 默认x1y1x2y2的形式,支持1对多，相等的多对多计算
def cal_iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1.t()
    box2_x1, box2_y1, box2_x2, box2_y2 = box2.t()

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)
    box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-16)


# nms过滤
def nms(predictions):
    predictions[..., :4] = xywh2xyxy(predictions[..., :4])
    output = [None for i in range(len(predictions))]  # 共batch_size个

    for img_i, img_pred in enumerate(predictions):
        img_pred = img_pred[img_pred[:, 4] >= 0.5]  # 过滤掉置信度低的
        if not img_pred.size(0):
            continue

        score = img_pred[:, 4] * img_pred[:, 5:].max(1)[0]
        img_pred = img_pred[(-score).argsort()]  # 按概率从大到小排序
        cls_conf, label = img_pred[:, 5:].max(1, keepdim=True)
        img_pred = torch.cat((img_pred[:, :5], cls_conf.float(), label.float()), 1)

        keep_boxes = []
        while img_pred.size(0):
            label_overlap = cal_iou(img_pred[0, :4].unsqueeze(0), img_pred[:, :4]) > 0.4  # iou超过阈值的
            label_match = img_pred[0, -1] == img_pred[:, -1]  # 预测标签一样的
            invalid = label_overlap & label_match

            weights = img_pred[invalid, 4]
            img_pred[0, :4] = (weights * img_pred[invalid, :4]).sum(0) / weights.sum()  # 预测框融合

            keep_boxes += [img_pred[0]]
            img_pred = img_pred[~invalid]

        if keep_boxes:
            output[img_i] = torch.stack(keep_boxes)

    return output


# soft_nms过滤
def soft_nms():
    pass


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
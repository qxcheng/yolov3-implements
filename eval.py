import torch
from torch.autograd import Variable
from model import YoloNet
from datasets import MyDataset
from utils import *

def load_classes(path):
    with open(path, "r") as f:
        names = [name.rstrip('\n').rstrip() for name in f.readlines()]
    return names

# 评估
def eval(valid_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    class_names = load_classes('D:/datasets/voc-custom/classes.names')

    model = YoloNet().to(device)
    model.load_state_dict(torch.load('yolov3_ckpt_10.pth'))
    model.eval()

    dataset = MyDataset(valid_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )
    labels = []
    total_metrics = []
    for batch_i, (imgs, targets) in enumerate(dataloader):
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:]) * 416

        imgs = Variable(imgs.type(FloatTensor), requires_grad=False)

        with torch.no_grad():
            outputs, _ = model(imgs)
            outputs = nms(outputs)

        # 遍历每张图
        batch_metrics = []
        for img_i in range(len(outputs)):
            if outputs[img_i] is None:
                continue

            output = outputs[img_i]
            pred_boxes = output[:, :4]
            pred_scores = output[:, 4]
            pred_labels = output[:, -1]

            true_positives = np.zeros(pred_boxes.shape[0])

            target = targets[targets[:, 0] == img_i][:, 1:]
            target_labels = target[:, 0] if len(target) else []

            if len(target):
                detected_boxes = []  # 预测正确的标签框索引
                target_boxes = target[:, 1:]
                # 遍历每个预测框
                for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                    if len(detected_boxes) == len(target):
                        break
                    if pred_label not in target_labels:
                        continue
                    iou, box_index = cal_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                    if iou >= 0.5 and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]

            batch_metrics.append([true_positives, pred_scores, pred_labels])
        total_metrics += batch_metrics

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")

if __name__ == '__main__':
    eval('D:/datasets/voc-custom/valid.txt')
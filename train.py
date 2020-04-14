import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler

from model import YoloNet
from datasets import MyDataset
from utils import *

import time

# 训练
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    dataset = MyDataset(data_path='D:/datasets/voc-custom/train.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=3, pin_memory=True, collate_fn=dataset.collate_fn
    )

    model = YoloNet().to(device)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters())
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(10):
        model.train()
        #scheduler.step()

        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            print(imgs, targets)

            output, loss = model(imgs, targets)
            print(epoch,batch_i,loss.detach().cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"yolov3_ckpt_%d.pth" % epoch)

if __name__ ==  '__main__':
    train()
import torch
import torchvision.transforms as transforms
from PIL import  Image
from model import YoloNet
from datasets import pad_to_square, resize
from utils import nms

# 预测
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YoloNet().to(device)
    model.load_state_dict(torch.load('yolov3_ckpt_10.pth'))

    img_path = ''
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    img, _ = pad_to_square(img, 0)
    img = resize(img, 416)
    img = img.to(device)

    #print(net)
    outputs, _ = model(img)
    outputs = nms(outputs)

    print(outputs)


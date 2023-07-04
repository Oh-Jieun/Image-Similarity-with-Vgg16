import os
import cv2
import torch
from PIL import Image
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

#이미지 폴더에서 데이터 로드
train_set = ImageFolder(
    root="C:/Users/KETI/Desktop/imagedata",
    
    transform=transforms.Compose([
    transforms.Resize((224,224)),   #이미지 크기가 다양하므로, Resize로 크기 통일
    transforms.ToTensor(),
    ])
    )

# compute
def torch_L2Norm(pred1, pred2):
    return torch.pow(torch.sum(torch.pow(pred1 - pred2, 2)), 0.5)

# 1. data 불러오기
imgPath = r"C:\Users\KETI\Desktop\validation\easy\easy11"
imgList = os.listdir(imgPath)
imgList = [os.path.join(imgPath, file) for file in imgList]
imgList = [file for file in imgList if file.endswith(".jpg") or file.endswith(".png")]

# 2. model 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
model = model.to(device)
model = model.eval()

global preprocess
preprocess=transforms.Compose([
        transforms.Resize((224,224)),  
        transforms.ToTensor()
        ])

with torch.no_grad():    

    method = 0
    if method:
        print("Cosine Similarity")
        cosineSim = nn.CosineSimilarity()
    else:
        print("Euclidean Distance")

    preds = []
    for cnt, file in enumerate(imgList):
        if cnt == 0:
            ref = imgList[cnt]
            img = Image.open(ref).convert("RGB")
            img_tensor = preprocess(img)

            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            ref_output = model(img_tensor)
            preds.append((0, file))
        else:
            img = Image.open(file).convert("RGB")
            img_tensor = preprocess(img)

            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)

            output = model(img_tensor)

            # compute

            if method:
                diff = cosineSim(ref_output, output).detach().cpu().numpy()
            else:
                diff = torch_L2Norm(ref_output, output).detach().cpu().numpy()

            preds.append((diff, file))
            preds.sort(key=lambda x: x[0])

    preds2 = []
    for cnt, file in enumerate(preds):
        if cnt == 0:
            img = cv2.imread(file[1])
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = cv2.rectangle(img, (0, 0), (223, 223), (0, 255, 255), 5)
            preds2.append(str(file))
        else:
            new_img = cv2.imread(file[1])
            new_img = cv2.resize(new_img, (224, 224), interpolation=cv2.INTER_AREA)
            img = cv2.hconcat([img, new_img])
            print(file[1])
            preds2.append(str(file))
    cv2.imwrite(imgPath + "\\" + "vgg16_result" + ".jpg", img)

    filepath = os.path.join(imgPath, 'vgg16_result.txt')
    textfile = open(filepath, 'w')

    for idx, i in enumerate(preds2):
#        if idx != 5:
            textfile.write(str(i)+'\n')
    textfile.close()

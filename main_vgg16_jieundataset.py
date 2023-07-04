from xml.sax import make_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torchsummary import summary
import matplotlib.pyplot as plt
import os
from multiprocessing import freeze_support
from torch.utils.data import random_split
from torchvision import models
from tqdm.notebook import tqdm  #Progress Bar 출력
from tqdm import tqdm


def run():
    torch.multiprocessing.freeze_support()

#이미지 폴더에서 데이터 로드
dataset = ImageFolder(root=r"C:\Users\KETI\Desktop\imagedata",
transform=transforms.Compose([
    transforms.Resize((224,224)),   
    transforms.ToTensor(),
    ]))

data_loader = DataLoader(dataset,
                         batch_size=32,
                         shuffle=True,
                         num_workers=0)
print(dataset.classes)

images, labels = next(iter(data_loader))

labels_map = {v:k for k, v in dataset.class_to_idx.items()}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 4

for i in range(1, cols * rows +1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(torch.permute(img, (1, 2, 0)))
#plt.show()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),               
    transforms.RandomHorizontalFlip(0.5),        
    transforms.ToTensor(),                       
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
])

if __name__ =='__main__':
    freeze_support()
    images, labels = next(iter(data_loader))    


labels_map = {v:k for k, v in dataset.class_to_idx.items()}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 4

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(torch.permute(img, (1, 2, 0)))
#plt.show()

ratio = 0.8

train_size = int(ratio * len(dataset))
test_size = len(dataset) - train_size
print(f'total: {len(dataset)}\ntrain_size: {train_size}\ntest_size: {test_size}')

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, 
                          batch_size=32,
                          shuffle=True, 
                          num_workers=0
                         )
test_loader = DataLoader(test_data, 
                         batch_size=32,
                         shuffle=False, 
                         num_workers=0
                        )

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.vgg16(pretrained=True)

fc = nn.Sequential(
    nn.Linear(7*7*512, 3),
)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

def model_train(model, data_loader, loss_fn, optimizer, device):

    model.train()

    running_loss = 0
    corr = 0

    progress_bar = tqdm(data_loader)

    for img, lbl in progress_bar:

        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        
        output = model(img)

        loss = loss_fn(output, lbl)

        loss.backward()

        optimizer.step()

        _, pred = output.max(dim=1)

        corr += pred.eq(lbl).sum().item()

        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)

    return running_loss / len(data_loader.dataset), acc


def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        corr = 0
        running_loss = 0

        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)

            output = model(img)

            _, pred = output.max(dim=1)

            corr += torch.sum(pred.eq(lbl)).item()

            running_loss += loss_fn(output, lbl).item() * img.size(0)

        acc = corr / len(data_loader.dataset)

        return running_loss / len(data_loader.dataset), acc

num_epochs = 50

min_loss = np.inf

for epoch in range(num_epochs):
    train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)

    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model')
        min_loss = val_loss
        torch.save(model.state_dict(), 'jieunDatasetModel_1118.pth')

    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
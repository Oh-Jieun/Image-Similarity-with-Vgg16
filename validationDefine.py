import os
import cv2
import torch
from PIL import Image
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn


train_set = ImageFolder(
    root="C:/Users/KETI/Desktop/imagedata",
    transform=transforms.Compose([
    transforms.Resize((224,224)),   
    transforms.ToTensor(),
    ])
    )

def torch_L2Norm(pred1, pred2):
    return torch.pow(torch.sum(torch.pow(pred1 - pred2, 2)), 0.5)
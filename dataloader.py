import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# 自定义噪声/模糊增强
class AddSaltPepperNoise(object):
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[self.density/2, self.density/2, 1-self.density])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
            return Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        noise = self.amplitude * np.random.normal(self.mean, self.variance, (h, w, 1))
        noise = np.repeat(noise, c, axis=2)
        img = np.clip(img + noise, 0, 255)
        return Image.fromarray(img.astype('uint8')).convert('RGB')

class Addblur(object):
    def __init__(self, p=0.5, blur="Gaussian"):
        self.p = p
        self.blur = blur
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if self.blur == "Gaussian":
                return img.filter(ImageFilter.GaussianBlur)
            elif self.blur == "normal":
                return img.filter(ImageFilter.BLUR)
            else:
                return img.filter(ImageFilter.BoxBlur)
        return img

# 数据增强流水线
train_transforms = transforms.Compose([
    transforms.RandomOrder([
        transforms.RandomApply([transforms.RandomRotation((-30, 30))], p=0.1),
        transforms.RandomApply([transforms.RandomCrop((150, 200))], p=0.1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.1),
        transforms.RandomApply([Addblur(p=1)], p=0.1),
        transforms.RandomApply([AddSaltPepperNoise(0.05, 1)], p=0.1),
        transforms.RandomApply([AddGaussianNoise(0.5, 0.5, random.uniform(0, 45))], p=0.1),
    ]),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size=32, train_root="./dataset/train", test_root="./dataset/test"):
    """
    本地运行请修改 train_root / test_root 为你的CL-COVIDset数据集路径
    数据集下载地址：Kaggle chiong-continual-learning-of-covid19
    """
    full_train_ds = torchvision.datasets.ImageFolder(train_root, transform=train_transforms)
    test_ds = torchvision.datasets.ImageFolder(test_root, transform=val_test_transforms)

    val_size = int(0.2 * len(full_train_ds))
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

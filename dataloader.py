import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
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
    def __init__(self, mean=0.0, variance=1.0, max_amp=45):
        self.mean = mean
        self.variance = variance
        self.max_amp = max_amp
    def __call__(self, img):
        # 修复：随机幅度放到前向传播内，每次增强随机
        amplitude = random.uniform(0, self.max_amp)
        img = np.array(img)
        h, w, c = img.shape
        noise = amplitude * np.random.normal(self.mean, self.variance, (h, w, 1))
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
                # 修复：GaussianBlur必须传入radius参数
                return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
            elif self.blur == "normal":
                return img.filter(ImageFilter.BLUR)
            else:
                return img.filter(ImageFilter.BoxBlur(radius=2))
        return img

# 数据增强流水线
train_transforms = transforms.Compose([
    transforms.RandomOrder([
        transforms.RandomApply([transforms.RandomRotation((-30, 30))], p=0.1),
        transforms.RandomApply([transforms.RandomCrop((150, 200))], p=0.1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.1),
        transforms.RandomApply([Addblur(p=1)], p=0.1),
        transforms.RandomApply([AddSaltPepperNoise(0.05, 1)], p=0.1),
        transforms.RandomApply([AddGaussianNoise(0.5, 0.5, 45)], p=0.1),
    ]),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 修复：原代码第三个均值笔误0.456 → 0.406，和训练集统一
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size=32, train_root="./dataset/train", test_root="./dataset/test"):
    """
    本地运行请修改 train_root / test_root 为你的CL-COVIDset数据集路径
    数据集下载地址：Kaggle chiong-continual-learning-of-covid19
    """
    # 自动创建dataset文件夹并提示
    os.makedirs("./dataset/train", exist_ok=True)
    os.makedirs("./dataset/test", exist_ok=True)
    # 自动校验数据集文件夹是否存在
    if not os.path.exists(train_root) or len(os.listdir(train_root)) == 0:
        raise FileNotFoundError(f"训练集路径为空：{train_root}\n请下载CL-COVIDset数据集放入dataset/train，内部为11个分类子文件夹")
    if not os.path.exists(test_root) or len(os.listdir(test_root)) == 0:
        raise FileNotFoundError(f"测试集路径为空：{test_root}\n请下载CL-COVIDset数据集放入dataset/test，内部为11个分类子文件夹")
        
    full_train_ds = torchvision.datasets.ImageFolder(train_root, transform=train_transforms)
    test_ds = torchvision.datasets.ImageFolder(test_root, transform=val_test_transforms)
    val_size = int(0.2 * len(full_train_ds))
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
    # 修复Windows多进程报错：自动判断系统设置num_workers
    workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader

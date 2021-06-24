import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
import matplotlib.pyplot as plt
import skimage.transform
import os
import numpy as np
import cv2

class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        # 3x3 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 배치 정규화(batch normalization)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 conv stride=1, padding=1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 배치 정규화(batch normalization)

        self.shortcut = nn.Sequential()  # identity인 경우
        if stride != 1:  # if stride is not 1, if not Identity mapping
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(self.bn2(out))
        out += self.shortcut(identity)  # skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock]],
            number_of_block: List[int],
            num_classes: int = 2
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 64개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, number_of_block[0], stride=1)
        self.layer2 = self._make_layer(block, 128, number_of_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, number_of_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, number_of_block[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, number_of_block: int, stride: int):
        strides = [stride] + [1] * (number_of_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes  # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ResNet18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
# ---------------------------------------------------------------------------------------------------------------------------------------------
    # Userdata input Test

NG_Crack_data_names = []
OK_data_names = []
root_dir = r".\Test"
for (root, dirs, files) in os.walk(root_dir):
    print(root)
    if root == r".\Test\NG_Crack":
        if len(files) > 0:
            for file_name in files:
                NG_Crack_data_names.append(r'.\Test\NG_Crack\\' + file_name)
                print(r'.\Test\NG_Crack\\' + file_name)

    if root == r".\Test\OK":
        if len(files) > 0:
            for file_name in files:
                OK_data_names.append(r'.\Test\OK\\' + file_name)
                print(r'.\Test\OK\\' + file_name)

OK_data_names = OK_data_names
NG_Crack_data_names = NG_Crack_data_names
print((len(OK_data_names) + len(NG_Crack_data_names)))
RGBimg_Test = np.zeros(((len(OK_data_names) + len(NG_Crack_data_names)), 3, 32, 32))  # 624, 3, 512, 512
labelimgOK = np.zeros((len(OK_data_names)))
labelimgNG = np.ones((len(NG_Crack_data_names)))
labelimg_Test = np.zeros((len(OK_data_names) + len(NG_Crack_data_names)))

labelimg_Test[0:(len(OK_data_names))] = labelimgOK
labelimg_Test[(len(labelimgOK)):] = labelimgNG

for File_idx, imgFile in enumerate(OK_data_names):
    # print("File_idx: ", File_idx)
    coloredImg = cv2.imread(imgFile)
    # IMG Downsampling
    coloredImg = cv2.pyrDown(coloredImg)
    for a in range(3):
        coloredImg = cv2.pyrDown(coloredImg)

    # cv2.imshow('gray_image', coloredImg)
    # cv2.waitKey(0)
    # print(coloredImg.shape)
    # exit()

    # IMG split
    b, g, r = cv2.split(coloredImg)
    RGBimg_Test[File_idx, 0, :, :] = r
    RGBimg_Test[File_idx, 1, :, :] = g
    RGBimg_Test[File_idx, 2, :, :] = b

    if File_idx % 50 == 0:
        print('\nCurrent batch:', str(File_idx))

save_File_idx = File_idx + 1
for File_idx, imgFile in enumerate(NG_Crack_data_names):
    num_File_idx = save_File_idx + File_idx
    # print("File_idx: ", File_idx)
    coloredImg = cv2.imread(imgFile)

    # IMG Downsampling
    coloredImg = cv2.pyrDown(coloredImg)
    for a in range(3):
        coloredImg = cv2.pyrDown(coloredImg)

    # IMG split
    b, g, r = cv2.split(coloredImg)
    RGBimg_Test[num_File_idx, 0, :, :] = r
    RGBimg_Test[num_File_idx, 1, :, :] = g
    RGBimg_Test[num_File_idx, 2, :, :] = b

    if num_File_idx % 50 == 0:
        print('\nCurrent batch:', str(num_File_idx))

s = np.arange(labelimg_Test.shape[0])
np.random.shuffle(s)
RGBimg_Test = RGBimg_Test[s, :, :, :]
labelimg_Test = labelimg_Test[s]

RGBimg_Test = torch.from_numpy(RGBimg_Test)
labelimg_Test = torch.from_numpy(labelimg_Test)

print("mk dataset")
batch_size = 50
print(RGBimg_Test.size())
print(int(RGBimg_Test.size()[0] / batch_size))

dataset_test = list(range(int(RGBimg_Test.size()[0] / batch_size)))
for a in range(0, int(RGBimg_Test.size()[0]) - batch_size, batch_size):
    print(int(a / batch_size))
    dataset_test[int(a / batch_size)] = (RGBimg_Test[a:a + batch_size, :, :, :], labelimg_Test[a:a + batch_size])
print(dataset_test[0][0].size())

# ---------------------------------------------------------------------------------------------------------------------------------------------
device = "cuda"
model = ResNet18()
model.load_state_dict(torch.load(r'C:\pyResNet\checkpoint\resnet18.pt'))
model.eval()
model = model.to(device)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


for data in dataset_test:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    print(model)
    print(model(images))
    exit()
    outputs, f = model(images)

    _, predicted = torch.max(outputs, 1)
    break

classes = ('0,1')
params = list(model.parameters())
num = 0
for num in range(64):
    print("ANS :", classes[int(predicted[num])], " REAL :", classes[int(labels[num])], num)

    # print(outputs[0])

    overlay = params[-2][int(predicted[num])].matmul(f[num].reshape(512, 49)).reshape(7, 7).cpu().data.numpy()

    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay)

    imshow(images[num].cpu())
    skimage.transform.resize(overlay, [224, 224])
    plt.imshow(skimage.transform.resize(overlay, [224, 224]), alpha=0.4, cmap='jet')
    plt.show()
    imshow(images[num].cpu())
    plt.show()
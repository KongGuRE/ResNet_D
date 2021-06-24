import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            # 3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        # 512 7 4

        self.avg_pool = nn.AvgPool2d(7)
        # 512 1 1
        self.classifier = nn.Linear(512, 10)
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

    def forward(self, x):
        # print(x.size())
        features = self.conv(x)
        # print(features.size())
        x = self.avg_pool(features)
        # print(avg_pool.size())
        x = x.view(features.size(0), -1)
        # print(flatten.size())
        x = self.classifier(x)
        # x = self.softmax(x)
        return x, features




trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net = torch.load(r'C:\pyResNet\vgg19.pt')
net = net.to(device)

import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


for data in testloader:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs, f = net(images)
    _, predicted = torch.max(outputs, 1)
    break

import skimage.transform

classes = ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
params = list(net.parameters())
num = 0
for num in range(20):
    print("ANS :", classes[int(predicted[num])], " REAL :", classes[int(labels[num])], num)

    # print(outputs[0])

    overlay = params[-2][int(predicted[num])].matmul(f[num].reshape(512, 49)).reshape(7, 7).cpu().data.numpy()

    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay)

    imshow(images[num].cpu())
    skimage.transform.resize(overlay, [224, 224])
    plt.imshow(skimage.transform.resize(overlay, [224, 224]), alpha=0.4, cmap='jet')
    plt.show()
    # imshow(images[num].cpu())
    # plt.show()

# plt.imshow(params[-2][int(predicted[num])].matmul(f[num].reshape(512,49)).reshape(7,7).cpu().data.numpy(), alpha=0.5,cmap='jet');
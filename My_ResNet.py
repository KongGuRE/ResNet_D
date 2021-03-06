from typing import Type, Any, Callable, Union, List, Optional
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import skimage.transform
import math
from tqdm import tqdm, trange
from torch.autograd import Variable


# from torchinfo import summary

# ResNet18 BasicBlock class
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
        features = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)

        out = F.avg_pool2d(features, 32)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, features


# ResNet18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def train(epoch):
    # print('\n[ Train epoch: %d ]' % epoch)
    print('\n')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataset_train):
        # targets = torch.tensor(targets, dtype=torch.long, device=device)
        # targets.requires_grad = True
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs, f = net(inputs)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('batch:', str(batch_idx),
                  'train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)),
                  'train loss:', loss.item(), end='\r', flush=True)

    print('Total benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)
    #
    # state = {
    #     'net': net.state_dict()
    # }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    file_name = 'resnet18.pt'
    torch.save(net.state_dict(), './checkpoint/' + file_name)
    # torch.save(state, './checkpoint/' + file_name)
    # print('net Saved!')


def test(epoch):
    net.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataset_test):
        # targets = torch.tensor(targets, type=torch.long, device=device)
        # targets.requires_grad = True
        # print(inputs.size())
        # print(targets.size())
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs, f = net(inputs)
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    print('Test accuarcy:', 100. * correct / total)
    print('Test average loss:', loss / total)


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    # Userdata input train
    NG_Crack_data_names = []
    OK_data_names = []
    root_dir = r".\Train"
    for (root, dirs, files) in os.walk(root_dir):
        # print(root)
        if root == r".\Train\NG_Crack":
            if len(files) > 0:
                for file_name in files:
                    NG_Crack_data_names.append(r'.\Train\NG_Crack\\' + file_name)
                    # print(r'.\Train\NG_Crack\\' + file_name)

        if root == r".\Train\OK":
            if len(files) > 0:
                for file_name in files:
                    OK_data_names.append(r'.\Train\OK\\' + file_name)
                    # print(r'.\Train\OK\\' + file_name)

    OK_data_names = OK_data_names
    NG_Crack_data_names = NG_Crack_data_names
    RGBimg_train = np.zeros(((len(OK_data_names) + len(NG_Crack_data_names)), 3, 256, 256))  # 624, 3, 512, 512
    labelimgOK = np.zeros((len(OK_data_names)))
    labelimgNG = np.ones((len(NG_Crack_data_names)))
    labelimg_train = np.int64(np.zeros((len(OK_data_names) + len(NG_Crack_data_names))))

    labelimg_train[0:(len(OK_data_names))] = (labelimgOK)
    labelimg_train[(len(labelimgOK)):] = (labelimgNG)

    for File_idx, imgFile in tqdm(enumerate(OK_data_names), desc="ok Train data"):
        # print("File_idx: ", File_idx)
        coloredImg = cv2.imread(imgFile)
        # IMG Downsampling
        coloredImg = cv2.pyrDown(coloredImg)
        # for a in range(3):
        #     coloredImg = cv2.pyrDown(coloredImg)

        # cv2.imshow('gray_image', coloredImg)
        # cv2.waitKey(0)
        # print(coloredImg.shape)
        # exit()

        # IMG split
        b, g, r = cv2.split(coloredImg)
        RGBimg_train[File_idx, 0, :, :] = (r)
        RGBimg_train[File_idx, 1, :, :] = (g)
        RGBimg_train[File_idx, 2, :, :] = (b)

        # if File_idx % 50 == 0:
        #     print('\nTrain data:', str(File_idx),end='\r',flush=True)

    save_File_idx = File_idx + 1
    for File_idx, imgFile in tqdm(enumerate(NG_Crack_data_names), desc="NG Train data"):
        num_File_idx = save_File_idx + File_idx
        # print("File_idx: ", File_idx)
        coloredImg = cv2.imread(imgFile)

        # IMG Downsampling
        coloredImg = cv2.pyrDown(coloredImg)
        # for a in range(3):
        #     coloredImg = cv2.pyrDown(coloredImg)

        # IMG split
        b, g, r = cv2.split(coloredImg)
        RGBimg_train[num_File_idx, 0, :, :] = (r)
        RGBimg_train[num_File_idx, 1, :, :] = (g)
        RGBimg_train[num_File_idx, 2, :, :] = (b)

        # if num_File_idx % 50 == 0:
        #     print('\nCurrent batch:', str(num_File_idx), end='\r',flush=True)

    s = np.arange(labelimg_train.shape[0])
    np.random.shuffle(s)
    RGBimg_train = RGBimg_train[s, :, :, :]
    labelimg_train = labelimg_train[s]

    RGBimg_train = torch.from_numpy(RGBimg_train)
    labelimg_train = torch.from_numpy(labelimg_train)

    batch_size = 10
    # print(RGBimg_train.size())
    # print(int(RGBimg_train.size()[0] / batch_size))

    # dataset_train = list(range(math.ceil(RGBimg_train.size()[0] / batch_size)))
    # for a in range(0, int(RGBimg_train.size()[0]) - batch_size, batch_size):
    # dataset_train[int(a / batch_size)] = (RGBimg_train[a:a + batch_size, :, :, :], labelimg_train[a:a + batch_size])

    dataset_train = list(range(math.ceil(RGBimg_train.size()[0] / batch_size)))
    for a in tqdm(range(0, math.ceil(RGBimg_train.size()[0] / batch_size)), desc="Train data set :"):
        dataset_train[a] = (RGBimg_train[a * batch_size:((a + 1) * batch_size) - 1, :, :, :],
                            labelimg_train[a * batch_size:((a + 1) * batch_size) - 1])
        # print("a*batch_size: ", a * batch_size, "(a+1)*batch_size)-1: ", ((a + 1) * batch_size) - 1)
    # print(dataset_train[0][0].size())

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Userdata input Test
    NG_Crack_data_names = []
    OK_data_names = []
    root_dir = r".\Test"
    for (root, dirs, files) in os.walk(root_dir):
        print(root, end='\r', flush=True)
        if root == r".\Test\NG_Crack":
            if len(files) > 0:
                for file_name in files:
                    NG_Crack_data_names.append(r'.\Test\NG_Crack\\' + file_name)
                    # print(r'.\Test\NG_Crack\\' + file_name)

        if root == r".\Test\OK":
            if len(files) > 0:
                for file_name in files:
                    OK_data_names.append(r'.\Test\OK\\' + file_name)
                    # print(r'.\Test\OK\\' + file_name)

    OK_data_names = OK_data_names
    NG_Crack_data_names = NG_Crack_data_names
    # print((len(OK_data_names) + len(NG_Crack_data_names)))
    RGBimg_Test = np.zeros(((len(OK_data_names) + len(NG_Crack_data_names)), 3, 256, 256))  # 624, 3, 32, 32
    RGBimg_Test_origin = np.zeros(((len(OK_data_names) + len(NG_Crack_data_names)), 3, 512, 512))  # 624, 3, 512, 512
    labelimgOK = np.zeros((len(OK_data_names)))
    labelimgNG = np.ones((len(NG_Crack_data_names)))
    labelimg_Test = np.int64(np.zeros((len(OK_data_names) + len(NG_Crack_data_names))))

    labelimg_Test[0:(len(OK_data_names))] = labelimgOK
    labelimg_Test[(len(labelimgOK)):] = labelimgNG

    for File_idx, imgFile in tqdm(enumerate(OK_data_names), desc="ok Test data : "):
        # print("File_idx: ", File_idx)
        coloredImg_origin = cv2.imread(imgFile)
        # IMG Downsampling
        coloredImg = cv2.pyrDown(coloredImg_origin)
        # for a in range(3):
        #     coloredImg = cv2.pyrDown(coloredImg)

        # cv2.imshow('gray_image', coloredImg)
        # cv2.waitKey(0)
        # print(coloredImg.shape)
        # exit()

        # IMG split
        b, g, r = cv2.split(coloredImg_origin)
        RGBimg_Test_origin[File_idx, 0, :, :] = (r)
        RGBimg_Test_origin[File_idx, 1, :, :] = (g)
        RGBimg_Test_origin[File_idx, 2, :, :] = (b)
        b, g, r = cv2.split(coloredImg)
        RGBimg_Test[File_idx, 0, :, :] = (r)
        RGBimg_Test[File_idx, 1, :, :] = (g)
        RGBimg_Test[File_idx, 2, :, :] = (b)

        # if File_idx % 50 == 0:
        #     print('\nCurrent batch:', str(File_idx),end='\r',flush=True)

    save_File_idx = File_idx + 1
    for File_idx, imgFile in tqdm(enumerate(NG_Crack_data_names), desc="NG Test data"):
        num_File_idx = save_File_idx + File_idx
        # print("File_idx: ", File_idx)
        coloredImg_origin = cv2.imread(imgFile)

        # IMG Downsampling
        coloredImg = cv2.pyrDown(coloredImg_origin)
        # for a in range(3):
        #     coloredImg = cv2.pyrDown(coloredImg)

        # IMG split
        b, g, r = cv2.split(coloredImg_origin)
        RGBimg_Test_origin[num_File_idx, 0, :, :] = (r)
        RGBimg_Test_origin[num_File_idx, 1, :, :] = (g)
        RGBimg_Test_origin[num_File_idx, 2, :, :] = (b)
        b, g, r = cv2.split(coloredImg)
        RGBimg_Test[num_File_idx, 0, :, :] = (r)
        RGBimg_Test[num_File_idx, 1, :, :] = (g)
        RGBimg_Test[num_File_idx, 2, :, :] = (b)

        # if num_File_idx % 50 == 0:
        #     print('\nCurrent b'
        #           ''
        #           ''
        #           'atch:', str(num_File_idx),end='\r',flush=True)

    s = np.arange(labelimg_Test.shape[0])
    np.random.shuffle(s)
    RGBimg_Test = RGBimg_Test[s, :, :, :]
    RGBimg_Test_origin = RGBimg_Test_origin[s, :, :, :]
    labelimg_Test = labelimg_Test[s]

    RGBimg_Test = torch.from_numpy(RGBimg_Test)
    labelimg_Test = torch.from_numpy(labelimg_Test)

    batch_size = 5

    # for a in range(0, int(RGBimg_Test.size()[0]) - batch_size, batch_size):
    #     print(int(a / batch_size))
    #     dataset_test[int(a / batch_size)] = (RGBimg_Test[a:a + batch_size, :, :, :], labelimg_Test[a:a + batch_size])
    # print(dataset_test[0][0].size())

    dataset_test = list(range(math.ceil(RGBimg_Test.size()[0] / batch_size)))
    for a in tqdm(range(0, math.ceil(RGBimg_Test.size()[0] / batch_size)), desc="Test data set :"):
        dataset_test[a] = (RGBimg_Test[a * batch_size:((a + 1) * batch_size) - 1, :, :, :],
                           labelimg_Test[a * batch_size:((a + 1) * batch_size) - 1])

    dataset_test1 = [RGBimg_Test, labelimg_Test]

    # ---------------------------------------------------------------------------------------------------------------------------------------------

    # test
    device = 'cuda'
    net = ResNet18()

    pretrained_dict = torch.load(r'C:\pyResNet\checkpoint\resnet18.pt')
    net.load_state_dict(pretrained_dict)
    net.eval()

    net = net.to(device)
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    torch.device(device)

    learning_rate = 0.001
    # file_name = 'resne
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

    # for epoch in range(0, 200):
    for epoch in trange((200), desc="epoch"):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)


    # ================================================================================

    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # for data in dataset_test1:
    images, labels = dataset_test1
    images = images.cuda()
    labels = labels.cuda()
    outputs, f = net(images)

    _, predicted = torch.max(outputs, 1)
    # break

    classes = ('0', '1')
    params = list(net.parameters())
    num = 0
    RGBimg_Test_origin = torch.tensor(RGBimg_Test_origin)
    RGBimg_Test_origin = RGBimg_Test_origin / 255
    # RGBimg_Test_origin -= RGBimg_Test_origin.min(1, keepdim=True)[0]
    # RGBimg_Test_origin /= RGBimg_Test_origin.max(1, keepdim=True)[0]
    import time

    for num in range(10):
        print("ANS :", classes[int(predicted[num])], " REAL :", classes[int(labels[num])], num)
        # print(outputs[0])
        overlay = params[-2][int(predicted[num])].matmul(f[num].reshape(256, 16)).reshape(4, 4).cpu().data.numpy()
        # overlay = params[-2][int(predicted[num])].matmul(f[num].reshape(256, 64)).reshape(8, 8).cpu().data.numpy()
        # overlay = overlay - np.min(overlay)
        # overlay = overlay / np.max(overlay)
        # RGBimg_Test_origin[num] = F.normalize((RGBimg_Test_origin[num]))
        # RGBimg_Test_origin[num] = RGBimg_Test_origin[num].view(RGBimg_Test_origin[num].size(0),-1)
        # imshow(images[num].cpu())
        imshow((RGBimg_Test_origin[num]).cpu())
        skimage.transform.resize(overlay, [512, 512])
        plt.imshow(skimage.transform.resize(overlay, [512, 512]), alpha=0.3, cmap='jet')
        plt.show()

    # print(list(labels.cpu().numpy()).index(1))

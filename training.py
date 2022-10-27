# @Author: zechenghe
# @Date:   2019-01-20T16:46:24-05:00
# @Last modified by:   zechenghe
# @Last modified time: 2019-02-01T14:01:19-05:00

import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from net import *
from utils import *

def train(DATASET = 'CIFAR10', network = 'CIFAR10CNN', NEpochs = 200, imageWidth = 32,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10,
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3,
        AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", gpu = True):

    print("DATASET: ", DATASET)

    # 暂时不考虑
    if DATASET == 'MNIST':

        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

        tsf = {
            'train': transforms.Compose(
            [
                transforms.ToTensor(),
                Normalize
            ]),

            'test': transforms.Compose(
            [
                transforms.ToTensor(),
                Normalize
            ])
        }

        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True,
                                        download=True, transform = tsf['train'])

        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False,
                                       download=True, transform = tsf['test'])
    # 最终的要的部分
    elif DATASET == 'CIFAR10':

        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        # 针对三个三个数：[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
        # 别人的解答：数据如果分布在(0,1)之间，可能实际的bias，就是神经网络的输入b会比较大
        # 而模型初始化时b=0的，这样会导致神经网络收敛比较慢，经过Normalize后，可以加快模型的收敛速度。
        # 因为对RGB图片而言，数据范围是[0-255]的，需要先经过ToTensor除以255归一化到[0,1]之后
        # 再通过Normalize计算过后，将数据归一化到[-1,1]。

        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        # 这里是使用字典标注了两种类型的变换
        tsf = {
            'train': transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
                # 对图像进行仿射变换，仿射变换是 2 维的线性变换，由 5 种基本操作组成，分别是旋转、平移、缩放、错切和翻转。
            transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
                # 应用了torchvision.transforms.ToTensor，其作用是将数据归一化到[0,1]（是将数据除以255）
                # transforms.ToTensor（）会把HWC会变成C *H *W（拓展：格式为(h,w,c)，像素顺序为RGB）
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ])
        }
        # trainset 的原始形状是(50000, 32, 32, 3)
        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                        download=True, transform = tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=True, transform = tsf['test'])
    # trainset.
    netDict = {
        'LeNet': LeNet,
        'CIFAR10CNN': CIFAR10CNN
    }

    if network in netDict:
        # NChannels 图像通道数 默认为3
        net = netDict[network](NChannels)
    else:
        print("Network not found")
        exit(1)

    print(net)
    print("len(trainset) ", len(trainset))
    print("len(testset) ", len(testset))
    # print testset
    x_train, y_train = trainset.data, trainset.targets
    x_test, y_test = testset.data, testset.targets

    print("x_train.shape ", x_train.shape)
    print("x_test.shape ", x_test.shape)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                      shuffle = True, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    # trainloader
    trainIter = iter(trainloader)
    testIter = iter(testloader)

    criterion = nn.CrossEntropyLoss()
    # 当dim=1时，指的是在维度1上的元素相加等于1。
    # 注：批次所在的维度是第0维
    softmax = nn.Softmax(dim=1)

    if gpu:
        net.cuda()
        criterion.cuda()
        softmax.cuda()

    optimizer = optim.Adam(params = net.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

    NBatch = len(trainset) / BatchSize

    cudnn.benchmark = True
    # 如果将上面的一行注释掉，则代码可以在py2的环境情况下运行
    # 但是会报如下错误，报错之后可以运行
    # THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1544084119927/work/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument

    #
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i in range(int(NBatch)):
            try:
                batchX, batchY = trainIter.next()
            except StopIteration:
                trainIter = iter(trainloader)
                batchX, batchY = trainIter.next()

            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            logits = net.forward(batchX)
            prob = softmax(logits)

            loss = criterion(logits, batchY)
            loss.backward()
            optimizer.step()

            lossTrain += loss.cpu().detach().numpy() / NBatch
            if gpu:
                pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
                groundTruth = batchY.cpu().detach().numpy()
            else:
                pred = np.argmax(prob.detach().numpy(), axis = 1)
                groundTruth = batchY.detach().numpy()

            acc = np.mean(pred == groundTruth)
            accTrain += acc / NBatch

        # 逐步减低学习的速率 默认每20个epoch速率减半
        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)

        print("Epoch: ", epoch, "Loss: ", lossTrain, "Train accuracy: ", accTrain)

        accTest = evalTest(testloader, net, gpu = gpu)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, model_dir + model_name)
    print("Model saved")

    newNet = torch.load(model_dir + model_name)
    newNet.eval()
    accTest = evalTest(testloader, net, gpu = gpu)
    print("Model restore done")


if __name__ == '__main__':
    
    import argparse
    import sys
    import traceback

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNN')
        parser.add_argument('--epochs', type = int, default = 200)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"

        if args.dataset == 'MNIST':

            imageWidth = 28
            imageHeight = 28
            imageSize = imageWidth * imageHeight
            NChannels = 1
            NClasses = 10
            network = 'LeNet'

        elif args.dataset == 'CIFAR10':

            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10
            network = 'CIFAR10CNN'

        else:
            print("No Dataset Found")
            exit(0)

        train(DATASET = args.dataset, network = network, NEpochs = args.epochs, imageWidth = imageWidth,
        imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses,
        BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps,
        AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, gpu = args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

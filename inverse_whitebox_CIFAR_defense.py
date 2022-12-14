# @Author: Zecheng He
# @Date:   2020-04-20

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

from skimage.measure import compare_ssim

#####################
# Useful Hyperparameters:

# CIFAR conv11
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer conv11 --lambda_TV 0.0 --lambda_l2 0.0

# CIFAR ReLU22
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer ReLU22 --lambda_TV 1e1 --lambda_l2 0.0

# CIFAR ReLU32
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer ReLU32 --lambda_TV 1e3 --lambda_l2 0.0

# CIFAR fc1
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-3 --layer fc1 --lambda_TV 1e4 --lambda_l2 0.0
# python inverse_whitebox_CIFAR.py --iters 1000 --learning_rate 5e-2 --layer fc1 --lambda_TV 1e5 --lambda_l2 0.0

# CIFAR fc2
# python inverse_whitebox_CIFAR.py --iters 500 --learning_rate 1e-1 --layer fc2 --lambda_TV 5e2 --lambda_l2 1e2

# CIFAR label
# python inverse_whitebox_CIFAR.py --iters 500 --learning_rate 5e-2 --layer label --lambda_TV 5e-1 --lambda_l2 0.0

# Gaussian and laplace noise
# python inverse_whitebox_CIFAR_defense.py --noise_type Laplace --layer ReLU22
# python inverse_whitebox_CIFAR_defense.py --noise_type Laplace --layer ReLU22 --add_noise_to_input

# Dropout
# python inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer ReLU22
# python inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer ReLU22 --add_noise_to_input

#####################

def eval_DP_defense(args, noise_type, noise_level, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth"):
    if args.dataset == 'CIFAR10':

        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
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

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=False, transform = tsf['test'])

    #print("len(testset) ", len(testset)
    x_test, y_test = testset.data, testset.targets,

    #print("x_test.shape ", x_test.shape
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    testIter = iter(testloader)

    net = torch.load(model_dir + model_name)
    if not args.gpu:
        net = net.cpu()

    net.eval()
    #print("Validate the model accuracy..."

    acc = evalTestSplitModel(testloader, net, net, layer=args.layer, gpu = args.gpu,
            noise_type = noise_type,
            noise_level = noise_level,
            args = args
        )
    return acc

def inverse(DATASET = 'CIFAR10', network = 'CIFAR10CNN', NIters = 500, imageWidth = 32, inverseClass = None,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv22',
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3, lambda_TV = 1e3, lambda_l2 = 1.0,
        AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth",
        save_img_dir = "inverted/CIFAR10/MSE_TV/", saveIter = 10, gpu = True, validation=False,
        noise_type = None, noise_level = 0.0, args=None):

    assert inverseClass < NClasses

    if DATASET == 'CIFAR10':

        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
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

        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                        download=False, transform = tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=False, transform = tsf['test'])

    x_train, y_train = trainset.data, trainset.targets,
    x_test, y_test = testset.data, testset.targets,

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    inverseloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)
    inverseIter = iter(inverseloader)

    net = torch.load(model_dir + model_name)
    if not gpu:
        net = net.cpu()

    net.eval()

    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    deprocessImg = deprocess(targetImg.clone())

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    torchvision.utils.save_image(deprocessImg, save_img_dir + str(inverseClass) + '-ref.png')

    if gpu:
        targetImg = targetImg.cuda()
        softmaxLayer = nn.Softmax().cuda()

    #if hasattr(args, 'add_noise_to_input') and args.add_noise_to_input:
    #    targetImg = apply_noise(targetImg, noise_type, noise_level, gpu=args.gpu, args=args)

    if layer == 'prob':
        if hasattr(args, 'add_noise_to_input') and args.add_noise_to_input:
            targetImg_noised = apply_noise(targetImg, noise_type, noise_level, gpu=args.gpu, args=args)
            reflogits = net.forward(targetImg_noised)
        else:
            reflogits = net.forward(targetImg)
        refFeature = softmaxLayer(reflogits)
    elif layer == 'label':
        refFeature = torch.zeros(1,NClasses)
        refFeature[0, inverseClass] = 1
    else:
        targetLayer = net.layerDict[layer]
        if hasattr(args, 'add_noise_to_input') and args.add_noise_to_input:
            #print("Noise added to input"
            targetImg_noised = apply_noise(targetImg, noise_type, noise_level, gpu=args.gpu, args=args)
            refFeature = net.getLayerOutput(targetImg_noised, targetLayer)
        else:
            refFeature = net.getLayerOutput(targetImg, targetLayer)

    # Apply noise
    if noise_type != None and not (hasattr(args, 'add_noise_to_input') and args.add_noise_to_input):
        refFeature = apply_noise(refFeature, noise_type, noise_level, gpu=args.gpu, args=args)

    if gpu:
        xGen = torch.zeros(targetImg.size(), requires_grad = True, device="cuda:0")
    else:
        xGen = torch.zeros(targetImg.size(), requires_grad = True)

    optimizer = optim.Adam(params = [xGen], lr = learningRate, eps = eps, amsgrad = AMSGrad)

    for i in range(NIters):

        optimizer.zero_grad()
        if layer == 'prob':
            xlogits = net.forward(xGen)
            xFeature = softmaxLayer(xlogits)
            featureLoss = ((xFeature - refFeature)**2).mean()
        elif layer == 'label':
            xlogits = net.forward(xGen)
            xFeature = softmaxLayer(xlogits)
            featureLoss = - torch.log(xFeature[0, inverseClass])
        else:
            xFeature = net.getLayerOutput(xGen, targetLayer)
            featureLoss = ((xFeature - refFeature)**2).mean()

        TVLoss = TV(xGen)
        normLoss = l2loss(xGen)

        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss #- 1.0 * conv1Loss

        totalLoss.backward(retain_graph=True)
        optimizer.step()

        #print("Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy()

    # save the final result
    imgGen = xGen.clone()
    imgGen = deprocess(imgGen)
    torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

    ref_img = deprocessImg.detach().cpu().numpy().squeeze()
    inv_img = imgGen.detach().cpu().numpy().squeeze()

    #print("ref_img.shape", ref_img.shape, "inv_img.shape", inv_img.shape
    #print("ref_img ", ref_img.min(), ref_img.max()
    #print("inv_img ", inv_img.min(), inv_img.max()

    psnr = get_PSNR(ref_img, inv_img, peak=1.0)

    #print("ref_img.shape", ref_img.shape)
    #print("inv_img.shape", inv_img.shape)
    ssim = compare_ssim(
        X=np.moveaxis(ref_img, 0, -1),
        Y=np.moveaxis(inv_img, 0, -1),
        data_range = inv_img.max() - inv_img.min(),
        multichannel=True)

    #print("targetImg l1 Stat:"
    #getL1Stat(net, targetImg)
    #print("xGen l1 Stat:"
    #getL1Stat(net, xGen)
    #print("Done"

    return psnr, ssim


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNN')
        parser.add_argument('--iters', type = int, default = 500)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--lambda_TV', type = float, default = 1.0)
        parser.add_argument('--lambda_l2', type = float, default = 1.0)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)
        parser.add_argument('--layer', type = str, default = 'conv22')
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = None)

        parser.add_argument('--noise_type', type = str, default = None)
        parser.add_argument('--noise_level', type = float, default = None)

        parser.add_argument('--add_noise_to_input', dest='add_noise_to_input', action='store_true')

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)

        parser.add_argument('--novalidation', dest='validation', action='store_false')
        parser.set_defaults(validation=True)

        parser.add_argument('--noise_iters', type = int, default = 500)
        parser.add_argument('--noise_eps', type = float, default = 1e-3)
        parser.add_argument('--noise_AMSGrad', type = bool, default = True)
        parser.add_argument('--noise_learning_rate', type = float, default = 1e-1)
        parser.add_argument('--noise_lambda_sourcelayer', type = float, default = 1e-1)
        parser.add_argument('--noise_decrease_LR', type = int, default = 20)
        parser.add_argument('--noise_targetLayer', type = str, default = 'fc3')

        args = parser.parse_args()


        args.noise_sourceLayer = args.layer
        args.model_dir = "checkpoints/" + args.dataset + '/'
        args.model_name = "ckpt.pth"

        if args.dataset == 'CIFAR10':

            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10

        else:
            print("No Dataset Found")
            exit()

        noise_type = args.noise_type
        noise_hist = []
        acc_hist = []
        psnr_hist = []
        ssim_hist = []

        if 'noise_gen' in args.noise_type:
            default_nl = np.concatenate((np.arange(0, 110, 10), np.arange(100, 1100, 100)), axis=0)
        elif 'dropout' in args.noise_type:
            default_nl = np.arange(0, 1, 0.1)
        else:
            default_nl = np.concatenate((np.arange(0, 1, 0.1), np.arange(1.0, 5.5, 0.5)), axis=0)

        noise_range = [args.noise_level] if args.noise_level != None else default_nl

        for noise_level in noise_range:
            noise_hist.append(noise_level)

            if args.add_noise_to_input:
                save_img_dir = "inverted_whitebox/" + args.dataset + '/' + args.layer + '/' + 'noised_add_to_input/' + noise_type + '/' + str(round(noise_level,1)) + '/'
            else:
                save_img_dir = "inverted_whitebox/" + args.dataset + '/' + args.layer + '/' + 'noised/' + noise_type + '/' + str(round(noise_level,1)) + '/'

            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)

            acc = eval_DP_defense(args, noise_type, noise_level)
            acc_hist.append(acc)

            psnr_sum = 0.0
            ssim_sum = 0.0
            for c in range(NClasses):
                psnr, ssim = inverse(DATASET = args.dataset, network = args.network, NIters = args.iters, imageWidth = imageWidth, inverseClass = c,
                imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer,
                BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps, lambda_TV = args.lambda_TV, lambda_l2 = args.lambda_l2,
                AMSGrad = args.AMSGrad, model_dir = args.model_dir, model_name = args.model_name, save_img_dir = save_img_dir, saveIter = args.save_iter,
                gpu = args.gpu, validation=args.validation, noise_type = noise_type, noise_level = noise_level, args = args)

                psnr_sum += psnr / NClasses
                ssim_sum += ssim / NClasses

            psnr_hist.append(psnr_sum)
            ssim_hist.append(ssim_sum)
            print("Noise_type:", noise_type, " Add to input:", args.add_noise_to_input, " Noise_level:", round(noise_level,2), " Acc:", round(acc,4), " PSNR:", round(psnr_sum,4), " SSIM:", round(ssim_sum,4))

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

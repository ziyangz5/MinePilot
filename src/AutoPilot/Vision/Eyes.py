import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import init
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL
from PIL import Image
from torchvision.models.segmentation import segmentation
class Cfg:
    lr = 0.0002
    workers = 2
    batchSize = 4
    imageSize = 64
    n_epoch = 10
    beta1 = 0.5
    seed = 0
    cuda = True
    start_epo = 0
    pretrain = True
    nd_kpts = 6
cfg = Cfg()
device = torch.device("cuda:0" if cfg.cuda else "cpu")

class Eyes:
    def __init__(self):
        self.net =  segmentation.fcn_resnet50(num_classes=5).to(device)
        self.net.eval()
        if cfg.pretrain:
            self.net.load_state_dict(torch.load("../Vision/10_epoch.wts"))
    def get_result(self,img):
        return torch.argmax(self.net(img.to(device))['out'], dim=1)
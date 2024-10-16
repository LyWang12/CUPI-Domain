import os
import random
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils.getdata import Cus_Dataset
from utils.util import get_mnist_data, get_usps_data, get_svhn_data, get_mnist_m_data, get_augment_data

lr = 0.0001
nepoch = 30
workers = 10
batch_size = 32
device = torch.device("cuda:0")
NUM_CLASSES = 10
TRAIN_SIZE = 5000
VAL_SIZE = 1000

SOURCE = 'mnist_m'   # mnist, usps, svhn, mnist_m
PSEUDO_SOURCE = 'mnist_m'   # SOURCE=PSEUDO_SOURCE

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


class CUPI(nn.Module):
    def __init__(self):
        super(CUPI, self).__init__()
    def forward(self, x):
        if self.training:
            batch_size, C = x.size()[0] // 3, x.size()[1]
            style_mean, style_std = calc_ins_mean_std(x[:batch_size])
            c_mean, c_std = calc_ins_mean_std(x[batch_size:2 * batch_size])
            conv_mean = nn.Conv2d(C, C, 1, bias=False).to(device)
            conv_std = nn.Conv2d(C, C, 1, bias=False).to(device)
            mean = torch.sigmoid(conv_mean(style_mean))
            std = torch.sigmoid(conv_std(style_std))
            x_a = (x[batch_size:2 * batch_size] - c_mean) / (c_std + 1e-6) * std + mean
            x = torch.cat((x[:batch_size], x_a, x[2 * batch_size:]), 0)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.layer_n = len(features)
        self.bottleneck = nn.Linear(2048, 256)
        self.classifier1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.chan_ex = CUPI()

    def forward(self, x, y=None, z=None, action=None):
        if action == 'val':
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.bottleneck(x)
            x = self.classifier1(x)
            return x

        elif action == 'memory':
            input = torch.cat((x, y), 0)
            input1 = self.chan_ex(self.features[:3](input))
            input2 = self.chan_ex(self.features[3:6](input1))
            input3 = self.chan_ex(self.features[6:11](input2))
            input4 = self.chan_ex(self.features[11:16](input3))
            input5 = self.chan_ex(self.features[16:](input4))
            fx1, fy1 = input1.chunk(2, dim=0)
            fx2, fy2 = input2.chunk(2, dim=0)
            fx3, fy3 = input3.chunk(2, dim=0)
            fx4, fy4 = input4.chunk(2, dim=0)
            fx5, fy5 = input5.chunk(2, dim=0)

            input5 = input5.view(input5.size(0), -1)
            input5 = self.bottleneck(input5)
            px, py = input5.chunk(2, dim=0)

            x = self.classifier1(px)
            y = self.classifier1(py)
            return fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5, px, py, x, y

        elif action == 'train':
            input = torch.cat((x, y, z), 0)
            input1 = self.chan_ex(self.features[:3](input))
            input2 = self.chan_ex(self.features[3:6](input1))
            input3 = self.chan_ex(self.features[6:11](input2))
            input4 = self.chan_ex(self.features[11:16](input3))
            input5 = self.chan_ex(self.features[16:](input4))
            fx1, fy1, fz1 = input1.chunk(3, dim=0)
            fx2, fy2, fz2 = input2.chunk(3, dim=0)
            fx3, fy3, fz3 = input3.chunk(3, dim=0)
            fx4, fy4, fz4 = input4.chunk(3, dim=0)
            fx5, fy5, fz5 = input5.chunk(3, dim=0)

            input5 = input5.view(input5.size(0), -1)
            input5 = self.bottleneck(input5)
            px, py, pz = input5.chunk(3, dim=0)

            x = self.classifier1(px)
            y = self.classifier1(py)
            z = self.classifier1(pz)

            return fx1, fy1, fz1, fx2, fy2, fz2, fx3, fy3, fz3, fx4, fy4, fz4, fx5, fy5, fz5, px, py, pz, x, y, z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i in range(len(cfg)):
        v = cfg[i]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)

    model.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict = False)
    return model


def validate_class(val_loader, model, epoch, num_class=10):
    model.eval()
    correct = 0
    total = 0
    c_class = [0 for i in range(num_class)]
    t_class = [0 for i in range(num_class)]
    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        y_pred = model(x=images, action='val')
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        true_label = torch.argmax(labels, axis=1)
        correct += (predicted == true_label).sum().item()
        for j in range(predicted.shape[0]):
            t_class[true_label[j]] += 1
            if predicted[j] == true_label[j]:
                c_class[true_label[j]] += 1
        
    acc = 100.0 * correct / total
    print('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=acc))
    model.train()
    return acc


def Memory(dataloader, model):
    memory_dict = {
        "memory_CUPI_features": torch.zeros(TRAIN_SIZE, 256).to(device),
        "memory_CUPI_labels": torch.zeros(TRAIN_SIZE).long().to(device),
        "memory_source_mean1": torch.zeros(TRAIN_SIZE, 64).to(device),
        "memory_source_std1": torch.zeros(TRAIN_SIZE, 64).to(device),
        "memory_source_mean2": torch.zeros(TRAIN_SIZE, 128).to(device),
        "memory_source_std2": torch.zeros(TRAIN_SIZE, 128).to(device),
        "memory_source_mean3": torch.zeros(TRAIN_SIZE, 256).to(device),
        "memory_source_std3": torch.zeros(TRAIN_SIZE, 256).to(device),
        "memory_source_mean4": torch.zeros(TRAIN_SIZE, 512).to(device),
        "memory_source_std4": torch.zeros(TRAIN_SIZE, 512).to(device),
        "memory_source_mean5": torch.zeros(TRAIN_SIZE, 512).to(device),
        "memory_source_std5": torch.zeros(TRAIN_SIZE, 512).to(device),
        "memory_source_labels": torch.zeros(TRAIN_SIZE).long().to(device)
    }
    with torch.no_grad():
        for i, (imgs, labels, imgc, labelc, _, _, idx) in enumerate(dataloader):
            imgs, labels, imgc, labelc = imgs.to(device), labels.to(device), imgc.to(device), labelc.to(device)
            imgs, labels, imgc, labelc =  imgs.float(), labels.float(), imgc.float(), labelc.float()
            fs1, fc1, fs2, fc2, fs3, fc3, fs4, fc4, fs5, fc5, ps, pc, ys, yc = model(x=imgs, y=imgc, action='memory')

            # source style
            mean1, std1 = calc_ins_mean_std(fs1)
            memory_dict["memory_source_mean1"][idx] = mean1.squeeze().detach()
            memory_dict["memory_source_std1"][idx] = std1.squeeze().detach()
            mean2, std2 = calc_ins_mean_std(fs2)
            memory_dict["memory_source_mean2"][idx] = mean2.squeeze().detach()
            memory_dict["memory_source_std2"][idx] = std2.squeeze().detach()
            mean3, std3 = calc_ins_mean_std(fs3)
            memory_dict["memory_source_mean3"][idx] = mean3.squeeze().detach()
            memory_dict["memory_source_std3"][idx] = std3.squeeze().detach()
            mean4, std4 = calc_ins_mean_std(fs4)
            memory_dict["memory_source_mean4"][idx] = mean4.squeeze().detach()
            memory_dict["memory_source_std4"][idx] = std4.squeeze().detach()
            mean5, std5 = calc_ins_mean_std(fs5)
            memory_dict["memory_source_mean5"][idx] = mean5.squeeze().detach()
            memory_dict["memory_source_std5"][idx] = std5.squeeze().detach()
            # CUPI prediction
            memory_dict["memory_CUPI_features"][idx] = pc
            memory_dict["memory_CUPI_labels"][idx] = torch.LongTensor([one_label.tolist().index(1) for one_label in labelc]).to(device)
            memory_dict["memory_source_labels"][idx] = torch.LongTensor([one_label.tolist().index(1) for one_label in labels]).to(device)

    print('Memory initial!')
    return memory_dict


def Discrimination_loss(memory_dict, ps, pc, pt, index, label1, label3):
    memory_dict["memory_CUPI_features"][index] = pc.detach()
    mean_CUPI = CalculateMean(memory_dict["memory_CUPI_features"], memory_dict["memory_CUPI_labels"])  # 10,256
    loss_d_sc = F.mse_loss(mean_CUPI[torch.LongTensor([one_label.tolist().index(1) for one_label in label1]).cuda()], ps)
    loss_d_tc = F.mse_loss(mean_CUPI[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], pt)
    return loss_d_sc, loss_d_tc, memory_dict


def Style_loss(memory_dict, fs1, ft1, fs2, ft2, fs3, ft3, fs4, ft4, fs5, ft5, index, label3):
    fs_mean1, fs_std1 = calc_ins_mean_std(fs1)
    memory_dict["memory_source_mean1"][index] = fs_mean1.squeeze().detach()
    memory_dict["memory_source_std1"][index] = fs_std1.squeeze().detach()
    mean_source1 = CalculateMean(memory_dict["memory_source_mean1"], memory_dict["memory_source_labels"])
    std_source1 = CalculateMean(memory_dict["memory_source_std1"], memory_dict["memory_source_labels"])
    ft_mean1, ft_std1 = calc_ins_mean_std(ft1)
    loss_s_mean_t_1 = F.mse_loss(mean_source1[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean1.squeeze())
    loss_s_std_t_1 = F.mse_loss(std_source1[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std1.squeeze())

    fs_mean2, fs_std2 = calc_ins_mean_std(fs2)
    memory_dict["memory_source_mean2"][index] = fs_mean2.squeeze().detach()
    memory_dict["memory_source_std2"][index] = fs_std2.squeeze().detach()
    mean_source2 = CalculateMean(memory_dict["memory_source_mean2"], memory_dict["memory_source_labels"])
    std_source2 = CalculateMean(memory_dict["memory_source_std2"], memory_dict["memory_source_labels"])
    ft_mean2, ft_std2 = calc_ins_mean_std(ft2)
    loss_s_mean_t_2 = F.mse_loss(mean_source2[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean2.squeeze())
    loss_s_std_t_2 = F.mse_loss(std_source2[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std2.squeeze())

    fs_mean3, fs_std3 = calc_ins_mean_std(fs3)
    memory_dict["memory_source_mean3"][index] = fs_mean3.squeeze().detach()
    memory_dict["memory_source_std3"][index] = fs_std3.squeeze().detach()
    mean_source3 = CalculateMean(memory_dict["memory_source_mean3"], memory_dict["memory_source_labels"])
    std_source3 = CalculateMean(memory_dict["memory_source_std3"], memory_dict["memory_source_labels"])
    ft_mean3, ft_std3 = calc_ins_mean_std(ft3)
    loss_s_mean_t_3 = F.mse_loss(mean_source3[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean3.squeeze())
    loss_s_std_t_3 = F.mse_loss(std_source3[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std3.squeeze())

    fs_mean4, fs_std4 = calc_ins_mean_std(fs4)
    memory_dict["memory_source_mean4"][index] = fs_mean4.squeeze().detach()
    memory_dict["memory_source_std4"][index] = fs_std4.squeeze().detach()
    mean_source4 = CalculateMean(memory_dict["memory_source_mean4"], memory_dict["memory_source_labels"])
    std_source4 = CalculateMean(memory_dict["memory_source_std4"], memory_dict["memory_source_labels"])
    ft_mean4, ft_std4 = calc_ins_mean_std(ft4)
    loss_s_mean_t_4 = F.mse_loss(mean_source4[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean4.squeeze())
    loss_s_std_t_4 = F.mse_loss(std_source4[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std4.squeeze())

    fs_mean5, fs_std5 = calc_ins_mean_std(fs5)
    memory_dict["memory_source_mean5"][index] = fs_mean5.squeeze().detach()
    memory_dict["memory_source_std5"][index] = fs_std5.squeeze().detach()
    mean_source5 = CalculateMean(memory_dict["memory_source_mean5"], memory_dict["memory_source_labels"])
    std_source5 = CalculateMean(memory_dict["memory_source_std5"], memory_dict["memory_source_labels"])
    ft_mean5, ft_std5 = calc_ins_mean_std(ft5)
    loss_s_mean_t_5 = F.mse_loss(mean_source5[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean5.squeeze())
    loss_s_std_t_5 = F.mse_loss(std_source5[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std5.squeeze())

    loss_s_mean_t = loss_s_mean_t_1 + loss_s_mean_t_2 + loss_s_mean_t_3 + loss_s_mean_t_4 + loss_s_mean_t_5
    loss_s_std_t = loss_s_std_t_1 + loss_s_std_t_2 + loss_s_std_t_3 + loss_s_std_t_4 + loss_s_std_t_5

    return loss_s_mean_t, loss_s_std_t, memory_dict


def CalculateMean(features, labels):
    N = features.size(0)
    C = NUM_CLASSES
    A = features.size(1)

    avg_CxA = torch.zeros(C, A).to(device)
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).to(device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(NUM_CLASSES):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()


def Data_loader(source, target=None):
    if source == 'mnist':
        dataset_s = get_mnist_data()
    elif source == 'usps':
        dataset_s = get_usps_data()
    elif source == 'svhn':
        dataset_s = get_svhn_data()
    elif source == 'mnist_m':
        dataset_s = get_mnist_m_data()

    dataset_t = get_augment_data(PSEUDO_SOURCE)
    # Load test datasets
    dataset1 = get_mnist_data()
    dataset2 = get_usps_data()
    dataset3 = get_svhn_data()
    dataset4 = get_mnist_m_data()
    return dataset_s, dataset_t, dataset1, dataset2, dataset3, dataset4


def train():
    # Load datasets
    dataset_s, dataset_t, dataset1, dataset2, dataset3, dataset4 = Data_loader(source=SOURCE)

    print('original data loaded...')
    PATH = '/data1/WLY/code/PAMI2/git/model_save/tf/digit/' + SOURCE + '_free' + '.pth'

    datafile = Cus_Dataset(mode='train',
                           dataset_1=dataset_s, begin_ind1=0, size1=TRAIN_SIZE,
                           dataset_2=dataset_t, begin_ind2=0, size2=TRAIN_SIZE,
                           dataset_3=dataset_t, begin_ind3=0, size3=TRAIN_SIZE)
    datafile_val1 = Cus_Dataset(mode='val', dataset_1=dataset1, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)
    datafile_val2 = Cus_Dataset(mode='val', dataset_1=dataset2, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)
    datafile_val3 = Cus_Dataset(mode='val', dataset_1=dataset3, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)
    datafile_val4 = Cus_Dataset(mode='val', dataset_1=dataset4, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)

    valloader1 = DataLoader(datafile_val1, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader2 = DataLoader(datafile_val2, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader3 = DataLoader(datafile_val3, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader4 = DataLoader(datafile_val4, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    model = vgg11(pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    # memory
    model.eval()
    memory_dict = Memory(dataloader=dataloader, model=model)
    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch: 0.999 ** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion_KL = torch.nn.KLDivLoss()

    for epoch in range(nepoch):
        for i, (img1, label1, img2, label2, img3, label3, index) in enumerate(dataloader):
            img1, label1, img2, label2, img3, label3 = img1.to(device), label1.to(device), img2.to(device), label2.to(device), img3.to(device), label3.to(device)
            img1, label1, img2, label2, img3, label3 = img1.float(), label1.float(), img2.float(), label2.float(), img3.float(), label3.float()
            fs1, fc1, ft1, fs2, fc2, ft2, fs3, fc3, ft3, fs4, fc4, ft4, fs5, fc5, ft5, ps, pc, pt, ys, yc, yt = model(x=img1, y=img2, z=img3, action='train')

            alpha = 0.1
            kl = 0.1 * (epoch + 1 / nepoch) ** 0.9

            ys = F.log_softmax(ys, dim=1)
            loss1 = criterion_KL(ys, label1)

            yc = F.log_softmax(yc, dim=1)
            loss2 = criterion_KL(yc, label2)
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)

            yt = F.log_softmax(yt, dim=1)
            loss3 = criterion_KL(yt, label3)
            loss3 = loss3 * alpha
            if loss3 > 1:
                loss3 = torch.clamp(loss3, 0, 1)

            # discrimination loss
            loss_d_sc, loss_d_tc, memory_dict = Discrimination_loss(memory_dict, ps, pc, pt, index, label1, label3)
            loss_d_sc = loss_d_sc * kl
            if loss_d_sc > 1:
                loss_d_sc = torch.clamp(loss_d_sc, 0, 1)

            loss_d_tc = loss_d_tc * kl
            if loss_d_tc > 1:
                loss_d_tc = torch.clamp(loss_d_tc, 0, 1)

            # style loss
            loss_s_mean_t, loss_s_std_t, memory_dict = Style_loss(memory_dict, fs1, ft1, fs2, ft2, fs3, ft3, fs4, ft4, fs5, ft5, index, label3)
            loss_s_mean_t = loss_s_mean_t * kl
            if loss_s_mean_t > 1:
                loss_s_mean_t = torch.clamp(loss_s_mean_t, 0, 1)

            loss_s_std_t = loss_s_std_t * kl
            if loss_s_std_t > 1:
                loss_s_std_t = torch.clamp(loss_s_std_t, 0, 1)

            loss = loss1 - loss2 - loss3 - loss_d_sc + loss_d_tc - loss_s_mean_t - loss_s_std_t
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        acc1 = validate_class(valloader1, model, epoch, num_class=NUM_CLASSES)
        acc2 = validate_class(valloader2, model, epoch, num_class=NUM_CLASSES)
        acc3 = validate_class(valloader3, model, epoch, num_class=NUM_CLASSES)
        acc4 = validate_class(valloader4, model, epoch, num_class=NUM_CLASSES)
        f = open(PATH.split('.')[0] + '_acc.txt', "a+")
        f.write("epoch = {:02d}, acc1 = {:.3f}, acc2 = {:.3f}, acc3 = {:.3f}, acc4 = {:.3f}".format(epoch, acc1, acc2, acc3, acc4) + '\n')
        f.close()


if __name__ == "__main__":
    train()

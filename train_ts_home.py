import os
import random
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils.getdata import Cus_Dataset
from utils.util import get_home_data

from utils.network import ImageClassifier
from transformer import swin_tiny_patch4_window7_224, calc_ins_mean_std

lr = 0.0001
nepoch = 30
workers = 10
batch_size = 8
device = torch.device("cuda")

NUM_CLASSES = 65
TRAIN_SIZE = 2000
VAL_SIZE = 400
SOURCE = 'Art'   # Art, Clipart, Product, RealWorld
TARGET = 'Clipart'   # Art, Clipart, Product, RealWorld

def validate_class(val_loader, model, epoch, num_class=10):
    model.eval()

    correct = 0
    total = 0
    c_class = [0 for i in range(num_class)]
    t_class = [0 for i in range(num_class)]
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

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
        "memory_source_mean1": torch.zeros(TRAIN_SIZE, 192).to(device),
        "memory_source_std1": torch.zeros(TRAIN_SIZE, 192).to(device),
        "memory_source_mean2": torch.zeros(TRAIN_SIZE, 384).to(device),
        "memory_source_std2": torch.zeros(TRAIN_SIZE, 384).to(device),
        "memory_source_mean3": torch.zeros(TRAIN_SIZE, 768).to(device),
        "memory_source_std3": torch.zeros(TRAIN_SIZE, 768).to(device),
        "memory_source_mean4": torch.zeros(TRAIN_SIZE, 768).to(device),
        "memory_source_std4": torch.zeros(TRAIN_SIZE, 768).to(device),
        "memory_source_labels": torch.zeros(TRAIN_SIZE).long().to(device)
    }
    with torch.no_grad():
        for i, (imgs, labels, imgc, labelc, _, _, idx) in enumerate(dataloader):
            imgs, labels, imgc, labelc = imgs.to(device), labels.to(device), imgc.to(device), labelc.to(device)
            imgs, labels, imgc, labelc =  imgs.float(), labels.float(), imgc.float(), labelc.float()
            fs1, fc1, fs2, fc2, fs3, fc3, fs4, fc4, ps, pc, ys, yc = model(x=imgs, y=imgc, action='memory')  # B,65  B,256  B,49,768

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


def Style_loss(memory_dict, fs1, ft1, fs2, ft2, fs3, ft3, fs4, ft4, index, label3):
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

    loss_s_mean_t = loss_s_mean_t_1 + loss_s_mean_t_2 + loss_s_mean_t_3 + loss_s_mean_t_4
    loss_s_std_t = loss_s_std_t_1 + loss_s_std_t_2 + loss_s_std_t_3 + loss_s_std_t_4

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
    dataset1 = get_home_data(source)
    dataset2 = get_home_data(target)
    dataset3 = get_home_data(target)
    return dataset1, dataset2, dataset3


def train():
    # Load datasets
    dataset1, dataset2, dataset3 = Data_loader(source=SOURCE, target=TARGET)

    print('original data loaded...')
    PATH = '/data1/WLY/code/PAMI2/git/model_save/ts/home/' + SOURCE + '_to_' + TARGET + '.pth'

    datafile = Cus_Dataset(mode='train',
                           dataset_1=dataset1, begin_ind1=0, size1=TRAIN_SIZE,
                           dataset_2=dataset2, begin_ind2=0, size2=TRAIN_SIZE,
                           dataset_3=dataset3, begin_ind3=0, size3=TRAIN_SIZE)
    datafile_val1 = Cus_Dataset(mode='val', dataset_1=dataset1, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)
    datafile_val2 = Cus_Dataset(mode='val', dataset_1=dataset2, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)
    datafile_val3 = Cus_Dataset(mode='val', dataset_1=dataset3, begin_ind1=TRAIN_SIZE, size1=VAL_SIZE)

    valloader1 = DataLoader(datafile_val1, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader2 = DataLoader(datafile_val2, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader3 = DataLoader(datafile_val3, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    backbone = swin_tiny_patch4_window7_224()
    weights_dict = torch.load('swin_tiny_patch4_window7_224.pth')["model"]
    # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
    print(backbone.load_state_dict(weights_dict, strict=False))
    model = ImageClassifier(backbone, NUM_CLASSES).to(device)

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
            fs1, fc1, ft1, fs2, fc2, ft2, fs3, fc3, ft3, fs4, fc4, ft4, ps, pc, pt, ys, yc, yt = model(x=img1, y=img2, z=img3, action='train')

            alpha = 0.1
            kl = 0.1 * (epoch + 1 / nepoch) ** 0.9

            ys = F.log_softmax(ys, dim=1)
            loss1 = 2*criterion_KL(ys, label1)

            yc = F.log_softmax(yc, dim=1)
            loss2 = criterion_KL(yc, label2)
            if loss2 > loss1:
                loss2 = torch.clamp(loss2, 0, 1)
            loss2 = loss2 * alpha

            yt = F.log_softmax(yt, dim=1)
            loss3 = criterion_KL(yt, label3)
            if loss3 > loss1:
                loss3 = torch.clamp(loss3, 0, 1)
            loss3 = loss3 * alpha

            # discrimination loss
            loss_d_sc, loss_d_tc, memory_dict = Discrimination_loss(memory_dict, ps, pc, pt, index, label1, label3)
            if loss_d_sc > loss1:
                loss_d_sc = torch.clamp(loss_d_sc, 0, loss1.item())
            loss_d_sc = loss_d_sc * kl

            if loss_d_tc > loss1:
                loss_d_tc = torch.clamp(loss_d_tc, 0, loss1.item())
            loss_d_tc = loss_d_tc * kl

            # style loss
            loss_s_mean_t, loss_s_std_t, memory_dict = Style_loss(memory_dict, fs1, ft1, fs2, ft2, fs3, ft3, fs4, ft4, index, label3)
            if loss_s_mean_t > loss1:
                loss_s_mean_t = torch.clamp(loss_s_mean_t, 0, loss1.item())
            loss_s_mean_t = kl * loss_s_mean_t

            loss_s_std_t = loss_s_std_t * kl
            if loss_s_std_t > loss1:
                loss_s_std_t = torch.clamp(loss_s_std_t, 0, loss1.item())
            loss_s_std_t = kl * loss_s_std_t

            loss = loss1 - loss2 - loss3 - loss_d_sc + loss_d_tc - loss_s_mean_t - loss_s_std_t
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # test
        acc1 = validate_class(valloader1, model, epoch, num_class=NUM_CLASSES)
        acc2 = validate_class(valloader2, model, epoch, num_class=NUM_CLASSES)
        acc3 = validate_class(valloader3, model, epoch, num_class=NUM_CLASSES)
        f = open(PATH.split('.')[0] + '_acc.txt', "a+")
        f.write("epoch = {:02d}, acc1 = {:.3f}, acc2 = {:.3f}, acc1 = {:.3f}".format(epoch, acc1, acc2, acc3) + '\n')
        f.close()


if __name__ == "__main__":
    train()


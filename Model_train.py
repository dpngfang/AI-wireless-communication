#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder
import os
import torch.nn as nn


def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = 32
epochs = 200
learning_rate = 0.001
num_workers = 0
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 512

# Model construction
model = AutoEncoder(feedback_bits)
if use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()
model.encoder.load_state_dict(torch.load('./Modelsave/encoder.pth.tar')['state_dict'])
model.decoder.load_state_dict(torch.load('./Modelsave/decoder.pth.tar')['state_dict'])

import scipy.io as scio
criterion = NMSELoss().cuda()
criterion_mse = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
data_load_address = 'D:\AI无线通信\Model_pytorch_2021\data'
mat = scio.loadmat(data_load_address+'\Htrain.mat')
x_train = mat['H_train']  # shape=8000*126*128*2

x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
print(np.shape(x_train))
mat = scio.loadmat(data_load_address+'\Htest.mat')
x_test = mat['H_test']  # shape=2000*126*128*2

x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
print(np.shape(x_test))
# Data loading


# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


best_loss = 100
early_stop = 0
for epoch in range(epochs):
    # model training
    model.train()
    for i, input in enumerate(train_loader):
        # adjust learning rate
        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.1
        input = input.cuda()
        # compute output
        output = model(input)
        loss_nmse = criterion(output, input)
        loss_mse = criterion_mse(output, input)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss_nmse.backward()
        optimizer.step()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss NMSE {loss:.6f}\t'
                  'Loss MSE {loss_mse:.6f}\t'.format(
                epoch, i, len(train_loader), loss=loss_nmse.item(), loss_mse=loss_mse.item()))
    # model evaluating
    model.eval()
    total_loss = 0
    total_loss1 = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            total_loss += criterion(output, input).item() * input.size(0)
            total_loss1 += criterion_mse(output, input).item() * input.size(0)
        average_loss = total_loss / len(test_dataset)
        average_loss1 = total_loss1 / len(test_dataset)
        print('\tTest\t'
              'Loss {loss:.6f}\t'
              'Loss1 {loss1:.6f}\t'.format(loss=average_loss, loss1=average_loss1))

        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            print("Model saved")
            best_loss = average_loss
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= 10:
            break



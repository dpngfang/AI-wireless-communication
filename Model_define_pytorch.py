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
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict



# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True, padding_mode='circular')
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True, padding_mode='circular')

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=(inputs.size(2), inputs.size(3)))
        x = self.down(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.repeat(1, 1, inputs.size(2), inputs.size(3))
        return inputs * x

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.att = SEBlock(c2, c2 // 2)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.att(self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1)))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        self.att = SEBlock(c2, c2 // 2)

    def forward(self, x):
        return self.att(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)))



class WLBlock(nn.Module):
    def __init__(self, paths, in_c, k=16, n=[1, 1], e=[1.0, 1.0], quantization=True):

        super(WLBlock, self).__init__()
        self.paths = paths
        self.n = n
        self.e = e
        self.k = k
        self.in_c = in_c
        for i in range(self.paths):
            self.__setattr__(str(i), nn.Sequential(OrderedDict([
                ("Conv0", Conv(self.in_c, self.k, 3)),
                ("BCSP_1", BottleneckCSP(self.k, self.k, n=self.n[i], e=self.e[i])),
                ("C3_1", C3(self.k, self.k, n=self.n[i], e=self.n[i])),
                ("Conv1", Conv(self.k, self.k, 3)),
            ])))
        self.conv1 = conv3x3(self.k * self.paths, self.k)

    def forward(self, x):
        outs = []
        for i in range(self.paths):
            _ = self.__getattr__(str(i))(x)
            outs.append(_)
        out = torch.cat(tuple(outs), dim=1)
        out = self.conv1(out)
        out = out + x if self.in_c == self.k else out
        return out



class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.conv1 = conv3x3(2, 16)
        self.conv2 = conv3x3(16, 2)
        self.bottle = BottleneckCSP(2, 16, n=2, e=0.5)
        self.conv_to_2 = conv3x3(16, 2)
        self.fc = nn.Linear(32256, int(feedback_bits // self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.bottle(out + x))
        out = F.relu(self.conv_to_2(out))
        out = out.view(-1, 126*128*2)
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)
        return out


class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.multiConvs = nn.ModuleList()
        self.fc = nn.Linear(int(feedback_bits // self.B), 32256)
        self.out_cov = conv3x3(2, 2)
        self.sig = nn.Sigmoid()

        for _ in range(3):
            self.multiConvs.append(nn.Sequential(
                conv3x3(2, 16),
                nn.ReLU(),
                WLBlock(1, 16, 16, [1, 2, 3], [0.5, 1, 1.5]),
                nn.ReLU(),
                conv3x3(16, 2),
                nn.ReLU()))

    def forward(self, x):
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits // self.B))
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 126, 128)
        for i in range(3):
            residual = out
            out = self.multiConvs[i](out)
            out = residual + out
        out = self.out_cov(out)
        out = self.sig(out)
        return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]

    # def __getitem__(self, index):
    #     y = self.matdata[index]
    #     if self.phase == 'train' and random.random() < 0.5:
    #         y = y[:, ::-1, :].copy()
    #     if self.phase == 'train' and random.random() < 0.5:
    #         y = y[:, :, ::-1].copy()
    #     if self.phase == 'train' and random.random() < 0.5:
    #         y = 1 - self.matdata[index]  # 数据中存在类似正交的关系
    #     if self.phase == 'train' and random.random() < 0.5:
    #         _ = y
    #         _[0, :, :] = y[1, :, :]
    #         _[1, :, :] = y[0, :, :]
    #         y = _  # 不同时刻数据实虚存在部分相等的情况
    #     if self.phase == 'train' and random.random() < 0.5:
    #         index_ = random.randint(0, self.matdata.shape[0] // 3000 - 1) * 3000 + index % 3000
    #         p = random.random()
    #         rows = max(int(126 * p), 1)
    #         _rows = [i for i in range(126)]
    #         random.shuffle(_rows)
    #         _rows = _rows[:rows]
    #         if random.random() < 0.7:
    #             y[:, _rows, :] = self.matdata[index_][:, _rows, :]  # 不同采样点按行合并，保持采样点独有特性，减轻模型对24那个维度的依赖
    #         else:
    #             y = (1 - p * 0.2) * y + (p * 0.2) * self.matdata[index_]  # 增加数值扰动，保持采样点独有特性
    #     return y




# if __name__ == '__main__':
#     from torchsummary import summary
#     feedback_bits = 512
#     model = AutoEncoder(feedback_bits).cuda()
#     # model(torch.Tensor(np.random.rand(2, 2, 126, 128)).cuda())
#     # summary(model, input_size=(2, 126, 128), batch_size=-1)
#     summary(model.encoder, input_size=(2, 126, 128), batch_size=-1)
#     summary(model.decoder, input_size=(feedback_bits, ), batch_size=-1)


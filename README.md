# AI-wireless-communication
### 1. 必要的代码级样例展示
```
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
```
```
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
```
```
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```
```
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
```
```
class BottleneckCSP(nn.Module):
    
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
```
```
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
```

```
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
```

### 2. 算法思路			    		

本算法采用端到端系统，使输入的信道样本数据经过编码器编码，再经解码器解码，将最后得到的结果与最初的输入进行对比，根据公式算出归一化均方误差 NMSE，并根据这个误差去不断优化模型直至误差最小。


### 3. 亮点解读

本次代码所提交的模型，在baseline的基础上，主要加入了：
①、SEBlock模块，该模块包括全局池化层、降维的全连接层、RELU层、升维的全连接层、SIGMOD层，这样增加了更多的非线性处理过程，可以拟合通道之间复杂的相关性，并且在最后加上repeat层，即h * w * c 和 1 * 1 * c 的 feature map，这样可以得到不同通道重要性不一样的feature map。
②、BottleneckCSP模块，该模块包括一个标准卷积层、若干个Bottleneck层（标准瓶颈层）、BatchNorm2d层（可以加速收敛速度、增强模型稳定性）、LeakyReLU层。
③、C3模块（CSP Bottleneck with 3 convolutions），该模块与BottleneckCSP类似。
加入这三个模块后的模型优化效果显著，CSI压缩重建前后的损失明显降低


### 4. 建模算力与环境

a. 项目运行环境

i. 项目所需的工具包/框架
    numpy==1.19.5
    torch==1.7.1
    h5py==2.10.0

ii. 项目运行的资源环境

    单卡1050显卡

b. 项目运行办法

i. 项目的文件结构
- Model_define_pytorch.py: 模型定义
- Model_evaluation_decoder.py: 译码器
- Model_evaluation_encoder.py: 编码器
- Model_train.py: 模型训练、结果生成
- data/: 初赛数据
    - Htrain.mat: 初赛数据集
    - Htest.mat/: 初赛测试集
- Modelsave/: 存储最终结果

ii. 项目的运行步骤
    运行 Model_train.py
    在Modelsave文件夹中含有生成的结果

运行结果的位置


    ../Modelsave/decoder.pth.tar文件
    ../Modelsave/encoder.pth.tar文件
   

### 5. 使用的预训练模型相关论文及模型下载链接
无
### 6. 其他补充资料（如有）
无






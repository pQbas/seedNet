import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM 

class Residual(nn.Module):
  def __init__(self, num_channels, use_1x1conv=False, strides=1, use_cbam=False):
    super().__init__()
    self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,stride=strides)
    self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
    if use_1x1conv:
      self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
    else:
      self.conv3 = None
    self.bn1 = nn.LazyBatchNorm2d()
    self.bn2 = nn.LazyBatchNorm2d()

    if use_cbam:
        self.cbam = CBAM(num_channels, 2, 7)
    else:
        self.cbam = None

  def forward(self, X):
    Y = F.relu(self.bn1(self.conv1(X)))
    Y = self.bn2(self.conv2(Y))

    if self.cbam:
        print("Si se agrego!!!")
        ''' Here goes CBAM '''
        Y = self.cbam(Y)

    if self.conv3:
      X = self.conv3(X)
    Y += X
    return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, arch, lr=0.1, num_classes=10, use_cbam=True):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1())
        self.use_cbam = use_cbam

        for b in arch:
            num_residuals = b[0]
            num_channels = b[1]            
            for i in range(num_residuals):
                self.net.append(Residual(num_channels, use_1x1conv=(i==0), strides=1, use_cbam=self.use_cbam))

        self.net.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.LazyLinear(num_classes))
        )

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10, use_cbam=False):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),lr, num_classes, use_cbam)




model = ResNet18()
x = torch.rand([1,3,24,24])
y = model(x)

print(model)
print(x.shape)
print(y.shape)

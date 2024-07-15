import torch.nn as nn
import torch.nn.functional as F

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    #elif classname.find('BatchNorm') != -1:
    #    m.weight.data.normal_(1, 0.02)
    #    m.bias.data.fill_(0)

class DecodeBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        resolution,
        is_last,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.is_last = is_last
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        #self.bn2d = nn.BatchNorm2d(out_channels)
        self.bn2d = nn.BatchNorm2d(in_channels, affine=False)
        self.activate =  nn.ReLU(True)
        self.activatel = nn.Tanh()

    def forward(self,x):
        #x = self.conv(x)
        if not self.is_last:
            x = self.bn2d(x)
            x = self.conv(x)
            x = self.activate(x)
        else:
            x = self.conv(x)
            x = self.activatel(x)
        return x

class ResUpBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """
    def __init__(self, in_features, out_features, is_last=False, kernel_size=3, padding=1):
        super(ResUpBlock2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False)
        #self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
        #                       padding=padding)
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(out_features, affine=True)
        self.is_last = is_last
        self.activate = nn.ReLU()
        self.activatel = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.norm1(out)
        out = self.activate(out)

        #out = self.conv2(out)
        #out = self.norm1(out)
        out += residual
        #out = self.activate(out)
        #out = F.interpolate(out, scale_factor=2)
        out = self.conv(out)
        if not self.is_last:
            #out = self.norm2(out)
            out = self.activate(out)
        else:
            out = self.activatel(out)

        return out

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        #self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class EncodeBlock(nn.Module):
    def __init__(self,
        in_dims,
        out_dims,
        is_last,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.is_last = is_last
        self.activate =  nn.ReLU(True)
        self.fc = nn.Linear(in_dims, out_dims)
    def forward(self,x):
        x = self.fc(x)
        if not self.is_last:
            x = self.activate(x)
        return x
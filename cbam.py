import torch.nn as nn
import torch
import torch.nn.functional as F


class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chann_att = self.channel_attention(f)
        fp = chann_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size = kernel_size, padding = int((kernel_size-1)/2))
    
    def forward(self, x):
        max_pool = torch.max(x,1)[0].unsqueeze(1)
        avg_pool = torch.mean(x,1).unsqueeze(1)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att



class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.sharedMLP = nn.Sequential(
                nn.Linear(self.n_channels_in, self.middle_layer_size),
                nn.ReLU(),
                nn.Linear(self.middle_layer_size, self.n_channels_in)
                )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.sharedMLP(avg_pool)
        max_pool_bck = self.sharedMLP(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out



if __name__ == '__main__':

    cbam = CBAM(64, 2, 7)

    f = torch.rand([1,64,24,24])
    fpp = cbam(f)

    #chAtt = ChannelAttention(64, 2)
    #fp = f * chAtt(f)
    #spAtt = SpatialAttention(7)
    #fpp = spAtt(fp)*fp
    print(fpp.shape)
    


    


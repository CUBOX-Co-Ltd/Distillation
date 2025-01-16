import torch
import torch.nn as nn

class BasicConv2dProjection(nn.Module):
    def __init__(self, in_ch, feature_dim_s, feature_dim_t):
        self.projector = nn.Conv2d(in_ch, max(feature_dim_s, feature_dim_t), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.projector(x)

class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class OFAProjcetor(nn.Module):
    def __init__(self, in_ch, feature_dim_s, feature_dim_t):
        self.projector = nn.nn.Sequential(
            SepConv(),
            nn.Conv2d(in_ch, max(feature_dim_s, feature_dim_t), kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes) 
        )

    def forward(self, x):
        return self.projector(x)


projectors = {
    'basic_conv': BasicConv2dProjection,
    'ofa': OFAProjcetor
}
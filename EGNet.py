import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from thop import profile
from TBB_.mix_transformer import mit_b2
from torch.nn.functional import interpolate


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class EnhancementModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
        super(EnhancementModule, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_planes))
        else:
            self.register_parameter('bias', None)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.attention(x)
        batch_size, _, height, width = x.size()

        weight = self.weight.view(self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)


        aggregate_weight = weight.view(self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = self.bias.repeat(batch_size)
        else:
            aggregate_bias = None

        output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                          groups=1)

        return output * attention


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class FusionModule(nn.Module):
    def __init__(self, channels):
        super(FusionModule, self).__init__()

        rgb_channels = channels
        depth_channels = channels
        out_channels = channels

        self.rgb_conv = nn.Conv2d(rgb_channels, out_channels, kernel_size=3, padding=1)
        self.depth_conv = nn.Conv2d(depth_channels, out_channels, kernel_size=3, padding=1)

        self.channel_attention = ChannelAttention(out_channels * 2)
        self.spatial_attention = SpatialAttention()

        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        # 多尺度特征提取
        self.multiscale_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=4, dilation=4),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=8, dilation=8)
        ])

        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_conv(rgb)
        depth_feat = self.depth_conv(depth)

        concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)

        channel_weight = self.channel_attention(concat_feat)
        concat_feat = concat_feat * channel_weight

        spatial_weight = self.spatial_attention(concat_feat)
        concat_feat = concat_feat * spatial_weight

        fused_feat = self.fusion_conv(concat_feat)

        multiscale_feats = [conv(fused_feat) for conv in self.multiscale_convs]
        multiscale_feat = torch.cat(multiscale_feats, dim=1)

        output = self.output_conv(multiscale_feat)

        return output


class Decoder(nn.Module):
    def __init__(self, in1, in2, in3, in4):
        super(Decoder, self).__init__()
        self.bcon4 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in4, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3 = BasicConv2d(in3, in4, kernel_size=3, stride=1, padding=1)
        self.bcon2 = BasicConv2d(in2, in3, kernel_size=3, stride=1, padding=1)
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in2, kernel_size=1, stride=1, padding=0)

        self.bcon4_3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4 * 2, in3, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3_2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3 * 2, in2, kernel_size=3, stride=1, padding=1)
        )
        self.bcon2_1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, f):
        f3 = self.bcon4(f[3])
        f2 = self.bcon3(f[2])
        f1 = self.bcon2(f[1])
        f0 = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f3, f2), 1))
        d32 = self.bcon3_2(torch.cat((d43, f1), 1))
        d21 = self.bcon2_1(torch.cat((d32, f0), 1))
        out = d21

        return out


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        self.backbone_rgb = mit_b2()

        self.fm1 = FusionModule(64)
        self.fm2 = FusionModule(128)
        self.fm3 = FusionModule(320)
        self.fm4 = FusionModule(512)

        self.decoder = Decoder(64, 128, 320, 512)

        self.dynamic_conv = nn.ModuleList([
            EnhancementModule(64, 64, 3, padding=1),
            EnhancementModule(128, 128, 3, padding=1),
            EnhancementModule(320, 320, 3, padding=1),
            EnhancementModule(512, 512, 3, padding=1)
        ])

        self.last1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 5, 3, 1, 1)
        )

    def forward(self, x,y):
        rgb = self.backbone_rgb(x)
        depth = self.backbone_rgb(y)

        f1 = self.fm1(rgb[0], depth[0])
        f2 = self.fm2(rgb[1], depth[1])
        f3 = self.fm3(rgb[2], depth[2])
        f4 = self.fm4(rgb[3], depth[3])

        sall = []
        sall.append(f1)
        sall.append(f2)
        sall.append(f3)
        sall.append(f4)

        merges = []
        for i in range(4):
            fused = self.dynamic_conv[i](sall[i])
            merges.append(fused)

        out = self.decoder(merges)

        return self.last1(out)

#
if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = DNet().cuda()

    right = torch.randn(1, 3, 480, 640).cuda()
    left = torch.randn(1, 3, 480, 640).cuda()


    out = model(right, left)
    for i in out:
        print(i.shape)

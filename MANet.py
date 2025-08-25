import torch
from torch import nn
import torch.nn.functional as F
from Deformer.DFormer import DFormer_Base
from toolbox.model.norm import ChannelAttention, SpatialAttention, BasicConv2d, Up

from toolbox.model.segmodels.SpatialAware import DTFP


class Attention(nn.Module):
    def __init__(self, in_x, reduction=16):
        super(Attention, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_x, in_x, 3, padding=1),
            nn.BatchNorm2d(in_x),
            nn.LeakyReLU())
        in_x = in_x * 2
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_x)

    def forward(self, r, d):
        r = self.down_conv(r)
        d = self.down_conv(d)
        mul_rd = r * d
        sa = self.sa(mul_rd)
        r_f = r * sa
        r_f = r + r_f
        r_ca = self.ca(r_f)
        r_out = r * r_ca
        return r_out


class DFFM(nn.Module):#DualFeatureFusionModule
    def __init__(self, in_x):
        super(DFFM, self).__init__()
        self.A = Attention(in_x)
        in_x = in_x * 2
        self.c = BasicConv2d(in_x, in_x // 2, 1)

        self.conv_c = nn.Conv2d(in_x // 2, 1, 1)
        self.conv_c_d = nn.Conv2d(in_x // 2, 1, 1)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_x, in_x, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_x), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_x, in_x, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_x), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_x, in_x, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_x), nn.LeakyReLU(0.1, inplace=True))

        self.b_1 = BasicConv2d(in_x * 3, in_x, kernel_size=3, padding=1)  # CBR33
        self.conv_res = BasicConv2d(in_x, in_x, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.dftp = DTFP(in_x, in_x)

    def aspp(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))  # fu
        out = self.relu(buffer_1 + self.conv_res(x))  # CAi
        return out

    def forward(self, r, d):

        r_out = self.A(r, d)  # 64,96
        d_out = self.A(d, r)
        RD = torch.cat([ r_out,  d_out], dim=1)  # 512,13

        RD = self.dftp(RD)

        out1 = self.aspp(RD)  # 32 48

        out1 = self.c(out1)
        return out1


class fusion(nn.Module):
    def __init__(self, inc):
        super(fusion, self).__init__()
        self.sof = nn.Softmax(dim=1)
        self.DFFM = DFFM(inc)
        self.dropout = nn.Dropout(.1)

    def forward(self, r, d):
        out = self.DFFM(r, d)
        out = self.dropout(out)

        return out


class BM(nn.Module):
    def __init__(self, inc):
        super(BM, self).__init__()
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc // 2, 1)
        self.fun = fusion(inc // 2)
        self.con3 = BasicConv2d(inc // 2, inc, 1)

    def forward(self, r, d, s):
        r = self.con2(r)  # 128,52
        d = self.con2(d)  # 64,104
        s = self.con1(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s


class BM2(nn.Module):
    def __init__(self, inc=144):
        super(BM2, self).__init__()
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc // 2, 1)
        self.fun = fusion(inc // 2)
        self.con3 = BasicConv2d(inc // 2, inc, 1)

    def forward(self, r, d, s):
        r = self.con2(r)
        d = self.con2(d)
        s = self.con1(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s


class BM3(nn.Module):
    def __init__(self, inc=144):
        super(BM3, self).__init__()
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc // 2, 1)
        self.fun = fusion(inc // 2)
        self.con3 = BasicConv2d(inc // 2, inc, 1)

    def forward(self, r, d, s):
        r = self.con2(r)
        d = self.con2(d)
        s = self.con1(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s


####################################################################################
class Decoder(nn.Module):
    def __init__(self, inc=128):
        super(Decoder, self).__init__()

        self.fuse1 = fusion(64)
        self.fuse2 = BM(128)
        self.fuse3 = BM2(256)
        self.fuse4 = BM3(512)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(768, inc, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True),
                                    nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True))

        self.Conv432 = nn.Sequential(nn.Conv2d(256, inc, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(inc),
                                     nn.ReLU(True),
                                     nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inc),
                                     nn.ReLU(True))

        self.Conv4321 = nn.Sequential(nn.Conv2d(192, inc, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(inc),
                                      nn.ReLU(True),
                                      nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(inc),
                                      nn.ReLU(True))

        self.sal_pred = nn.Sequential(nn.Conv2d(inc, 128, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(True),
                                      nn.Conv2d(128, 9, 3, 1, 1, bias=False))
        self.linear2 = nn.Conv2d(512, 9, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(128, 9, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(128, 9, kernel_size=3, stride=1, padding=1)



    def forward(self, x, rgb_list, depth_list):
        R1, R2, R3, R4 = rgb_list[0], rgb_list[1], rgb_list[2], rgb_list[3]
        D1, D2, D3, D4 = depth_list[0], depth_list[1], depth_list[2], depth_list[3]

        RD1 = self.fuse1(R1, D1)
        RD2 = self.fuse2(R2, D2, RD1)
        RD3 = self.fuse3(R3, D3, RD2)
        RD4 = self.fuse4(R4, D4, RD3)
        RD43 = self.up2(RD4)
        RD43 = torch.cat((RD43, RD3), dim=1)
        RD43 = self.Conv43(RD43)

        RD432 = self.up2(RD43)
        RD432 = torch.cat((RD432, RD2), dim=1)
        RD432 = self.Conv432(RD432)

        RD4321 = self.up2(RD432)
        RD4321 = torch.cat((RD4321, RD1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        sal_map = self.sal_pred(RD4321)
        out1 = self.up4(sal_map)
        out2 = F.interpolate(self.linear2(RD4), size=x.size()[2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.linear3(RD43), size=x.size()[2:], mode='bilinear', align_corners=False)
        out4 = F.interpolate(self.linear4(RD432), size=x.size()[2:], mode='bilinear', align_corners=False)
        global count
        count += 1
        return out1, out2, out3, out4, RD1, RD2, RD3, RD4, RD43, RD432, RD4321

global count
count=1

######################################
class Dfo_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.dformer = DFormer_Base(pretrained=True)
        self.decoder = Decoder(128)  #
        self.sig = nn.Sigmoid()

    def forward(self, input_rgb):
        input_depth=input_rgb
        out = self.dformer(input_rgb, input_depth)
        out1, out2, out3, out4, RD1, RD2, RD3, RD4, RD43, RD432, RD4321 = self.decoder(input_rgb, out, out)
        return out1, out2, out3, out4


if __name__ == '__main__':
    img = torch.randn(1, 3,  480, 640).cuda()
    depth = torch.randn(1, 3,  480, 640).cuda()

    model = Dfo_T().cuda()
    out = model(img)
    for i in range(len(out)):
        print(out[i].shape)

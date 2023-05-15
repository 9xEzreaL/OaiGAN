# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : networks.py
# @Author: Jeffrey, Jehovah
# @Date  : 19-9

import torch
import torch.nn as nn
#from config.SAND_pix_opt import TrainOptions
from models.genre.base_network import BaseNetwork
from models.genre.blocks.unet_block import UNetDown, SPADEUp


class SPADEUNet(BaseNetwork):
    def __init__(self, opt, in_channels=3, out_channels=3):
        super(SPADEUNet, self).__init__()
        self.opt = opt
        self.layer = int(self.opt.netG.split('_')[1])
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = UNetDown(64, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        if self.layer >=6:
            self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        if self.layer >=7:
            self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)

        self.up0 = SPADEUp(self.opt, 512, 512, 0.5)
        if self.layer>=7:
            self.up0p = SPADEUp(self.opt, 1024, 512, 0.5)
        if self.layer>=6:
            self.up0pp = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up1 = SPADEUp(self.opt, 1024, 512)
        self.up2 = SPADEUp(self.opt, 768, 256)
        self.up3 = SPADEUp(self.opt, 384, 64)

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, parsing):
        d1 = self.down1(x) # 256->128
        d2 = self.down2(d1) # 128->64
        d3 = self.down3(d2) # 64->32
        d4 = self.down4(d3) # 32->16
        d5 = self.down5(d4) # 16->8
        if self.layer ==5:
            u1 = self.up0(d5, parsing)
            u2 = self.up1(u1, parsing, d4) # 8->16 (512+512->512)
            u3 = self.up2(u2, parsing, d3) # 16->32 (512+256((u:512 d:256))->512)
            u4 = self.up3(u3, parsing, d2) # 32->64
        if self.layer ==6:
            d6 = self.down6(d5) # 8->4
            u1 = self.up0(d6, parsing)
            u2 = self.up0pp(u1, parsing, d5)
            u3 = self.up1(u2, parsing, d4) # 8->16 (512+512->512)
            u4 = self.up2(u3, parsing, d3) # 16->32 (512+256((u:512 d:256))->512)
            u4 = self.up3(u4, parsing, d2) # 32->64
        if self.layer == 7:
            d6 = self.down6(d5)
            d7 = self.down7(d6) # 4->2
            u1 = self.up0(d7, parsing)
            u2 = self.up0p(u1, parsing, d6) # 2->4 (512+512->512)
            u3 = self.up0pp(u2, parsing, d5) # 4->8 (512+512->512
            u4 = self.up1(u3, parsing, d4)  # 8->16 (512+512->512)
            u4 = self.up2(u4, parsing, d3)  # 16->32 (512+256((u:512 d:256))->512)
            u4 = self.up3(u4, parsing, d2)  # 32->64

        u5 = torch.cat([u4, d1], dim=1) # 128->256
        u6 = self.final(u5)
        return u6


class SPADEUNet_YPar(BaseNetwork):
    def __init__(self, opt, img_channel, par_channel, out_channels=3):
        super(SPADEUNet_YPar, self).__init__()
        self.opt = opt
        self.down_rgb = nn.Conv2d(img_channel, 64, 4, 2, 1)
        self.down_par = nn.Conv2d(par_channel, 64, 4, 2, 1)
        self.down2 = UNetDown(128, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down8 = UNetDown(512, 512, normalize=False)
        self.up0 = SPADEUp(self.opt, 512, 512, 0.5, first=True)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final_rgb = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        self.final_par = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, par_channel, 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, x_parsing, y_parsing):
        d1_rgb = self.down_rgb(x)
        d1_par = self.down_par(x_parsing)
        d1 = torch.cat([d1_rgb, d1_par], dim=1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u0 = self.up0(d8, y_parsing)
        u1 = self.up1(u0, y_parsing, d7)
        u2 = self.up2(u1, y_parsing, d6)
        u3 = self.up3(u2, y_parsing, d5)
        u4 = self.up4(u3, y_parsing, d4)
        u5 = self.up5(u4, y_parsing, d3)
        u6, gamma_beta = self.up6(u5, y_parsing, d2, gamma_mode="final")

        u7_rgb = torch.cat([u6, d1_rgb], dim=1)
        u7_par = torch.cat([u6, d1_par], dim=1)
        u8_rgb = self.final_rgb(u7_rgb)
        u8_par = self.final_par(u7_par)

        return u8_rgb, u8_par
        # return u8_rgb


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()  # add_help=False)
    # Env
    parser.add_argument('--netG', type=str, default='SPADE_7')
    parser.add_argument('--parsing_nc', type=int, default=5,
                        help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--norm_G', type=str, default='spectralspadebatch3x3',
                        help='instance normalization or batch normalization')
    parser.add_argument('--spade_mode', type=str, default="res2", choices=('org', 'res', 'concat', 'res2'),
                        help='type of spade shortcut connection : |org|res|concat|res2|')
    parser.add_argument('--use_en_feature', action='store_true', help='cat encoder feature to parsing')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()

    #opt = TrainOptions().parse()
    #opt.input_size = 256
    #opt.spade_mode = 'res'
    #opt.norm_G = 'spectraloutterbatch3x3'
    # opt = SPADEGenerator.modify_commandline_options(opt)
    style = torch.randn([2, 3, 256, 256])
    x = torch.randn([2, 5, 256, 256])

    m = SPADEUNet(opt=opt, in_channels=3, out_channels=3)
    m(style, x)

    # model = OutterUNet(opt, in_channels=3, out_channels=1)
    #y_identity, gamma_beta = model.forward(style)
    #hat_y, _ = model.forward(x, use_basic=False, gamma=gamma_beta[0], beta=gamma_beta[1])
    #print(hat_y.size())

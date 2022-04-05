from __future__ import print_function
import argparse, json
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.data_utils import norm_01
from skimage import io
import torchvision.transforms as transforms
import tifffile as tiff

load_dotenv('.env')


def overlap_red(x0, y0):
    y = 1 * y0
    x = 1 * x0
    x = x - x.min()
    x = x / x.max()

    c = 0
    x[1, y == c] = 0.1 * x[1, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 2
    x[0, y == c] = 0.1 * x[0, y == c]
    x[1, y == c] = 0.1 * x[1, y == c]

    c = 4
    x[0, y == c] = 0.1 * x[0, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    # c = 2
    # x[0, y == c] = 0.1 * x[0, y == c]
    # x[1, y == c] = 0.1 * x[1, y == c]
    #
    # c = 4
    # x[0, y == c] = 0.1 * x[0, y == c]
    # x[1, y == c] = 0.1 * x[1, y == c]
    return x


class Pix2PixModel:
    def __init__(self, args):
        self.args = args
        self.net_g = None
        self.dir_checkpoints = os.environ.get('LOGS')
        if self.args.cartonly:
            from dataloader.data_multi import MultiData as Dataset
            self.test_set = Dataset(root=os.environ.get('DATASET') + args.testset,
                                    path=args.direction,
                                    opt=args, mode='test')
        else:
            from dataloader.data_multi import MultiData as Dataset
            self.test_set = Dataset(root=os.environ.get('DATASET') + args.testset,
                                    path=args.direction,
                                    opt=args, mode='test')

        os.makedirs(os.path.join("outputs/results", args.dataset, args.prj), exist_ok=True)

        # self.seg_cartilage = torch.load('submodels/model_seg.pth')
        self.seg_cartilage = torch.load('submodels/model_seg256.pth')#model_seg_ZIB.pth')
        # self.seg_cartilage = torch.load('submodels/SegScale.pth')
        #self.seg_cartilage = torch.load('submodels/model_seg_ZIB_res18_256.pth')
        # self.seg_bone = torch.load('submodels/model_seg_ZIB.pth')
        #self.cartilage = torch.load('submodels/femur_tibia_fc_tc.pth').cuda()

        # self.netg_t2d.eval()

        # self.seg_bone.eval()

        #self.magic256 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()
        #self.magic286 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()
        # self.magic286 = torch.load('/media/ExtHDD01/checkpointsold/FlyZ_WpOp286Mask/netG_model_epoch_10.pth').cuda()

        self.device = torch.device("cuda:0")

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset, self.args.prj, 'checkpoints') + \
               ('/' + self.args.netg + '_model_epoch_{}.pth').format(epoch)
        print(model_path)
        net = torch.load(model_path).to(self.device)
        if eval:
            net.eval()
        else:
            net.train()
        self.net_g = net

    def get_one_output(self, i, xy, alpha=None):
        # inputs
        x = self.test_set.__getitem__(i)
        if len(x[0])==4:
            oriX = x[0][0].unsqueeze(0).to(self.device) #(b,c,256,256)
            oriY = x[0][1].unsqueeze(0).to(self.device)
            maskX = x[0][2].unsqueeze(0).to(self.device)
            maskY = x[0][3].unsqueeze(0).to(self.device)
            oriX = torch.cat((oriX,maskY),1)
        elif len(x[0])==2:
            oriX = x[0][0].unsqueeze(0).to(self.device)
            oriY = x[0][1].unsqueeze(0).to(self.device)
        else:
            print('Error in data direction number, must be 2 or 4')
        if xy == 'x':
            in_img = oriX
            out_img = oriY
        elif xy == 'y':
            in_img = oriY
            out_img = oriX
        alpha = alpha / 100

        try:
            output0, output1 = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())
            output = output0 #- output0
        except:
            try:
                output = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())[0]
            except:
                try:
                    output = self.net_g(in_img, alpha * torch.ones(1, 1).cuda())[0]
                except:
                    output = self.net_g(in_img)[0] # (1,5,256,256)3

        if in_img.shape[1] in [5]:
            in_img = squ_channel(in_img)
            out_img = squ_channel(out_img)
            output = squ_channel(output)
        elif in_img.shape[1] in [8]:
            in_img = in_img[:,:3,::]
        else:
            in_img = in_img
        in_img = in_img.detach().cpu()[0,:3,::]
        out_img = out_img.detach().cpu()[0, ::]
        output = output[0, ::].detach().cpu()
        return in_img, out_img, output

    # not used
    def get_t2d(self, ori):
        t2d = self.netg_t2d(ori.cuda().unsqueeze(0))[0][0, ::].detach().cpu()
        return t2d

    def get_seg(self, ori):
        # bone = self.seg_bone(ori.cuda().unsqueeze(0))
        # bone = torch.argmax(bone, 1)[0,::].detach().cpu()
        self.seg_cartilage.eval()
        cartilage = self.seg_cartilage(ori.cuda().unsqueeze(0))
        cartilage = torch.argmax(cartilage, 1)[0,::].detach().cpu()

        #seg[seg == 3] = 0
        #seg[seg == 4] = 0

        # seg = 1 * bone

        #cartilage = self.cartilage(norm_01(ori).cuda().unsqueeze(0))
        #cartilage = torch.argmax(cartilage, 1)[0,::].detach().cpu()

        return cartilage

    def get_magic(self, ori):
        magic = self.magic286(ori.cuda().unsqueeze(0))
        magic = magic[0][0, 0, ::].detach().cpu()
        return magic

    def get_all_seg(self, input):
        if self.args.t2d:
            input = list(map(lambda k: list(map(lambda v: self.get_t2d(v), k)), input))  # (3, 256, 256) (-1, 1)
        #list_norm = list(map(lambda k: list(map(lambda v: norm_01(v), k)), input))  # (3, 256, 256) (0, 1)
        list_seg = list(map(lambda k: list(map(lambda v: self.get_seg(v), k)), input))
        #list_seg = list(map(lambda k: list(map(lambda v: self.get_magic(v), k)), input))
        return list_seg

# squ_ch
def squ_channel(input):
    _, output = torch.max(input, 1)
    output = torch.unsqueeze(output.type(torch.float32), 1)
    output = torch.cat([output] * 3, 1)
    return output


def to_print(to_show, save_name):

    to_show = [torch.cat(x, len(x[0].shape) - 1) for x in to_show]
    to_show = [x - x.min() for x in to_show]

    for i in range(len(to_show)):
        if len(to_show[i].shape) == 2:
            to_show[i] = torch.cat([to_show[i].unsqueeze(0)] * 3, 0)

    to_print = np.concatenate([x / x.max() for x in to_show], 1).astype(np.float16)
    imagesc(to_print, show=False, save=save_name)


# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--jsn', type=str, default='womac3', help='name of ini file')
parser.add_argument('--dataset', help='name of training dataset')
parser.add_argument('--testset', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, help='prjname')
parser.add_argument('--direction', type=str, help='a2b or b2a')
parser.add_argument('--unpaired', action='store_true', dest='unpaired', default=False)
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
parser.add_argument('--cropsize', default=256, type=int, help='size for cropping, 0 for no crop')
parser.add_argument('--t2d', action='store_true', dest='t2d', default=False)
parser.add_argument('--flip', action='store_true', dest='flip')
parser.add_argument('--eval', action='store_true', dest='eval')
parser.add_argument('--nepochs', default=(190, 200, 10), nargs='+', help='which checkpoints to be interfered with', type=int)
parser.add_argument('--nalpha', default=(0, 100, 1), nargs='+', help='range of additional input parameter for generator', type=int)
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--cart', action='store_true', dest='cartonly')

with open('outputs/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)

if len(args.nepochs) == 1:
    args.nepochs = [args.nepochs[0], args.nepochs[0]+1, 1]
if len(args.nalpha) == 1:
    args.nalpha = [args.nalpha[0], args.nalpha[0]+1, 1]

test_unit = Pix2PixModel(args=args)
print(len(test_unit.test_set))
for epoch in range(*args.nepochs):
    #try:
    test_unit.get_model(epoch, eval=args.eval)
    for ii in range(1):#range(len(test_unit.test_set)):
        #args.irange = [ii]
        for alpha in np.linspace(*args.nalpha):
            out_xy = list(map(lambda v: test_unit.get_one_output(v, 'x', alpha), args.irange))
            out_xy = list(zip(*out_xy))

            seg_xy = test_unit.get_all_seg(out_xy)
            diff_xy = [(x[1] - x[0]) for x in list(zip(out_xy[2], out_xy[0]))]

            diff_xy[0][0, 0, 0] = 2
            diff_xy[0][0, 0, 1] = -2

            x_seg = []
            for n in range(len(diff_xy)):
                seg_use = seg_xy[0][n]
                a = 1 * out_xy[0][n]
                #a[a < 0] = 0
                a = a / a.max()
                #a[:, seg_use == 0] = 0
                #a[:, seg_use == 2] = 0
                #a[:, seg_use == 4] = 0
                x_seg.append(a)

            # diff_seg = []
            # for n in range(len(diff_xy)):
            #     seg_use = seg_xy[2][n]
            #     a = 1 * diff_xy[n]
            #     a[a < 0] = 0
            #     a[:, seg_use == 0] = 0
            #     a[:, seg_use == 2] = 0
            #     a[:, seg_use == 4] = 0
            #     diff_seg.append(a)
            to_show = [out_xy[0],
                       list(map(lambda x, y: overlap_red(x, y), out_xy[0], seg_xy[0])),
                       out_xy[1],
                       # seg_xy[1],
                       # list(map(lambda x, y: overlap_red(x, y), out_xy[1], seg_xy[1])),
                       out_xy[2],
                       # seg_xy[2],
                       #diff_xy,
                       #x_seg,
                       # diff_seg,
                       list(map(lambda x, y: overlap_red(x, y), out_xy[2], seg_xy[2])),
                       ]

            to_print(to_show, save_name=os.path.join("outputs/results", args.dataset, args.prj,
                                                     str(epoch) + '_' + str(alpha) + '_' + str(ii).zfill(4) + '.jpg'))
        #except:
        #    print('failed for some reason')


# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY


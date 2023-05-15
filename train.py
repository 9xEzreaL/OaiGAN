from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil, copy
from dotenv import load_dotenv
from utils.make_config import *
load_dotenv('.env')

# Arguments
parser = argparse.ArgumentParser() # add_help=False)
# Project name
parser.add_argument('--prj', type=str, default='', help='name of the project')
parser.add_argument('--engine', dest='engine', type=str, default='mydcgan', help='use which engine')
# Data
parser.add_argument('--dataset', type=str, default='pain')
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
parser.add_argument('--n01', action='store_true', dest='n01', default=False)
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
parser.add_argument('--size_z', type=int, default=64, help='axis-z thickness')
parser.add_argument('--cart', type=str, default='none', dest='cartonly')
# Model
parser.add_argument('--gan_mode', type=str, default='vanilla', help='gan mode')
parser.add_argument('--netG', type=str, default='unet_128', help='netG model')
parser.add_argument('--mc', action='store_true', dest='mc', default=False, help='monte carlo dropout for pix2pix generator')
parser.add_argument('--netD', type=str, default='patchgan_16', help='netD model')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument("--n_attrs", type=int, default=1)
# Training
parser.add_argument('-b', dest='batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--n_epochs', type=int, default=201, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate f -or adam')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, default=0, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
# Loss
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
# Misc
parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

# Model-specific Arguments
engine = parser.parse_known_args()[0].engine
GAN = getattr(__import__('engine.' + engine), engine).GAN
# parser = GAN.add_model_specific_args(parser)
opt = parser.parse_args()

# Finalize Arguments and create files for logging
def prepare_log(opt):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    opt.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + opt.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + opt.dataset + '/' + opt.prj + '/', exist_ok=True)
    save_json(opt, os.environ.get('LOGS') + opt.dataset + '/' + opt.prj + '/' + '0.json')
    shutil.copy('engine/' + opt.engine + '.py', os.environ.get('LOGS') + opt.dataset + '/' + opt.prj + '/' + opt.engine + '.py')
    return opt

opt = prepare_log(opt)

#  Define Dataset Class
from dataloader.data_multi_new import MultiData as Dataset # OAI_pretrain
# from dataloader.data_multi_new import MultiData as Dataset

train_set = Dataset(root=os.environ.get('DATASET') + opt.dataset + '/train/',
                    path=opt.direction,
                    opt=opt, mode='train')
if opt.cartonly == 'cartonly8':
    opt.input_nc = train_set.__getitem__(0)[0].shape[0] + train_set.__getitem__(0)[2].shape[0]
    opt.output_nc = train_set.__getitem__(0)[0].shape[0]
elif opt.cartonly == 'cartonly6':
    opt.input_nc = 6
    opt.output_nc = train_set.__getitem__(0)[0].shape[0]
elif opt.cartonly == 'cartonly4':
    opt.input_nc = 4
    opt.output_nc = train_set.__getitem__(0)[0].shape[0]
elif opt.cartonly == 'cartonly4SPADE':
    opt.input_nc = 4
    opt.output_nc = train_set.__getitem__(0)[0].shape[0]
else:
    opt.input_nc = train_set.__getitem__(0)[0].shape[0]
    opt.output_nc = train_set.__getitem__(0)[0].shape[0]

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

#  Pytorch Lightning Module
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS') + opt.dataset + '/', name=opt.prj)
checkpoints = os.path.join(os.environ.get('LOGS'), opt.dataset, opt.prj, 'checkpoints')
os.makedirs(checkpoints, exist_ok=True)
net = GAN(hparams=opt, train_loader=None,
          test_loader=None, checkpoints=checkpoints)
trainer = pl.Trainer(gpus=[0],  # distributed_backend='ddp',
                     max_epochs=opt.n_epochs, progress_bar_refresh_rate=20, logger=logger)
trainer.fit(net, train_loader)  #, test_loader)  # test loader not used during training


# Example Usage
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset cartilage -b 16 --prj seg_unet --direction badregseg_goodseg --engine pix2pixNS --lamb 10
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset OAI_seg -b 16 --prj seg_attgan_patch4_cartonly --direction badreg_good --engine pix2pixNS --lamb 10 --netG attgan --netD patchgan_4

# CUDA_VISIBLE_DEVICES=3 python train.py --dataset OAI_DESS_segmentation/ZIB_3D_gan -b 16 --prj seg_attgan_patch4_cartonly --direction original_mask --engine pix2pixNS --lamb 10 --netG attgan --netD patchgan_4
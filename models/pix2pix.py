import torch
import torch.nn as nn
import torch.optim as optim
from models.networks import define_G, define_D
from models.networks import get_scheduler
from models.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.bias, 0)


class Pix2PixModel(pl.LightningModule):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        super(Pix2PixModel, self).__init__()
        print('using pix2pix.py')
        # initialize
        self.tini = time.time()
        self.epoch = 0
        self.avg_psnr = 0

        # opts
        self.hparams.update(vars(hparams))
        self.train_loader = train_loader
        self.test_loader = test_loader

        # original
        self.net_g = define_G(input_nc=hparams.input_nc, output_nc=hparams.output_nc, ngf=64, netG=hparams.netG,
                              norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
        # Discriminator
        if (hparams.netD).startswith('patchgan'):
            from models.cyclegan.models import Discriminator
            self.net_d = Discriminator(input_shape=(6, 256, 256), patch=(hparams.netD).split('_')[-1])
        else:
            self.net_d = define_D(input_nc=hparams.output_nc * 2, ndf=64, netD=hparams.netD)

        self.net_g = self.net_g.apply(_weights_init)
        self.net_d = self.net_d.apply(_weights_init)

        if self.hparams.lseg > 0:
            self.seg_model = torch.load(os.environ.get('model_seg')).cuda()

        [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, hparams)

        self.dir_checkpoints = checkpoints

        self.criterionL1 = nn.L1Loss().cuda()
        if hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(hparams.gan_mode).cuda()

        self.segLoss = SegmentationCrossEntropyLoss()

        # final hparams
        self.hparams.update(vars(hparams))

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))

        return [self.optimizer_d, self.optimizer_g], []

    def backward_g(self, inputs):
        self.net_g.zero_grad()
        conditioned_images = inputs[0]
        real_images = inputs[1]
        if self.hparams.bysubject:
            conditioned_images = conditioned_images[0, ::]
            real_images = real_images[0, ::]

        gout = self.net_g(conditioned_images)
        fake_images = gout[0]
        disc_logits = self.net_d(torch.cat((fake_images, conditioned_images), 1))
        adversarial_loss = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.criterionL1(fake_images, real_images)
        loss_g = adversarial_loss + self.hparams.lamb * recon_loss
        self.log('loss_recon_a', self.hparams.lamb * recon_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # target domain identity loss
        fake_images_b = self.net_g(real_images)[0]
        #recon_loss_b = nn.MSELoss()(fake_images_b, real_images)
        recon_loss_b = self.criterionL1(fake_images_b, real_images)
        loss_g = loss_g + self.hparams.lamb_b * recon_loss_b
        self.log('loss_recon_b', self.hparams.lamb_b * recon_loss_b, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return loss_g

    def backward_d(self, inputs):
        self.net_d.zero_grad()
        conditioned_images = inputs[0]
        real_images = inputs[1]
        if self.hparams.bysubject:
            conditioned_images = conditioned_images[0, ::]
            real_images = real_images[0, ::]
        gout = self.net_g(conditioned_images)
        fake_images = gout[0].detach()

        fake_logits = self.net_d(torch.cat((fake_images, conditioned_images), 1))
        real_logits = self.net_d(torch.cat((real_images, conditioned_images), 1))

        fake_loss = self.criterionGAN(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.criterionGAN(real_logits, torch.ones_like(real_logits))

        # Combined D loss
        loss_d = (real_loss + fake_loss) * 0.5
        return loss_d

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch
        if optimizer_idx == 0:
            #self.net_d.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = True
            loss_d = self.backward_d(inputs)
            self.log('loss_d', loss_d, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_d

        if optimizer_idx == 1:
            #self.net_g.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = False
            loss_g = self.backward_g(inputs)
            self.log('loss_g', loss_g, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_g

    def training_epoch_end(self, outputs):
        hparams = self.hparams
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()
        # checkpoint
        dir_checkpoints = self.dir_checkpoints
        if self.epoch % 20 == 0:
            if not os.path.exists(dir_checkpoints):
                os.mkdir(dir_checkpoints)
            if not os.path.exists(os.path.join(dir_checkpoints, hparams.prj)):
                os.mkdir(os.path.join(dir_checkpoints, hparams.prj))
            net_g_model_out_path = dir_checkpoints + "/{}/netG_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            net_d_model_out_path = dir_checkpoints + "/{}/netD_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            torch.save(self.net_g, net_g_model_out_path)
            torch.save(self.net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format(dir_checkpoints + '/' + hparams.prj))

        self.epoch += 1
        self.tini = time.time()
        self.avg_psnr = 0

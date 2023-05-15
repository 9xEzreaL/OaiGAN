from engine.base import BaseModel
import torch
import copy

# squ_ch
def squ_channel(input, hparam):
    _, output = torch.max(input, 1)
    output = torch.unsqueeze(output.type(torch.float32), 1)
    output = torch.cat([output] * 3, 1)
    if hparam.input_nc == 4:
        output = output[:, 0, :, :].unsqueeze(1)
    return output

def exclude_bone(mri, mask):
    excluded_mask = ((mask==0))
    return mri*excluded_mask

class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        if self.hparams.input_nc != 4:
            self.net_dX = self.net_d
            self.net_dY = copy.deepcopy(self.net_d)
        # save model names and optimize
            self.netd_names = {'net_dX': 'netDX', 'net_dY': 'netDY'}
        # self.netd_names = {'net_dX': 'netDX'}
    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def generation(self):
        self.seg_model = torch.load('submodels/model_seg256.pth')
        self.seg_model.eval()

        self.oriX = self.batch[0] # bad mri (3,256,256)
        self.oriY = self.batch[1] # good mri (3,256,256)
        self.maskX = self.batch[2] # bad mask (5,256,256)
        self.maskY = self.batch[3] # good mask (5,256,256)
        self.maskX = squ_channel(self.maskX, self.hparams)
        self.maskY = squ_channel(self.maskY, self.hparams)

        self.catXY = torch.cat((self.oriX,self.maskX),1)
        try:
            self.imgX0 = self.net_g(self.catXY, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        except:
            try:
                self.imgX0 = self.net_g(self.catXY)[0]
            except:
                self.imgX0, self.imgX1 = self.net_g(self.catXY, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())

        self.maskX0 = self.seg_model(self.imgX0) # (5,256,256)
        self.maskX0 = squ_channel(self.maskX0, self.hparams)

        self.exclude_imgX0 = self.imgX0 * ((self.maskX==0))
        self.exclude_oriX = self.oriX * ((self.maskX==0))

    def backward_g(self, inputs):
        # ADV(X0)+
        loss_g = 0
        loss_g = self.add_loss_adv(a=self.imgX0, net_d=self.net_dX, loss=loss_g, coeff=0.5, truth=True, stacked=False) # 3+3 channel

        # ADV(maskX0)+
        loss_g = self.add_loss_adv(a=self.maskX0, net_d=self.net_dY, loss=loss_g, coeff=0.5, truth=True, stacked=False) # 3+3 channel

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=self.imgX0, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb)
        # loss_g = self.add_loss_L1(a=self.exclude_imgX0, b=self.exclude_oriX, loss=loss_g, coeff=self.hparams.lamb)
        loss_g = self.add_loss_L1(a=self.maskX0, b=self.maskX, loss=loss_g, coeff=self.hparams.lamb*10)

        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(X0)-
        loss_d = self.add_loss_adv(a=self.imgX0, net_d=self.net_dX, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_dX, loss=loss_d, coeff=0.5, truth=True, stacked=False)

        # ADV(maskX0, maskY)+
        loss_d = self.add_loss_adv(a=self.maskX0, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=False, stacked=False)

        # ADV(catXX)-
        loss_d = self.add_loss_adv(a=self.maskX, net_d=self.net_dY, loss=loss_d, coeff=0.5, truth=True, stacked=False)
        return loss_d



# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlyZ -b 16 --prj WpOp/sb256 --direction xyweak_xyorisb --resize 256 --engine pix2pixNS
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 16 --prj NS/NoL1 --direction aregis1_b --cropsize 256 --engine pix2pixNS --lamb 0

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset kl3 -b 16 --prj NS/0 --direction badKL3afterreg_gooodKL3reg --cropsize 256 --engine pix2pixNS


# CUDA_VISIBLE_DEVICES=2 python train.py --dataset cartilage -b 16 --prj seg/unet64/patch8_L0_inc5_pro --direction bad5c_good5c --engine pix2pixNS --lamb 0 --netG unet_64 --netD patchgan_8 --cart --input_nc 5 --output_nc 5
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset cartilage -b 32 --prj ori+seg/SegBoneL10_unet32patch4 --direction badreg_good_SegBonel1unet32patch8_good5csm --engine new_model --lamb 10 --netG unet_32 --netD patchgan_4 --cartonly
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset cartilage -b 16 --prj ori+seg/Use3cSegOriBoneSegBoneL10_attganpatch4_dataload0to1 --direction badreg_good_segbone_good5csm --engine new_model2 --lamb 10 --netG attgan --netD patchgan_4 --cartonly6
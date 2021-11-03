from os import listdir
from os.path import join
import glob
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from dataloader.utils import is_image_file
from PIL import Image
import numpy as np


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


def separate_subjects_n_slices(img_list):
    temp = [x.split('.')[0].split('_') for x in img_list]
    subject = dict()
    for y in temp:
        if int(y[0]) not in subject.keys():
            subject[int(y[0])] = []
        subject[int(y[0])] = subject[int(y[0])] + [int(y[1])]
    for k in list(subject.keys()):
        subject[k].sort()
    return subject


class DatasetFromFolderSubjects(data.Dataset):
    def __init__(self, image_dir, opt, mode):
        super(DatasetFromFolderSubjects, self).__init__()
        self.single_data = DatasetFromFolder(image_dir, opt, mode)
        self.image = self.single_data.image
        self.subject = separate_subjects_n_slices(self.image)
        self.subject_keys = list(self.subject)
        self.subject_keys.sort()

    def __len__(self):
        return len(self.subject_keys)

    def __getitem__(self, index):
        sub = self.subject_keys[index]
        image_list = self.subject[sub]

        a_list = []
        b_list = []
        for image in image_list:
            image_name = str(sub) + '_' + str(image) + '.png'
            a, b = self.single_data.__getitem__(self.image.index(image_name))
            a_list.append(a.unsqueeze(0))
            b_list.append(b.unsqueeze(0))

        a_list = torch.cat(a_list, 0)
        b_list = torch.cat(b_list, 0)

        return a_list, b_list


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, opt, mode):
        super(DatasetFromFolder, self).__init__()
        self.opt = opt
        self.mode = mode
        self.a_path = join(image_dir, self.opt.direction.split('_')[0])
        self.b_path = join(image_dir, self.opt.direction.split('_')[1])
        self.image = sorted([x.split('/')[-1] for x in glob.glob(self.a_path+'/*')])

        #transform_list = [transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image[index]))  #.convert('RGB') (DESS: 294>294) (PAIN: 224>286)
        #print(np.array(a).shape)
        b = Image.open(join(self.b_path, self.image[index]))  #.convert('RGB')
        a = transforms.ToTensor()(np.array(a))
        b = transforms.ToTensor()(np.array(b))
        a = a.type(torch.float32)
        b = b.type(torch.float32)
        a = a / a.max()
        b = b / b.max()

        # random crop
        osize = a.shape[2]
        if self.mode == 'train':
            w_offset = random.randint(0, max(0, osize - 256 - 1))
            h_offset = random.randint(0, max(0, osize - 256 - 1))
            a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        elif self.mode == 'test':
            w_offset = 15
            h_offset = 15
            print(a.shape)
            print(b.shape)

        if a.shape[0] != 3:
            a = torch.cat([a] * 3, 0)
            b = torch.cat([b] * 3, 0)
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        #print(a.shape)

        if self.opt.flip:
            if self.mode == 'train':  # flipping
                if random.random() < 0.5:
                    idx = [i for i in range(a.size(2) - 1, -1, -1)]
                    idx = torch.LongTensor(idx)
                    a = a.index_select(2, idx)
                    b = b.index_select(2, idx)

        return a, b


if __name__ == '__main__':
    ds = get_training_set('/media/ghc/GHc_data1/paired_images/TSE_DESS/', direction='a_b', mode='train')

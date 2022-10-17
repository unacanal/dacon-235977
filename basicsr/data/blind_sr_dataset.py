import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from utils import utils_blindsr as blindsr

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.utils import FileClient, imfrombytes
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BlindSRDataset(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(BlindSRDataset, self).__init__()
        self.opt = opt
        
        self.n_channels = opt['in_chans'] if opt['in_chans'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.patch_size = self.opt['gt_size'] if self.opt['gt_size'] else 512
        self.lq_patchsize = self.opt['gt_size'] // self.sf

        self.gt_folder = opt['dataroot_gt']
        
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths_H = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths_H = [os.path.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths_H = sorted(list(scandir(self.gt_folder, full_path=True)))

        # self.paths_H = util.get_image_paths(opt['dataroot_gt'])
        print(len(self.paths_H))

#        for n, v in enumerate(self.paths_H):
#            if 'face' in v:
#                del self.paths_H[n]
#        time.sleep(1)
        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):
        L_path = None
        
        # ------------------------------------
        # get H image
        # ------------------------------------
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        H_path = self.paths_H[index]
        # img_H = util.imread_uint(H_path, self.n_channels)
        img_bytes = self.file_client.get(H_path, 'gt')
        img_H = imfrombytes(img_bytes, float32=True)
        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        if H < self.patch_size or W < self.patch_size:
            img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
            else:
                mode = random.randint(0, 7)
                img_H = util.augment_img(img_H, mode=mode)

            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'lq': img_L, 'gt': img_H, 'gt_path': H_path}

    def __len__(self):
        return len(self.paths_H)
###
### This code owes a lot to the SPIN authors
### https://github.com/nkolot/SPIN
###

from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join
import os
import json
import glob

from global_utils import config
from dataset_dogs.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

from scipy.io import loadmat
import scipy.misc
from pycocotools.mask import decode as decode_RLE

def seg_from_anno(entry):
	"""Given a .json entry, returns the binary mask as a numpy array"""

	rle = {
		"size": [entry['img_height'], entry['img_width']],
		"counts": entry['seg']
	}

	decoded = decode_RLE(rle)
	return decoded

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, 
        dataset, 
        param_dir=None,
        use_augmentation=True, 
        is_train=True, 
        img_res=224):

        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.param_dir = param_dir
        
        BASE_FOLDER = config.DATASET_FOLDERS[dataset]

        self.img_dir = os.path.join(BASE_FOLDER, 'Images')
        self.jsonfile   = os.path.join(BASE_FOLDER, config.JSON_NAME[dataset]) # accessing new version of keypoints.json

        # self.jsonfile = "/home/bjb10042/projects/data/dogs_v2/stanford/keypoints_v101.json"

        # create train/test split
        with open(self.jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.data_idx = np.load(os.path.join(config.DATASET_FILES[is_train][dataset]))

        # self.options = options
        self.normalize_img = Normalize(
            mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)

        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]'
        self.noise_factor = 0.4 # Random rotation in the range [-rot_factor, rot_factor]
        self.scale_factor = 0.25 # Rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.img_res = img_res
	
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train and self.use_augmentation:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            # pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # rot = min(2*self.options.rot_factor,
            #         max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))

            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            # sc = min(1+self.options.scale_factor,
            #         max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))

            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn,border_grey_intensity=0.0):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [self.img_res, self.img_res], rot=rot, 
                      border_grey_intensity=border_grey_intensity)

        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            # kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
            #                       [self.options.img_res, self.options.img_res], rot=r)

            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S = np.einsum('ij,kj->ki', rot_mat, S) 
        # flip the x coordinates
        if f:
             S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        idx = index
        img_idx = self.data_idx[idx]
        a = self.anno[img_idx]

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        imgname_raw = a['img_path']

         # Load image
        imgname = join(self.img_dir, imgname_raw)

        # Some datatsets store \ instead of /, so convert
        imgname = imgname.replace("\\", "/")

        assert os.path.exists(imgname), "Cannot find image: {0}".format(imgname)
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)

        seg = seg_from_anno(a)  # (H, W) bool
        seg = seg.astype(np.float) * 255.
        seg = np.dstack([seg, seg, seg])  # (H, W, 3) as float

        x0, y0, width, height = a['img_bbox']
        
        scaleFactor = 1.2
        scale = scaleFactor*max(width, height)/200

        kp_S24 = np.array(a['joints'])
        center = np.array([x0+width/2, y0+height/2])  # Center of dog
      
        kp_S24_norm = torch.from_numpy(self.j2d_processing(kp_S24.copy(), center, sc*scale, rot, flip)).float()
        img_crop = self.rgb_processing(img, center, sc*scale, rot, flip, pn, border_grey_intensity=255.0)
        seg_crop = self.rgb_processing(seg, center, sc*scale, rot, flip, np.array([1.0, 1.0, 1.0])) # No pixel noise multiplier

        item = {}

        item['has_pose_3d'] = False
        item['has_smpl'] = False
        item['keypoints_3d'] = np.zeros((24, 4))

        item['pred_pose'] = np.zeros((105))
        item['pred_shape'] = np.zeros((26))
        item['pred_camera'] = np.zeros((3))
        item['pred_trans'] = np.zeros((3))

        if self.param_dir is not None:
            inp_path = imgname_raw.replace("/", "_").replace(".jpg", ".npz")
            if self.dataset == 'animal_pose':
                inp_path = "images_{0}".format(inp_path)

            with np.load(os.path.join(self.param_dir, inp_path)) as f:
                item['pred_pose'] = f.f.pose
                item['pred_shape'] = f.f.betas
                item['pred_camera'] = f.f.camera
                item['pred_trans'] = f.f.trans

        item['imgname'] = imgname
        item['keypoints'] = kp_S24_norm[config.EVAL_KEYPOINTS]
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['index'] = img_idx

        img = torch.from_numpy(img).float()
        img_crop = torch.from_numpy(img_crop).float()
        seg_crop = torch.from_numpy(seg_crop[[0]]).float() # [3, h, w] -> [1, h, w]
        
        item['img_orig'] = img_crop.clone()
        item['img'] = self.normalize_img(img_crop)
        item['img_border_mask'] = torch.all(img_crop < 1.0, dim = 0).unsqueeze(0).float()
        item['seg'] = seg_crop.clone()
        item['has_seg'] = True
        item['dataset'] = self.dataset

        return item

    def __len__(self):
        return len(self.data_idx)
        # return 32
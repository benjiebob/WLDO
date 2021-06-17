"""Dataset for loading directory of unlabelled images for feeding through network"""

from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join
import os
import json
from tqdm import tqdm

from global_utils import config

imresize = cv2.resize

is_img_file = lambda f: any(ext in f for ext in [".png", ".jpg"])

class DemoDataset(Dataset):
    """Inherits from base dataset.
    Used for a sequence of images - no ground truth data provided."""

    def __init__(self, img_dir):

        super(DemoDataset, self).__init__()

        self.img_dir = img_dir

        self.fixed_breed_id = -1

        img_files = [f for f in os.listdir(self.img_dir) if is_img_file(f)]
        self.img_files = sorted(img_files)

        # self.options = options
        self.normalize_img = Normalize(
            mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
        self.img_res = config.IMG_RES # Square resolution images will be resized to


    def rgb_processing(self, rgb_img, bbox):
        """Process rgb image (no augmentation).
        First, crops to bbox.
        Then, resizes to img_res.
        Adjusts rgb to be 0-1 range
        Also shifts axes so output is in format (3, H, W)"""

        (x0, y0, width, height) = bbox
        rgb_img = rgb_img[y0:y0 +height, x0:x0 +width]
        rgb_img = imresize(rgb_img, (self.img_res, self.img_res)).astype(np.float32)
        if rgb_img.max() > 1: rgb_img *= 1/ 255
        return np.rollaxis(rgb_img, 2, 0).astype(np.float32)

    def __getitem__(self, index):
        idx = index

        imgname_raw = self.img_files[idx]

        # Load image
        imgname = join(self.img_dir, imgname_raw)

        assert os.path.exists(imgname), "Cannot find image: {0}".format(imgname)

        img_full_scale = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)

        # if in 0-255 RGB, convert to 0-1:
        if img_full_scale.max() > 1:
            img_full_scale *= 1 / 255

        y0, x0, height, width = 0, 0, *img_full_scale.shape[:2]

        H, W, *_ = img_full_scale.shape

        has_bbox = not all(
            [x0 == 0, y0 == 0, width == W, height == H])  # assume no bbox detected if bbox is same as image shape
        # scaleFactor = 1.2
        # # scale = scaleFactor * max(width, height)/200
        # scale = scaleFactor * max(width/W, height/H)
        # sc = 1 # aug scaling param = 1
        center = np.array([x0 + width / 2, y0 + height / 2])  # Center of dog

        img_crop = self.rgb_processing(img_full_scale, [x0, y0, width, height])  # cropped to bbox

        # for viewing image later, resize to square
        img_resized = self.rgb_processing(img_full_scale, (0, 0, W, H))  # scale full image

        ## work out bbox in resized frame
        _, H_res, W_res = img_resized.shape  # img_resized has shape (3, H, W)
        stretch_x, stretch_y = W_res / W, H_res / H  # stretch factors in image resizing
        bbox_res = [x0 * stretch_x, y0 * stretch_y, width * stretch_x, height * stretch_y]
        x0s, y0s, ws, hs = bbox_res
        center_res = np.array([x0s + ws / 2, y0s + hs / 2])  # Center of dog

        item = {}
        item['imgname'] = imgname
        item['scale'] = 1
        item['center'] = center_res
        item["bbox"] = torch.FloatTensor(bbox_res)  # bbox in img_resized ref frame

        if img_resized.max() > 1:
            img_resized *= 1 / 255

        img_resized = torch.from_numpy(img_resized).float()
        img_crop = torch.from_numpy(img_crop).float()

        item['img_orig'] = img_resized.clone()
        item['img'] = self.normalize_img(img_crop)
        item["has_seg"] = False
        item["has_bbox"] = has_bbox

        # terms required for model.forward
        item['keypoints'] = np.zeros((len(config.EVAL_KEYPOINTS), 3))
        item['seg'] = False
        item['index'] = index
        item['dataset'] = 'clip'
        item['img_border_mask'] = torch.all(img_crop < 1.0, dim = 0).unsqueeze(0).float()

        return item

    def __len__(self):
        len_data = len(self.img_files)
        return len_data
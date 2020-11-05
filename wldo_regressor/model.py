import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle as pkl
import time
import os

from global_utils.smal_model.smal_torch import SMAL
from global_utils.nnutils.smal_mesh_net import MeshNet
from global_utils.nnutils.nmr import NeuralRenderer
import global_utils.nnutils.geom_utils as geom_utils

from global_utils import config
from global_utils.helpers.visualize import Visualizer
from metrics import Metrics

class Model(nn.Module):
    def __init__( self, device, **kwargs):

        super(Model, self).__init__()

        self.model_renderer = NeuralRenderer(
            config.IMG_RES, proj_type=config.PROJECTION, 
            norm_f0=config.NORM_F0, 
            norm_f=config.NORM_F, 
            norm_z=config.NORM_Z)
        self.model_renderer.directional_light_only()

        self.netG_DETAIL = MeshNet(
            [config.IMG_RES, config.IMG_RES], 
            norm_f0=config.NORM_F0,
            nz_feat=config.NZ_FEAT)

        self.smal = SMAL(device, shape_family_id=config.SHAPE_FAMILY_ID)
        print ("INITIALIZED")
        

    def forward(self, batch_input, eval=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        If eval is False, do not calculate accuracy metrics (PCK, IOU)."""

        img = batch_input['img']
        keypoints = batch_input['keypoints']
        seg=batch_input['seg']
        index=batch_input['index']
        has_seg=batch_input['has_seg']
        dataset=batch_input['dataset']
        img_orig=batch_input['img_orig']
        scale=batch_input['scale']
        img_border_mask=batch_input['img_border_mask']

        batch_size = img.shape[0]
        
        pred_codes, _ = self.netG_DETAIL(img) # This is the generator
        scale_pred, trans_pred, pose_pred, betas_pred, betas_logscale = \
            pred_codes

        all_betas = torch.cat([betas_pred, betas_logscale], dim = 1)
        pred_camera = torch.cat([
                scale_pred[:, [0]], 
                torch.ones(batch_size, 2).cuda() * config.IMG_RES / 2
        ], dim = 1)

        verts, joints, _, _ = self.smal(
            betas_pred, pose_pred, 
            trans=trans_pred,
            betas_logscale=betas_logscale) # edit to take betas_logscale

        faces = self.smal.faces.unsqueeze(0).expand(
            verts.shape[0], 7774, 3)
       
        synth_rgb, _, synth_silhouettes = self.model_renderer(
            verts, faces, pred_camera)

        synth_rgb = torch.clamp(synth_rgb, 0.0, 1.0)
        synth_silhouettes = synth_silhouettes.unsqueeze(1)

        labelled_joints_3d = joints[:, config.MODEL_JOINTS]
        synth_landmarks = self.model_renderer.project_points(
            labelled_joints_3d, pred_camera)
        
        preds = {}
        preds['pose'] = pose_pred
        preds['betas'] = all_betas
        preds['camera'] = pred_camera
        preds['trans'] = trans_pred
        
        preds['verts'] = verts
        preds['joints_3d'] = labelled_joints_3d
        preds['faces'] = faces

        if eval:
            preds['acc_PCK'] = Metrics.PCK(
                synth_landmarks, keypoints, 
                seg, has_seg
            )

            preds['acc_IOU'] = Metrics.IOU(
                synth_silhouettes, seg, 
                img_border_mask, mask = has_seg
            )

            for group, group_kps in config.KEYPOINT_GROUPS.items():
                preds[f'{group}_PCK'] = Metrics.PCK(
                    synth_landmarks, keypoints, seg, has_seg, idxs=group_kps
                )
        
        preds['synth_xyz'] = synth_rgb
        preds['synth_silhouettes'] = synth_silhouettes
        preds['synth_landmarks'] = synth_landmarks

        return preds
import torch
import torch.nn.functional as F
from global_utils import config
import numpy as np

class Metrics():

    @staticmethod
    def PCK(
        synth_landmarks, keypoints, 
        gtseg, has_seg, 
        thresh=0.15, idxs:list=None):

        """Calc PCK with same method as in eval.
        idxs = optional list of subset of keypoints to index from
        """

        synth_landmarks, keypoints, gtseg = synth_landmarks[has_seg], keypoints[has_seg], gtseg[has_seg]

        if idxs is None:
            idxs = list(range(keypoints.shape[1]))

        idxs = np.array(idxs).astype(int)

        synth_landmarks = synth_landmarks[:, idxs]
        keypoints = keypoints[:, idxs]

        keypoints_gt = ((keypoints + 1.0) * 0.5) * config.IMG_RES
        dist = torch.norm(synth_landmarks - keypoints_gt[:, :, [1, 0]], dim = -1)
        seg_area = torch.sum(gtseg.reshape(gtseg.shape[0], -1), dim = -1).unsqueeze(-1)
        
        hits = (dist / torch.sqrt(seg_area)) < thresh
        total_visible = torch.sum(keypoints[:, :, -1], dim = -1)
        pck = torch.sum(hits.float() * keypoints[:, :, -1], dim = -1) / total_visible
        return pck

    @staticmethod
    def IOU(synth_silhouettes, gt_seg, img_border_mask, mask):
        for i in range(mask.shape[0]):
            synth_silhouettes[i] *= mask[i]

        # Do not penalize parts of the segmentation outside the img range
        gt_seg = (gt_seg * img_border_mask) + synth_silhouettes * (1.0 - img_border_mask)

        intersection = torch.sum((synth_silhouettes * gt_seg).reshape(synth_silhouettes.shape[0], -1), dim = -1)
        union = torch.sum(((synth_silhouettes + gt_seg).reshape(synth_silhouettes.shape[0], -1) > 0.0).float(), dim = -1)
        acc_IOU_SCORE = intersection / union
        
        return acc_IOU_SCORE
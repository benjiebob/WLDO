import numpy as np
import cv2

from torchvision.utils import make_grid
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch.nn.functional as F
import torch

class SMALJointDrawer():
    def __init__(self):
        self.jet_colormap = cm.ScalarMappable(norm = Normalize(0, 1), cmap = 'jet')

    def draw_joints(self, image, landmarks, visible = None, normalized = True, marker_size = 8, thickness = 3):
        image_np = np.transpose(image.cpu().data.numpy(), (0, 2, 3, 1))
        landmarks_np = landmarks.cpu().data.numpy()
        if visible is not None:
            visible_np = visible.cpu().data.numpy()
        else:
            visible_np = visible

        return_stack = self.draw_joints_np(image_np, landmarks_np, visible_np, normalized, marker_size=marker_size, thickness=thickness)
        return torch.FloatTensor(np.transpose(return_stack, (0, 3, 1, 2)))

    def draw_joints_np(self, image_np, landmarks_np, visible_np = None, normalized = False, marker_size = 8, thickness = 3):
        if normalized:
            image_np = (image_np * 0.5) + 0.5
         
        image_np = (image_np * 255.0).astype(np.uint8)

        bs, nj, _ = landmarks_np.shape
        if visible_np is None:
            visible_np = np.ones((bs, nj), dtype=bool)

        return_images = []
        for image_sgl, landmarks_sgl, visible_sgl in zip(image_np, landmarks_np, visible_np):
            image_sgl = image_sgl.copy()
            for joint_id, ((y_co, x_co), vis) in enumerate(zip(landmarks_sgl, visible_sgl)):
                color = np.array([255, 0, 0])
                marker_type = 0
                if not vis:
                    x_co, y_co = 0, 0

                cv2.drawMarker(image_sgl, (int(x_co), int(y_co)), (int(color[0]), int(color[1]), int(color[2])), marker_type, marker_size, thickness = thickness)
                cv2.putText(image_sgl, str(joint_id), (int(x_co) + 10, int(y_co)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                
            return_images.append(image_sgl)

        return_stack = np.stack(return_images, 0)
        return_stack = return_stack / 255.0
        if normalized:
            return_stack = (return_stack - 0.5) / 0.5 # Scale and re-normalize for pytorch
        
        return return_stack

    def draw_heatmap_grids(self, heatmaps, silhouettes, upsample_val = 1, alpha_blend = 0.9):
        bs, jts, h, w = heatmaps.shape
        heatmap_grids = []
        for heatmap, silhouette in zip(heatmaps, silhouettes):
            heatmap_rgb = heatmap[:, None, :, :].expand(jts, 3, h, w) * alpha_blend + silhouette * (1 - alpha_blend)
            grid = make_grid(heatmap_rgb, nrow = 5)
            heatmap_jet = self.jet_colormap.to_rgba(grid[0].cpu().numpy())[:, :, :3]
            heatmap_jet = np.transpose(heatmap_jet, (2, 0, 1))
            heatmap_grids.append(torch.FloatTensor(heatmap_jet))
        
        heatmap_stack = torch.stack(heatmap_grids, dim = 0)
        heatmap_stack = (heatmap_stack - 0.5) / 0.5

        return F.interpolate(heatmap_stack, [h * upsample_val, w * upsample_val])


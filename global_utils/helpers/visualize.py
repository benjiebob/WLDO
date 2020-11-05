
import torch
from global_utils.helpers.draw_smal_joints import SMALJointDrawer
# import plotly.graph_objects as go
import numpy as np

class Visualizer():

    @staticmethod
    def generate_output_figures(preds):
        marker_size = 8
        thickness = 4

        smal_drawer = SMALJointDrawer()
        synth_render_vis = smal_drawer.draw_joints(preds['synth_xyz'], preds['synth_landmarks'], marker_size=marker_size, thickness=thickness, normalized=False)

        keypoints_gt = ((preds['keypoints'] + 1) * 0.5) * 224
        real_rgb_vis = smal_drawer.draw_joints(preds['img_orig'], keypoints_gt[:, :, [1, 0]], marker_size=marker_size, thickness=thickness, normalized=False)

        pic_on_pic = preds['img_orig'].cpu() * 0.4 + preds['synth_xyz'].cpu() * 0.6
        pic_npy = pic_on_pic.data.cpu()

        sil_err = torch.cat([
            preds['seg'].cpu(), preds['synth_silhouettes'].cpu(), 1.0 - preds['img_border_mask'].cpu()
        ], dim = 1)

        sil_input = preds['seg'].cpu().expand(-1, 3, -1, -1)
        sil_err_npy = sil_err.data.cpu()
        output_figs = torch.stack(
            [real_rgb_vis, sil_input, synth_render_vis, pic_npy, sil_err_npy], dim = 1) # Batch Size, 4 (Images), RGB, H, W

        return output_figs

    @staticmethod
    def generate_demo_output(preds):
        """Figure output of: [raw_img, cropped, mesh_view, mesh & raw, silh_view]"""
        marker_size = 8
        thickness = 4

        smal_drawer = SMALJointDrawer()

        #synth_render_vis = smal_drawer.draw_joints(preds['synth_xyz'], preds['synth_landmarks'], marker_size=marker_size, thickness=thickness, normalized=False)
        synth_render_vis = preds['synth_xyz'].cpu()

        # Real image (with bbox overlayed)
        real_rgb_vis = preds['img_orig'].cpu()

        #Overlaid bbox
        # bbox_overlay = np.zeros(real_rgb_vis.shape).astype(np.float32)
        # for n, bbox in enumerate(preds['bbox']):
        #     (x0, y0, width, height) = list(map(int, bbox.cpu().numpy()))
        #     bbox_overlay[n, 0, y0:y0+height, x0:x0+width] = 1. # red channel overlay
        #
        # real_rgb_vis += .4 * bbox_overlay # overlay bbox

        # real cropped
        real_rgb_vis_cropped = preds['img'].cpu()

        pic_on_pic = preds['img'] * 0.4 + preds['synth_xyz'].cpu() * 0.6
        pic_npy = pic_on_pic.data.cpu()

        batch, _, H, W = pic_npy.shape
        sil_err = torch.cat([
            preds['synth_silhouettes'], preds['synth_silhouettes'], preds['synth_silhouettes']], dim = 1)

        sil_err_npy = sil_err.data.cpu()
        output_figs = torch.stack(
            [real_rgb_vis, synth_render_vis, pic_npy, sil_err_npy], dim = 1) # Batch Size, 4 (Images), RGB, H, W

        return output_figs

    @staticmethod
    def draw_mesh_plotly(
        viz,
        title,
        verts, faces, 
        visdom_env_imgs,
        up=dict(x=0,y=1,z=0), eye=dict(x=0.0, y=0.0, z=1.0),
        hack_box_size = 1.0,
        center_mesh = True):

        camera = dict(up=up, center=dict(x=0, y=0, z=0), eye=eye)
        scene = dict(
            xaxis = dict(nticks=10, range=[-1,1],),
            yaxis = dict(nticks=10, range=[-1,1],),
            zaxis = dict(nticks=10, range=[-1,1],),
            camera = camera)

        centre_of_mass = torch.mean(verts, dim = 0, keepdim=True)
        if not center_mesh:
            centre_of_mass = torch.zeros_like(centre_of_mass)
        
        output_verts = (verts - centre_of_mass).data.cpu().numpy()
        output_faces = faces.data.cpu().numpy()

        hack_points = np.array([
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0]]) * hack_box_size

        vis_fig = go.Figure(data = [
            go.Mesh3d(
                x = output_verts[:, 0],
                y = output_verts[:, 1],
                z = -1 * output_verts[:, 2],
                i = output_faces[:, 0],
                j = output_faces[:, 1],
                k = output_faces[:, 2],
                color='cornflowerblue'
            ),
            go.Scatter3d(
                x = hack_points[:, 0],
                y = hack_points[:, 1],
                z = hack_points[:, 2],
                mode='markers',
                name='_fake_pts',
                visible=True,
                marker=dict(
                    size=1,
                    opacity = 0,
                    color=(0.0, 0.0, 0.0),
                )
            )])

        vis_fig.update_scenes(patch = scene)
        vis_fig.update_layout(title=title)
        return vis_fig

    def draw_double_mesh_plotly(
        viz,
        title,
        verts, faces,
        verts2, 
        joints_3d,
        gt_joints_3d,
        visdom_env_imgs,
        up=dict(x=0,y=1,z=0), eye=dict(x=0.0, y=0.0, z=1.0),
        hack_box_size = 1.0,
        center_mesh = True):

        camera = dict(up=up, center=dict(x=0, y=0, z=0), eye=eye)
        scene = dict(
            xaxis = dict(nticks=10, range=[-1,1],),
            yaxis = dict(nticks=10, range=[-1,1],),
            zaxis = dict(nticks=10, range=[-1,1],),
            camera = camera)

        centre_of_mass = torch.mean(verts, dim = 0, keepdim=True)
        if not center_mesh:
            centre_of_mass = torch.zeros_like(centre_of_mass)
        
        output_joints_3d = (joints_3d - centre_of_mass).data.cpu().numpy()
        output_verts = (verts - centre_of_mass).data.cpu().numpy()
        output_verts2 = (verts2 - centre_of_mass).data.cpu().numpy()
        output_joints_3d2 = (gt_joints_3d - centre_of_mass).data.cpu().numpy()
        output_faces = faces.data.cpu().numpy()

        hack_points = np.array([
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0]]) * hack_box_size

        vis_fig = go.Figure(data = [
            go.Mesh3d(
                x = output_verts[:, 0],
                y = output_verts[:, 1],
                z = -1 * output_verts[:, 2],
                i = output_faces[:, 0],
                j = output_faces[:, 1],
                k = output_faces[:, 2],
                color='cornflowerblue',
                opacity=0.5,
            ),
            go.Scatter3d(
                x = output_joints_3d[:, 0],
                y = output_joints_3d[:, 1],
                z = -1 * output_joints_3d[:, 2],
                mode='markers',
                marker = dict(
                    size=8,
                    color='red'
                )
            ),
            go.Mesh3d(
                x = output_verts2[:, 0],
                y = output_verts2[:, 1],
                z = -1 * output_verts2[:, 2],
                i = output_faces[:, 0],
                j = output_faces[:, 1],
                k = output_faces[:, 2],
                color='green',
                opacity=0.5,
            ),
            go.Scatter3d(
                x = output_joints_3d2[:, 0],
                y = output_joints_3d2[:, 1],
                z = -1 * output_joints_3d2[:, 2],
                mode='markers',
                marker = dict(
                    size=8,
                    color='purple'
                )
            ),
            go.Scatter3d(
                x = hack_points[:, 0],
                y = hack_points[:, 1],
                z = hack_points[:, 2],
                mode='markers',
                name='_fake_pts',
                visible=True,
                marker=dict(
                    size=1,
                    opacity = 0,
                    color=(0.0, 0.0, 0.0),
                )
            )])

        vis_fig.update_scenes(patch = scene)
        vis_fig.update_layout(title=title)
        return vis_fig
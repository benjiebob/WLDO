"""
File for using the model on a single directory
"""

from model import Model

import torch, os
import argparse

from dataset_dogs.demo_dataset import DemoDataset
from torch.utils.data import DataLoader
from global_utils.helpers.visualize import Visualizer

import cv2
import numpy as np
from tqdm import tqdm

nn = torch.nn

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='../data/pretrained/3501_00034_clean.pth', help='Path to network checkpoint')
parser.add_argument('--src_dir', default="../example_imgs", type=str, help='The directory of input images')
parser.add_argument('--result_dir', default='../demo_out', help='Where to export the output data')
parser.add_argument('--batch_size', default=16, type=int)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def run_demo(args):
    """Run evaluation on the datasets and metrics we report in the paper. """

    os.makedirs(args.result_dir, exist_ok=True)

    model = load_model_from_disk(args.checkpoint, True, device)
    batch_size = args.batch_size

    # Load DataLoader
    dataset = DemoDataset(args.src_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)  # , num_workers=num_workers)

    # Store smal parameters
    smal_pose = np.zeros((len(dataset), 105))
    smal_betas = np.zeros((len(dataset), 26))
    smal_camera = np.zeros((len(dataset), 3))
    smal_joints3d = np.zeros((len(dataset), 22, 3))
    smal_imgname = []
    smal_has_bbox = []

    # Iterate over the entire dataset
    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))
    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            preds = model(batch, eval=False)

            # make sure we dont overwrite something
            assert not any(k in preds for k in batch.keys())
            preds.update(batch)  # merge everything into one big dict

        curr_batch_size = preds['img'].shape[0]

        smal_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smal_betas[step * batch_size:step * batch_size + curr_batch_size] = preds['betas'].data.cpu().numpy()
        smal_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()
        smal_joints3d[step * batch_size:step * batch_size + curr_batch_size] = preds['joints_3d'].data.cpu().numpy()

        output_figs = np.transpose(
            Visualizer.generate_demo_output(preds).data.cpu().numpy(),
            (0, 1, 3, 4, 2))

        for img_id in range(len(preds['imgname'])):
            imgname = preds['imgname'][img_id].replace("\\", "/")  # always keep in / format
            output_fig_list = output_figs[img_id]

            path_parts = imgname.split('/')
            smal_imgname.append("{0}/{1}".format(path_parts[-2], path_parts[-1]))
            path_suffix = "{0}_{1}".format(path_parts[-2], path_parts[-1])
            img_file = os.path.join(args.result_dir, path_suffix)
            output_fig = np.hstack(output_fig_list)
            smal_has_bbox.append(preds['has_bbox'][img_id])
            cv2.imwrite(img_file, output_fig[:, :, ::-1] * 255.0)

            npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])
            np.savez_compressed(npz_file,
                                imgname=preds['imgname'][img_id],
                                pose=preds['pose'][img_id].data.cpu().numpy(),
                                betas=preds['betas'][img_id].data.cpu().numpy(),
                                camera=preds['camera'][img_id].data.cpu().numpy(),
                                trans=preds['trans'][img_id].data.cpu().numpy(),
                                has_bbox=preds['has_bbox'][img_id],

                                )

    # Save reconstructions to a file for further processing
    param_file = os.path.join(args.result_dir, 'params.npz')
    np.savez(param_file,
             pose=smal_pose,
             betas=smal_betas,
             camera=smal_camera,
             joints3d=smal_joints3d,
             imgname=smal_imgname,
             has_bbox=smal_has_bbox)

    print("--> Exported param file: {0}".format(param_file))
    print('*** FINISHED ***')


def load_model_from_disk(model_path, device):
    model = Model(device)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    if model_path is not None:
        print("found previous model %s" % model_path)
        print("   -> resuming")
        model_state_dict = torch.load(model_path)

        own_state = model.state_dict()
        for name, param in model_state_dict.items():
            try:
                own_state[name].copy_(param)
            except:
                print("Unable to load: {0}".format(name))
    else:
        print('model_path is none')

        print("LOADED")

    return model


if __name__ == '__main__':
    args = parser.parse_args()
    run_demo(args)

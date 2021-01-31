import sys
sys.path.append("../")

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm
import glob
from shutil import rmtree

from dataset_dogs.base_dataset import BaseDataset
from   model import Model
from global_utils.helpers.visualize import Visualizer
import global_utils.config as config
import torch.nn as nn

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='../data/results', help='Where to export the SMAL fits')
# parser.add_argument('--checkpoint', default='../data/pretrained/model_epoch_00000999.pth', help='Path to network checkpoint')
parser.add_argument('--checkpoint', default='../data/pretrained/3501_00034_betas_v4.pth', help='Path to network checkpoint')
parser.add_argument('--dataset', default='stanford', choices=['stanford', 'animal_pose'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')
parser.add_argument('--shape_family_id', default=-1, type=int, help='Shape family to use')
parser.add_argument('--gpu_ids', default="0", type=str, help='GPUs to use. Format as string, e.g. "0,1,2')
parser.add_argument('--param_dir', default="NONE", type=str, help='Exported parameter folder to load')


def run_evaluation(model, dataset, device, result_dir,
                   batch_size=16,
                   num_workers=0, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    # Transfer model to the GPU
    model.to(device)

    save_results = result_dir is not None

    # Create dataloader for the dataset
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers)

    pck = np.zeros((len(dataset)))
    pck_by_part = {group:np.zeros((len(dataset))) for group in config.KEYPOINT_GROUPS}
    acc_sil_2d = np.zeros(len(dataset))

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 105))
    smpl_betas = np.zeros((len(dataset), 26))
    smpl_camera = np.zeros((len(dataset), 3))
    smpl_imgname = []

    # Iterate over the entire dataset
    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))
    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            preds = model(batch)

            # make sure we dont overwrite something
            assert not any( k in preds for k in batch.keys() )    
            preds.update(batch) # merge everything into one big dict

        curr_batch_size = preds['img'].shape[0]

        pck[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
        acc_sil_2d[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
        smpl_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :preds['betas'].shape[1]] = preds['betas'].data.cpu().numpy()
        smpl_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()

        for part in pck_by_part:
            pck_by_part[part][step * batch_size:step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()

        tqdm_iterator.desc = "PCK: {0:.2f}, IOU: {1:.2f}".format(
            pck[:(step * batch_size + curr_batch_size)].mean(),
            acc_sil_2d[:(step * batch_size + curr_batch_size)].mean())
        tqdm_iterator.update()

        output_figs = np.transpose(
            Visualizer.generate_output_figures(preds).data.cpu().numpy(), 
            (0, 1, 3, 4, 2))
        
        for img_id in range(len(preds['imgname'])):
            imgname = preds['imgname'][img_id]
            output_fig_list = output_figs[img_id]

            path_parts = imgname.split('/')
            path_suffix = "{0}_{1}".format(path_parts[-2], path_parts[-1])
            img_file = os.path.join(result_dir, path_suffix)
            output_fig = np.hstack(output_fig_list)
            smpl_imgname.append(path_suffix)
            npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])

            if save_results:
                cv2.imwrite(img_file, output_fig[:, :, ::-1] * 255.0)
                np.savez_compressed(npz_file,
                    imgname=preds['imgname'][img_id],
                    pose=preds['pose'][img_id].data.cpu().numpy(),
                    betas=preds['betas'][img_id].data.cpu().numpy(),
                    camera=preds['camera'][img_id].data.cpu().numpy(),
                    trans=preds['trans'][img_id].data.cpu().numpy(),
                    acc_PCK=preds['acc_PCK'][img_id].data.cpu().numpy(),
                    acc_SIL_2D=preds['acc_IOU'][img_id].data.cpu().numpy(),
                    **{f'{part}_PCK':preds[f'{part}_PCK'].data.cpu().numpy() for part in pck_by_part}
                )

    # Print final results during evaluation

    # pck_data = np.concatenate(
    # [
    #     np.array(smpl_imgname), 
    #     np.array(pck).astype(str)
    # ])

    # np.savetxt("../data/debug/pck.csv", pck_data, delimiter=",", fmt="%s")
    
    report = f"""*** Final Results ***

    SIL IOU 2D: {np.nanmean(acc_sil_2d):.5f}
    PCK 2D: {np.nanmean(pck):.5f}"""

    report_str = f"{np.nanmean(acc_sil_2d):.5f},{np.nanmean(pck):.5f},"

    for part in pck_by_part:
        report += f'\n   {part} PCK 2D: {np.nanmean(pck_by_part[part]):.5f}'
        report_str += f"{np.nanmean(pck_by_part[part]):.5f},"

    print(report)
    print(report_str)

    # save report to file
    with open(os.path.join(result_dir, '_report.txt'), 'w') as outfile:
        print(report, file=outfile)

def load_model_from_disk(model_path, shape_family_id, load_from_disk, device):
    model = Model(device, shape_family_id, load_from_disk)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    if model_path is not None:
        print( "found previous model %s" % model_path )
        print( "   -> resuming" )
        model_state_dict = torch.load(model_path)

        own_state = model.state_dict()
        for name, param in model_state_dict.items():
            try:
                own_state[name].copy_(param)
            except:
                print ("Unable to load: {0}".format(name))
    else:
        print ('model_path is none')

        print ("LOADED")

    return model


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print (os.environ['CUDA_VISIBLE_DEVICES'])

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    assert torch.cuda.device_count() == 1, "Currently only 1 GPU is supported"

    # Create new result output directory
    print ("RESULTS: {0}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print (f"Unable to find: {args.checkpoint}")
    
    load_from_disk = os.path.exists(args.param_dir)

    model = load_model_from_disk(
        args.checkpoint, args.shape_family_id, 
        load_from_disk, device)

    model.eval()

    if load_from_disk:
        dataset = BaseDataset(
            args.dataset, 
            param_dir=args.param_dir,
            is_train=False, 
            use_augmentation=False)

    else:
        # Setup evaluation dataset
        dataset = BaseDataset(
            args.dataset,
            is_train=False, 
            use_augmentation=False)
    
    
    # Run evaluation
    run_evaluation(
        model, dataset, device, args.output_dir,
        batch_size=args.batch_size,
        log_freq=args.log_freq)

    print ("------------------COMPLETED EVALUATION------------------")


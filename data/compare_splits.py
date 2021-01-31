import numpy as np
import json
import os
from pycocotools.mask import decode as decode_RLE
import imageio
import cv2
from tqdm import tqdm

def seg_from_anno(entry):
	"""Given a .json entry, returns the binary mask as a numpy array"""

	rle = {
		"size": [entry['img_height'], entry['img_width']],
		"counts": entry['seg']
	}

	decoded = decode_RLE(rle)
	return decoded


split_path_1 = "data/StanfordExtra_v12/test_stanford_StanfordExtra_v12.npy"
split_path_2 = "/home/bjb10042/projects/data/dogs_v2/splits/val_stanford.npy"

json_path_1 = "data/StanfordExtra_v12/StanfordExtra_v12.json"
json_path_2 = "/home/bjb10042/projects/data/dogs_v2/stanford/keypoints_v101.json"

seg_folder = "/home/bjb10042/projects/data/dogs_v2/stanford/segs"
img_folder = "/home/bjb10042/projects/data/dogs_v2/stanford/images/"

splits_1 = np.load(split_path_1)
splits_2 = np.load(split_path_2)

with open(json_path_1, 'r') as f1:
    json_1 = json.load(f1)

with open(json_path_2, 'r') as f2:
    json_2 = json.load(f2)

seg_errs = 0

for idx_1, idx_2 in tqdm(zip(splits_1, splits_2), total=len(splits_1)):
    if json_1[idx_1]['img_path'] != json_2[idx_2]['img_path']:
        print ("IMAGE PATHS DON'T MATCH")
        print (json_1[idx_1]['img_path'])
        print (json_2[idx_2]['img_path'])

    joints_1 = np.array(json_1[idx_1]['joints'])
    joints_2 = np.array(json_2[idx_2]['joints'])
    joints_2[[18, 19]] = joints_2[[19, 18]]

    if not np.all(joints_1 == joints_2):
        print ("JOINTS DON'T MATCH")
        print (joints_1)
        print (joints_2)

    seg_1 = seg_from_anno(json_1[idx_1]).astype(float)
    seg_2 = np.load(os.path.join(seg_folder, json_2[idx_2]['img_path'].replace(".jpg", ".npy")))
    seg_2 = (seg_2 != 0).astype(float)

    output_img = np.concatenate([seg_1, seg_2], axis = 1)

    imageio.imsave("data/debug/{0}".format(json_1[idx_1]['img_path'].replace("/", "_")), np.dstack([output_img] * 3))

    if not np.all(seg_1 == seg_2):
        print ("SEG DON'T MATCH")
        print (json_1[idx_1]['img_path'])
        print (json_2[idx_2]['img_path'])
        seg_errs += 1

    img_1 = cv2.imread(os.path.join(
        'data/StanfordExtra_v12/images', 
        json_1[idx_1]['img_path']))[:,:,::-1].copy().astype(np.float32)
    
    img_2 = cv2.imread(os.path.join(
        img_folder,
        json_2[idx_2]['img_path']
    ))[:,:,::-1].copy().astype(np.float32)

    if not np.all(img_1 == img_2):
        print ("IMGS DON'T MATCH")
        print (json_1[idx_1]['img_path'])
        print (json_2[idx_2]['img_path'])

print ("DONE")
print (seg_errs)

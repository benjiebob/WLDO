import numpy as np
from pycocotools.mask import decode as decode_RLE
from tqdm import tqdm
import json
import os
import imageio

def seg_from_anno(entry):
	"""Given a .json entry, returns the binary mask as a numpy array"""

	rle = {
		"size": [entry['img_height'], entry['img_width']],
		"counts": entry['seg']
	}

	decoded = decode_RLE(rle)
	return decoded

seg_folder = "/home/bjb10042/projects/data/dogs_v2/animal_pose/segs"

test_path = "data/splits/test_animal_pose.npy"
json_path_1 = "data/animal_pose/animal_pose_data.json"
json_path_2 = "/home/bjb10042/projects/data/dogs_v2/animal_pose/keypoints_v101.json"

split = np.load(test_path)

idx_list = []

with open(json_path_1, 'r') as f1:
    json_1 = json.load(f1)

with open(json_path_2, 'r') as f2:
    json_2 = json.load(f2)

for idx_1 in tqdm(split, total=len(split)):
    imgpath = json_1[idx_1]['img_path']
    joints_1 = np.array(json_1[idx_1]['joints'])

    found = False

    for idx_2, data_2 in enumerate(json_2):
        if data_2['img_path'] == imgpath:
            joints_2 = np.array(json_2[idx_2]['joints'])
            joints_2[[18, 19]] = joints_2[[19, 18]]

            idx_list.append(idx_2)

            if not np.all(joints_1 == joints_2):
                print ("JOINTS DON'T MATCH")
                print (joints_1)
                print (joints_2)

            seg_1 = seg_from_anno(json_1[idx_1]).astype(float)
            seg_2 = np.load(os.path.join(seg_folder, json_2[idx_2]['img_path'].replace(".jpg", ".npy")))
            seg_2 = (seg_2 != 0).astype(float)

            output_img = np.concatenate([seg_1, seg_2], axis = 1)
            imageio.imsave("data/debug_ap/{0}".format(imgpath.replace("/", "_")), np.dstack([output_img] * 3))

            if not np.all(seg_1 == seg_2):
                print ("SEG DON'T MATCH")
                print (json_1[idx_1]['img_path'])
                print (json_2[idx_2]['img_path'])

            found = True

    if not found:
        print ("DID NOT FIND: {0}".format(imgpath))

np.save('data/debug_ap/animal_pose_test_v2.npy', idx_list)

print ("COMPLETED")
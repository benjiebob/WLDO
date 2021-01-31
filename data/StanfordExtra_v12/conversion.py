"""Script for converting train & test splits from keypoints_v101.npy"""
import numpy as np
import json
import os
from pycocotools.mask import encode as encode_to_RLE
from matplotlib import pyplot as plt
from tqdm import tqdm
from csv import DictReader


def seg_to_RLE(seg):
	"""Converts binary segmentation to RLE"""
	if seg is False: return ""
	seg = seg.astype(np.uint8)
	encoded = encode_to_RLE(np.asfortranarray(seg))
	rle_string = encoded['counts'].decode('utf-8')
	return rle_string


def load_json(json_loc):
	with open(json_loc) as infile:
		data = json.load(infile)
	return data

img_dir = r'D:\IIB Project Data\Training data sets\Dogs\Full dog dataset'
seg_dir = r'D:\IIB Project Data\Training data sets\Dogs\Full dog dataset\segments\segs'


# load keypoint labels and colours
with open('../keypoint_definitions.csv') as infile:
	reader = DictReader(infile)
	colours = ["#"+row['Hex colour'] for row in reader]
	labels = [row['Name'] for row in reader]

def append_to(SE_1, SE_2, SE_out):
	"""Finds all elements in SE_2 that are *not* in SE_1. Copies these, along with all elements in SE_1
	to SE_out"""
	SE_1_data = load_json(SE_1)
	SE_2_data = load_json(SE_2)

	out = []
	SE_1_img_paths = {k['img_path'] for k in SE_1_data}
	SE_2_img_paths = {k['img_path'] for k in SE_2_data}

	append_counts = 0
	valid_segs = 0
	errs = []

	# Add all SE_1 to SE_out, converting ear tips & segmentations
	with tqdm(SE_1_data) as tqdm_iterator:
		for n, elem in enumerate(tqdm_iterator):
			out_elem = {k:elem[k] for k in ['img_path', 'img_width', 'img_height', 'img_bbox', 'is_multiple_dogs']}

			# swap ear joints
			joints = elem['joints']
			joints[18], joints[19] = joints[19], joints[18]
			out_elem['joints'] = joints

			seg_loc = os.path.join(seg_dir,elem['img_path'])
			if not os.path.isfile(seg_loc):
				seg_arr = np.zeros((elem['img_height'], elem['img_width'], 3))
				errs.append(elem['img_path'])
			else:
				seg_arr = plt.imread(seg_loc)

				# plt.imshow(plt.imread(os.path.join(img_dir, elem['img_path'])))
				# for idx, (x, y, v) in enumerate(elem['joints']):
				# 	if v == 1:
				# 		plt.scatter([x], [y], c=[colours[idx]], marker="x", s=50)
				# plt.title(elem['is_multiple_dogs'])
				# plt.show()

				valid_segs += 1

			out_elem['seg'] = seg_to_RLE(np.any(seg_arr>0, axis=-1))
			out.append(out_elem)
			tqdm_iterator.set_description(f'Adding all {SE_1}. Appended: {append_counts}. Valid: {valid_segs}')

	# Add all SE_2 not in SE_1
	SE_2_counter = 0
	with tqdm(SE_2_data) as tqdm_iterator:
		for elem in tqdm_iterator:
			if elem['img_path'] not in SE_1_img_paths:
				out.append(elem)
				SE_2_counter += 1
				tqdm_iterator.set_description(f'Adding all {SE_2}. Count: {SE_2_counter}')

	# print(errs)
	with open(SE_out, 'w') as outfile:
		json.dump(out, outfile)


def converter(SE_from, SE_to, split_file):
	"""Converts indices in split_file from the SE_from json file to the SE_to json file"""
	from_idxs = np.load(split_file)

	SE_from_data = load_json(SE_from)
	SE_to_data = load_json(SE_to)

	to_idxs = np.zeros_like(from_idxs)

	SE_to_lookup = {k['img_path']: n for n, k in enumerate(SE_to_data)}

	for i, from_idx in enumerate(from_idxs):
		to_idxs[i] = SE_to_lookup[SE_from_data[from_idx]['img_path']]

	out_name = os.path.split(SE_to)[-1].replace('.json', '')  # get name of SE_to file to save splits with
	out_loc = split_file.replace('.npy', f'_{out_name}.npy')
	np.save(out_loc, to_idxs)
	return out_loc


def make_val_split(SE_file, *split_files):
	"""Saves .npy file with all indices not in any of split files"""
	SE_data = load_json(SE_file)
	size = len(SE_data)
	split_data = [set(np.load(f)) for f in split_files]

	n_val = size - sum(map(len, split_data))  # Number of val entries
	val_idxs = np.zeros(n_val)

	val_idxs = {*range(size)}
	for s in split_data:
		val_idxs -= s

	out_name = os.path.split(SE_file)[-1].replace('.json', '')  # get name of SE_to file to save splits with
	out_loc = f'val_stanford_{out_name}.npy'
	np.save(out_loc, np.fromiter(val_idxs, dtype=np.int))
	return out_loc


def checker(SE_from, SE_to, split_file_from, split_file_to):
	"""Checks that all indices split_file_from -> SE_from match with
	 all indices SE_to -> split_file_to"""

	from_idxs = np.load(split_file_from)
	to_idxs = np.load(split_file_to)

	SE_from_data = load_json(SE_from)
	SE_to_data = load_json(SE_to)

	with tqdm(zip(from_idxs, to_idxs)) as tqdm_iterator:
		tqdm_iterator.set_description("Checking: ")
		for from_idx, to_idx in tqdm_iterator:
			from_path = SE_from_data[from_idx]['img_path']
			to_path = SE_to_data[to_idx]['img_path']
			assert from_path == to_path, f"Paths do not match! {from_path}, {to_path}"

			# checks that joints match too
			from_joints = SE_from_data[from_idx]['joints']
			to_joints = SE_to_data[to_idx]['joints']
			assert from_joints[:18] == to_joints[:18], f"First 18 joints do not match! {from_path}"
			assert from_joints[18:20] == to_joints[18:20][::-1], f"Ear tips do not match! {from_path}. j18 = {from_joints[18:20]}, {to_joints[18:20]}"
			assert from_joints[20:] == to_joints[20:], f"Last 4 joints do not match! {from_path}"

	print("Check passed!")


def check_all_splits(SE_file, *splits):
	"""Checks that all indices are contained in one of the split files, and that there is no overlap"""

	# all indices are in one split file:
	n_idxs = len(load_json(SE_file))
	split_idx_sets = [{item for item in np.load(splitlist)} for splitlist in splits]
	combined_idx_set = {item for splitlist in splits for item in np.load(splitlist)}

	[print(f'{fname}: {N} entries') for fname, N in zip(splits, map(len, split_idx_sets))]

	assert {*range(n_idxs)} == combined_idx_set, "Not all indices are found in splits"
	assert len(combined_idx_set) == sum(map(len, split_idx_sets)), "Some indices are repeated!"

	print("Check passed!")


if __name__ == '__main__':
	src = '_converter'
	SE_from = 'keypoints_v101.json'
	SE_to = 'StanfordExtra_v1_full.json'
	SE_out = 'StanfordExtra_v12.json'
	train_file = 'train_stanford.npy'
	test_file = 'test_stanford.npy'

	# Add entries from keypoints_v101 to SE_v2
	append_to(SE_from, SE_to, SE_out)

	# produce corrected train/test/val splits
	train_file_fixed = converter(SE_from, SE_out, train_file)
	test_file_fixed = converter(SE_from, SE_out, test_file)
	val_file_fixed = make_val_split(SE_out, train_file_fixed, test_file_fixed)
	#
	# # check these split files are consistent
	checker(SE_from, SE_out, train_file, train_file_fixed)
	checker(SE_from, SE_out, test_file, test_file_fixed)
	check_all_splits(SE_out, train_file_fixed, test_file_fixed, val_file_fixed)
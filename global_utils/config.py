"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join
import os

# Define paths to each dataset
BASE_FOLDER = '../data'
CODE_DIR = os.path.dirname(os.getcwd())

# Output folder to save test/train npz files
DATASET_NPZ_PATH = join(BASE_FOLDER, 'splits')

# Path to test/train npy files
DATASET_FILES = [
  {
    'stanford': join(BASE_FOLDER, 'StanfordExtra_v12', 'test_stanford_StanfordExtra_v12.npy'),
    'animal_pose' : join(BASE_FOLDER, 'animal_pose', 'test_animal_pose.npy'),
  },
]

DATASET_FOLDERS = {
  'stanford' : join(BASE_FOLDER, 'StanfordExtra_v12'),
  'animal_pose' : join(BASE_FOLDER, 'animal_pose'),
}

JSON_NAME = {
    'stanford': 'StanfordExtra_v12.json', # the latest version of the StanfordExtra dataset
    'animal_pose': 'animal_pose_data.json'
}

BREEDS_CSV = join(BASE_FOLDER, 'breeds.csv')

EM_DATASET_NAME = "stanford" # the dataset to learn the EM prior on

data_path = join(CODE_DIR, 'data')

# SMAL
SMAL_FILE = join(data_path, 'smal', 'my_smpl_00781_4_all.pkl')
SMAL_DATA_FILE = join(data_path, 'smal', 'my_smpl_data_00781_4_all.pkl')
SMAL_UV_FILE = join(data_path, 'smal', 'my_smpl_00781_4_all_template_w_tex_uv_001.pkl')
SMAL_SYM_FILE = join(data_path, 'smal', 'symIdx.pkl')
SHAPE_FAMILY_ID = 1 # the dog shape family

# PRIORS
WALKING_PRIOR_FILE = join(data_path, 'priors', 'walking_toy_symmetric_pose_prior_with_cov_35parts.pkl')
UNITY_POSE_PRIOR = join(data_path, 'priors', 'unity_pose_prior_with_cov_35parts.pkl')
UNITY_SHAPE_PRIOR = join(data_path, 'priors', 'unity_betas.npz')
SMAL_DOG_TOY_IDS = [0, 1, 2]

# DATALOADER
IMG_RES = 224

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

# RENDERER
PROJECTION = 'perspective'
NORM_F0 = 2700.0
NORM_F = 2700.0
NORM_Z = 20.0

MESH_COLOR = [0, 172, 223]

# MESH_NET
NZ_FEAT = 100

# ASSOCIATING SMAL TO ANNOTATED JOINTS
MODEL_JOINTS = [
  14, 13, 12, # left front (0, 1, 2)
  24, 23, 22, # left rear (3, 4, 5)
  10, 9, 8, # right front (6, 7, 8)
  20, 19, 18, # right rear (9, 10, 11)
  25, 31, # tail start -> end (12, 13)
  34, 33, # right ear, left ear (14, 15)
  35, 36, # nose, chin (16, 17)
  37, 38] # right tip, left tip (18, 19)

EVAL_KEYPOINTS = [
  0, 1, 2, # left front
  3, 4, 5, # left rear
  6, 7, 8, # right front
  9, 10, 11, # right rear
  12, 13, # tail start -> end
  14, 15, # left ear, right ear
  16, 17, # nose, chin
  18, 19] # left tip, right tip

KEYPOINT_GROUPS = {
  'legs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # legs
  'tail': [12, 13], # tail
  'ears': [14, 15, 18, 19], # ears
  'face': [16, 17] # face
}

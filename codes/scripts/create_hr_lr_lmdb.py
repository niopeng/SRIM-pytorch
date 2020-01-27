import sys
import os.path
import glob
import pickle
import lmdb
import cv2
import torch
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar

# configurations
img_folder = '/home/nio/data/n01531178/*'  # glob matching pattern
lmdb_save_path = '/home/nio/data/n01531178.lmdb'  # must end with .lmdb
down_lmdb_save_path = '/home/nio/data/n01531178_down_8.lmdb'

img_list = sorted(glob.glob(img_folder))
dataset = []

scale = 1. / 8
down_dataset = []

data_size = 0
down_data_size = 0

print('Read images...')
pbar = ProgressBar(len(img_list))
for i, v in enumerate(img_list):
    pbar.update('Read {}'.format(v))
    img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dataset.append(img)
    down_img = cv2.resize(img, dsize=(int(img.shape[0] * scale), int(img.shape[1] * scale)), interpolation=cv2.INTER_CUBIC)
    down_dataset.append(down_img)
    data_size += img.nbytes
    down_data_size += down_img.nbytes
env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
down_env = lmdb.open(down_lmdb_save_path, map_size=down_data_size * 10)
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

pbar = ProgressBar(len(img_list))
with env.begin(write=True) as txn:  # txn is a Transaction object
    for i, v in enumerate(img_list):
        pbar.update('Write {}'.format(v))
        base_name = os.path.splitext(os.path.basename(v))[0]
        key = base_name.encode('ascii')
        data = dataset[i]
        if dataset[i].ndim == 2:
            H, W = dataset[i].shape
            C = 1
        else:
            H, W, C = dataset[i].shape
        meta_key = (base_name + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(H, W, C)
        # The encode is only essential in Python 3
        txn.put(key, data)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing lmdb.')

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys cache.')


pbar = ProgressBar(len(img_list))
with down_env.begin(write=True) as txn:  # txn is a Transaction object
    for i, v in enumerate(img_list):
        pbar.update('Write {}'.format(v))
        base_name = os.path.splitext(os.path.basename(v))[0]
        key = base_name.encode('ascii')
        data = down_dataset[i]
        if down_dataset[i].ndim == 2:
            H, W = down_dataset[i].shape
            C = 1
        else:
            H, W, C = down_dataset[i].shape
        meta_key = (base_name + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(H, W, C)
        # The encode is only essential in Python 3
        txn.put(key, data.astype(np.uint8))
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing down_lmdb.')

# create keys cache
keys_cache_file = os.path.join(down_lmdb_save_path, '_keys_cache.p')
down_env = lmdb.open(down_lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with down_env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating down_lmdb keys cache.')

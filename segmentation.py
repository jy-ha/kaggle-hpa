!curl ipinfo.io  # us-central과 같은 대륙

!echo "deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
!sudo apt-get -y -q update
!sudo apt-get -y -q install gcsfuse
!pip install tensorflow_addons

!head /proc/cpuinfo


import os
from itertools import groupby
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import cv2
from multiprocessing import Pool
import pickle
import random
#from google.cloud import storage

from google.colab import auth
auth.authenticate_user()

print(os.cpu_count())
!mkdir -p data
!gcsfuse --implicit-dirs --limit-bytes-per-sec -1 --limit-ops-per-sec -1 --o allow_other --file-mode=777 --dir-mode=777 blonix-data data

cell_mask_dir = './data/input/hpa-mask/hpa_cell_mask'
ROOT = './data/hpa-data/input/hpa-single-cell-image-classification'
train_or_test = 'train'
IMAGE_SIZE = 224

df = pd.read_csv(os.path.join(ROOT, 'train.csv'))

debug = False
if debug:
    df = df[:4]


# image loader
def read_img(image_id, color, train_or_test='train', image_size=None):
    filename = f'{ROOT}/{train_or_test}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max()
        img = (img / 255).astype('uint8')
    return img


def generate_cell_data(image_idx):
    image_id = df.iloc[image_idx].ID
    class_id = df.iloc[image_idx].Label

    data_list = []
    red = read_img(image_id, "red", train_or_test)
    green = read_img(image_id, "green", train_or_test)
    blue = read_img(image_id, "blue", train_or_test)
    yellow = read_img(image_id, "yellow", train_or_test)

    mask = np.load(f'{cell_mask_dir}/{image_id}.npz')['arr_0']
    mask_ids = np.unique(mask)
    for val in mask_ids:
        if val == 0:
            continue
        binary_mask = np.where(mask == val, 1, 0).astype(np.uint8)
        pixels = cv2.findNonZero(binary_mask)
        x, y, w, h = cv2.boundingRect(pixels)
        binary_mask = binary_mask[y:y + h, x:x + w]
        masked_img_r = red[y:y + h, x:x + w] * binary_mask
        masked_img_g = green[y:y + h, x:x + w] * binary_mask
        masked_img_b = blue[y:y + h, x:x + w] * binary_mask
        masked_img_y = yellow[y:y + h, x:x + w] * binary_mask

        masked_img_r = (masked_img_r + masked_img_y) / 2
        masked_img_g = (masked_img_g + masked_img_y) / 2

        # stacked_img = np.transpose(np.array([masked_img_r, masked_img_g, masked_img_b, masked_img_y]), (1,2,0))
        stacked_img = np.transpose(np.array([masked_img_r, masked_img_g, masked_img_b]), (1, 2, 0))

        ratio = float(IMAGE_SIZE) / float(max(h, w))
        stacked_img = cv2.resize(stacked_img, (int(w * ratio), int(h * ratio)))
        delta_w = IMAGE_SIZE - int(w * ratio)
        delta_h = IMAGE_SIZE - int(h * ratio)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        pad_color = [0, 0, 0]
        stacked_img = cv2.copyMakeBorder(stacked_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        result, jpg_img = cv2.imencode('.jpg', stacked_img, ENCODE_PARAM)

        data = {
            'img': jpg_img,
            'id': image_id,
            'label': class_id
        }

        data_list.append(data)
        del stacked_img, masked_img_r, binary_mask, masked_img_g, masked_img_b, masked_img_y
    del red, green, blue, yellow, mask
    return data_list, image_id, image_idx

# count label number & undercut labels
print(df.groupby('Label')['ID'].nunique().sort_values(ascending=False).head(10))

print('----')
cutoff = 300
df = df.groupby('Label').head(cutoff)
print(df.head())

# create df for each cells
#columns = ['id_cell', 'label']
#df_cell = pd.DataFrame(columns=columns)
#print(df_cell)


MAX_GREEN = 64  # filter out dark green
MAX_DUMP_GROUP = 10000  # grouping imgs dumping to one file
MAX_READ_GROUP = 2000  # number of reading file at once
ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
MAX_THRE = 16

df = df[12000:]

p = Pool(processes=MAX_THRE)
cnt = 25
len_df = len(df)
data_list_all = []
for i in tqdm(range(len(df))):
    data_list, d_id, idx = generate_cell_data(i)
    if idx == None:
        continue

    # for cell_idx, data in enumerate(data_list):
    #    df_cell.append({'id_cell':d_id+str(cell_idx), 'label':data['label']})

    data_list_all.extend(data_list)
    del data_list
    if ((idx + 1) % MAX_READ_GROUP == 0) or (idx + 1 == len_df):
        random.shuffle(data_list_all)
        dump_counter = 0
        for i in range(int(len(data_list_all) / MAX_DUMP_GROUP)):
            with open(f'./data/work/seg_v6/dump_{cnt}', 'wb') as f:
                pickle.dump(data_list_all[i * MAX_DUMP_GROUP: (i + 1) * MAX_DUMP_GROUP], f, pickle.HIGHEST_PROTOCOL)
            dump_counter += 1
            cnt += 1
        # with open(f'/home/jupyter/data/work/seg_v6/dump_{cnt}', 'wb') as f:
        #    pickle.dump(data_list_all[dump_counter*MAX_DUMP_GROUP:], f, pickle.HIGHEST_PROTOCOL)
        del data_list_all[:dump_counter * MAX_DUMP_GROUP]
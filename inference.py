import os

os.system("pip install ../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl")
os.system("pip install ../input/hpapytorchzoozip/pytorch_zoo-master")
os.system("pip install ../input/hpacellsegmentatorraman/HPA-Cell-Segmentation")
###!pip install "../input/hpacellsegmentatormaster/HPA-Cell-Segmentation-master"

import pandas as pd
import numpy as np
import gc
import cv2
# import imageio
# from tqdm.notebook import tqdm
from tqdm import tqdm
# from itertools import groupby
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask
import matplotlib.pyplot as plt
import base64
import typing as t
import zlib
import random

random.seed(0)

import tensorflow as tf
from hpacellseg.cellsegmentator import *
from hpacellseg import cellsegmentator, utils


train_or_test = 'test'
ROOT = '/kaggle/input/hpa-single-cell-image-classification'
model_path = '/kaggle/input/hpa-model-v1/model-v1'
mask_path = '/kaggle/working/masks/'
os.system("mkdir -p /kaggle/working/masks/")

IMAGE_SIZE = 128
THRESHOLDS_METRICS = 0.5
labels = np.array(
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
df = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)
    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def read_img(image_id, color, train_or_test='train', image_size=None):
    filename = f'{ROOT}/{train_or_test}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.dtype == 'uint16':
        img = (img / 256).astype('uint8')
    return img


def all_cell_segmentation_old(image_ids, image_size=None):
    mt = [f'/kaggle/input/hpa-single-cell-image-classification/test/{image_id}_red.png' for image_id in image_ids]
    er = [f'/kaggle/input/hpa-single-cell-image-classification/test/{image_id}_yellow.png' for image_id in image_ids]
    nu = [f'/kaggle/input/hpa-single-cell-image-classification/test/{image_id}_blue.png' for image_id in image_ids]
    images = [mt, er, nu]

    NUC_MODEL = "/kaggle/input/hpacellsegmentatormodelweights/dpn_unet_nuclei_v1.pth"
    CELL_MODEL = "/kaggle/input/hpacellsegmentatormodelweights/dpn_unet_cell_3ch_v1.pth"
    segmentator = cellsegmentator.CellSegmentator(
        NUC_MODEL,
        CELL_MODEL,
        scale_factor=0.25,
        device="cpu",  # cuda
        padding=True,
        multi_channel_model=True,
    )
    nuc_segmentations = segmentator.pred_nuclei(images[2])
    cell_segmentations = segmentator.pred_cells(images)

    for i, pred in enumerate(tqdm(cell_segmentations)):
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
        # FOVname = os.path.basename(mt[i]).replace('red','predictedmask')
        # imageio.imwrite(os.path.join(mask_path,FOVname), cell_mask)

    del nuc_segmentations, cell_segmentations
    del nuclei_mask, cell_mask
    del segmentator
    gc.collect()


def all_cell_segmentation(image_ids, image_size=None):
    NUC_MODEL = "../input/hpacellsegmentatormodelweights/dpn_unet_nuclei_v1.pth"
    CELL_MODEL = "../input/hpacellsegmentatormodelweights/dpn_unet_cell_3ch_v1.pth"
    segmentator_even_faster = cellsegmentator.CellSegmentator(
        NUC_MODEL,
        CELL_MODEL,
        device="cuda",
        multi_channel_model=True,
    )

    cell_masks = []
    seg_batchsize = 24
    for i in tqdm(range(0, len(image_ids), seg_batchsize)):
        start = i
        end = min(len(image_ids), start + seg_batchsize)

        blue_batch = []
        rgb_batch = []
        id_list = image_ids[start:end]
        for ii in id_list:
            r = cv2.imread(f'/kaggle/input/hpa-single-cell-image-classification/test/{ii}_red.png', 0)
            y = cv2.imread(f'/kaggle/input/hpa-single-cell-image-classification/test/{ii}_yellow.png', 0)
            b = cv2.imread(f'/kaggle/input/hpa-single-cell-image-classification/test/{ii}_blue.png', 0)
            b_img = cv2.resize(b, (512, 512))
            rgb_img = cv2.resize(np.stack((r, y, b), axis=2), (512, 512))
            blue_batch.append(b_img / 255.)
            rgb_batch.append(rgb_img / 255.)

        # t_start = time.time()
        nuc_segmentations = segmentator_even_faster.pred_nuclei(blue_batch)
        # print("time1 :", time.time() - t_start)
        cell_segmentations = segmentator_even_faster.pred_cells(rgb_batch, precombined=True)
        # print("time2 :", time.time() - t_start)

        for data_id, nuc_seg, cell_seg in zip(id_list, nuc_segmentations, cell_segmentations):
            _, cell_mask = utils.label_cell(nuc_seg, cell_seg)
            # cv2.imwrite(os.path.join(mask_path,f'{data_id}_predictedmask.png'), cell_mask)
            # imageio.imwrite(os.path.join(mask_path,f'{data_id}_predictedmask.png'), cell_mask)
            cell_masks.append(cell_mask)
    return cell_masks


print('Start')
cell_masks = all_cell_segmentation(df.ID.tolist())
# all_cell_segmentation(df.ID.tolist())

print('Load Prediction Model')
loaded_model = tf.saved_model.load("../input/hpamodelv1/model_v1")

print('Start Prediction')
# fig, ax = plt.subplots(3, 2, figsize=(20,50))
with open('submission.csv', 'w') as outf:
    print('ID,ImageWidth,ImageHeight,PredictionString', file=outf)
    for idx in tqdm(range(len(df))):
        image_id = df.iloc[idx].ID
        red = read_img(image_id, "red", train_or_test)
        green = read_img(image_id, "green", train_or_test)
        blue = read_img(image_id, "blue", train_or_test)
        # mask = cv2.imread(f'{mask_path}/{image_id}_predictedmask.png', cv2.IMREAD_UNCHANGED)
        mask = cell_masks[idx]
        mask = cv2.resize(mask, (red.shape[0], red.shape[1]), interpolation=cv2.INTER_NEAREST)

        cell_imgs = []
        pred_strs = []
        rles = []
        mask_ids = np.unique(mask)
        for val in mask_ids:
            if val == 0:
                continue
            binary_mask = np.where(mask == val, 1, 0).astype(np.bool)
            rle = encode_binary_mask(binary_mask)
            binary_mask = binary_mask.astype(np.uint8)
            pixels = cv2.findNonZero(binary_mask)
            x, y, w, h = cv2.boundingRect(pixels)
            binary_mask = binary_mask[y:y + h, x:x + w]
            masked_img_r = red[y:y + h, x:x + w] * binary_mask
            masked_img_g = green[y:y + h, x:x + w] * binary_mask
            masked_img_b = blue[y:y + h, x:x + w] * binary_mask
            # masked_img_y = yellow[y:y+h, x:x+w] * binary_mask

            # cell_image = np.stack([masked_img_r, masked_img_g, masked_img_b], axis=2)
            # cell_image = cv2.resize(cell_image, (512, 512))
            # ax[idx, 0].imshow(cell_image)
            # ax[idx, 1].imshow(binary_mask)

            # img = np.transpose(np.array([masked_img_r, masked_img_g, masked_img_b, masked_img_y]), (1,2,0))
            img = np.transpose(np.array([masked_img_r, masked_img_g, masked_img_b]), (1, 2, 0))
            # img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = tf.cast(img, tf.float32)
            img = img / 255.
            img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
            cell_imgs.append(img)
            rles.append(rle)

        predictions = loaded_model(cell_imgs, training=False)
        inference = tf.math.equal(predictions, tf.math.reduce_max(predictions, axis=1, keepdims=True))
        inference = tf.bitwise.bitwise_or(tf.cast(inference, tf.int8),
                                          tf.cast(predictions > THRESHOLDS_METRICS, tf.int8))

        for idx in range(len(inference)):
            class_id_list = []
            for i, proba in enumerate(inference[idx]):
                if proba == 1:
                    class_id_list.append(labels[i])
            class_id = '|'.join(class_id_list)

            print()
            pred_strs.append(f'{class_id} 1.0 {rles[idx]}')

        print(f'{image_id},{red.shape[0]},{red.shape[1]},{" ".join(pred_strs)}', file=outf)
        # print(f'{image_id},{IMAGE_SIZE},{IMAGE_SIZE},{" ".join(pred_strs)}')
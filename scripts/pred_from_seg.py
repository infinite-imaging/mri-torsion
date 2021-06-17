import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy
numpy.random.seed(0)

from torsion.knee import calc_knee
from torsion.hip import calc_hip
from torsion.ankle import calc_ankle
from torsion.utils import get_mask, write_image, sep_seg

import os
import shutil
import numpy as np
import pydicom
import nibabel as nib
from skimage.measure import label
import time
import json
import argparse


files = []


def get_data_seg(root):

    data = nib.load(root)
    affine = data.affine
    img = data.get_fdata()
    img = np.swapaxes(img, 0, 2)

    return img, affine


def get_largest_CC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)
    largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint16)
    return largestCC


def save_nifti(data, affine, filename):
    data = np.swapaxes(data, 0, 2)
    img = nib.Nifti1Image(data, affine)
    img.header.get_xyzt_units()
    img.to_filename(os.path.join(OUTPUT_PATH, filename))


for i in files:
    time_start = time.time()
    ROOT = ''
    OUTPUT_PATH = ''

    INPUT_PATH = ROOT + OUTPUT_PATH + i['pat']
    INPUT_PATH_HUEFTE_L = INPUT_PATH + '/hip_left.nii.gz'
    INPUT_PATH_HUEFTE_R = INPUT_PATH + '/hip_right.nii.gz'
    INPUT_PATH_KNIE_L = INPUT_PATH + '/knee_left.nii.gz'
    INPUT_PATH_KNIE_R = INPUT_PATH + '/knee_right.nii.gz'
    INPUT_PATH_KNOECHEL_L = INPUT_PATH + '/ankle_left.nii.gz'
    INPUT_PATH_KNOECHEL_R = INPUT_PATH + '/ankle_right.nii.gz'

    OUTPUT_PATH = INPUT_PATH + '/torsion'
    OUTFILE = INPUT_PATH + '/torsion/output.txt'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    img = nib.load(INPUT_PATH_HUEFTE_L)
    if abs(img.affine[0, 0]) == abs(img.affine[1, 1]):
        z_ratio = abs(img.affine[2, 2]) / abs(img.affine[0, 0])
        spacing = (abs(img.affine[0, 0]), abs(img.affine[1, 1]), abs(img.affine[2, 2]))

    pred_hl, affine_l = get_data_seg(INPUT_PATH_HUEFTE_L)
    pred_hr, affine_r = get_data_seg(INPUT_PATH_HUEFTE_R)

    mask_hr, p1_hr, p2_hr = calc_hip(get_largest_CC(pred_hr), z_ratio)
    mask_hl, p1_hl, p2_hl = calc_hip(get_largest_CC(pred_hl), z_ratio)

    tors_hr = np.arctan((p2_hr[1]-p1_hr[1])/(p1_hr[2]-p2_hr[2]))*180/np.pi
    tors_hl = np.arctan((p2_hl[1]-p1_hl[1])/(p2_hl[2]-p1_hl[2]))*180/np.pi
    with open(OUTFILE, 'a') as f:
        print("Torsion Femur Hip: right: %4.2f°, left: %4.2f°" % (tors_hr, tors_hl), file=f)

    mask_hip = mask_hl + mask_hr

    save_nifti(mask_hip, affine_l, 'hip_ref.nii.gz')

    pred_kl, affine_l = get_data_seg(INPUT_PATH_KNIE_L)
    pred_kr, affine_r = get_data_seg(INPUT_PATH_KNIE_R)
    pred_kfl = (pred_kl == 1).astype(np.uint16)
    pred_kfr = (pred_kr == 1).astype(np.uint16)

    mask_kfr, p1_kfr, p2_kfr, _ = calc_knee('femur', get_largest_CC(pred_kfr))
    mask_kfl, p1_kfl, p2_kfl, _ = calc_knee('femur', get_largest_CC(pred_kfl))

    if p2_kfl[2] > p1_kfl[2]:
        tors_kfl = np.arctan((p1_kfl[1]-p2_kfl[1])/(p2_kfl[2]-p1_kfl[2]))*180/np.pi
    else:
        tors_kfl = np.arctan((p2_kfl[1]-p1_kfl[1])/(p1_kfl[2]-p2_kfl[2]))*180/np.pi

    if p2_kfr[2] > p1_kfr[2]:
        tors_kfr = np.arctan((p2_kfr[1]-p1_kfr[1])/(p2_kfr[2]-p1_kfr[2]))*180/np.pi
    else:
        tors_kfr = np.arctan((p1_kfr[1]-p2_kfr[1])/(p1_kfr[2]-p2_kfr[2]))*180/np.pi
    with open(OUTFILE, 'a') as f:
        print("Torsion Femur Knee: right: %4.2f°, left: %4.2f°" % (tors_kfr, tors_kfl), file=f)
        print("Femurtorsion: right: %4.2f°, left: %4.2f°" % (tors_hr+tors_kfr, tors_hl+tors_kfl), file=f)

    pred_ktl = (pred_kl == 2).astype(np.uint16)
    pred_ktr = (pred_kr == 2).astype(np.uint16)

    mask_ktr, p1_ktr, p2_ktr, _ = calc_knee('tibia', get_largest_CC(pred_ktr))
    mask_ktl, p1_ktl, p2_ktl, _ = calc_knee('tibia', get_largest_CC(pred_ktl))

    if p2_ktl[2] > p1_ktl[2]:
        tors_ktl = np.arctan((p1_ktl[1]-p2_ktl[1])/(p2_ktl[2]-p1_ktl[2]))*180/np.pi
    else:
        tors_ktl = np.arctan((p2_ktl[1]-p1_ktl[1])/(p1_ktl[2]-p2_ktl[2]))*180/np.pi

    if p2_ktr[2] > p1_ktr[2]:
        tors_ktr = np.arctan((p2_ktr[1]-p1_ktr[1])/(p2_ktr[2]-p1_ktr[2]))*180/np.pi
    else:
        tors_ktr = np.arctan((p1_ktr[1]-p2_ktr[1])/(p1_ktr[2]-p2_ktr[2]))*180/np.pi

    with open(OUTFILE, 'a') as f:
        print("Torsion Tibia Knee: right: %4.2f°, left: %4.2f°" % (tors_ktr, tors_ktl), file=f)
    mask_knee = 4*(mask_kfl + mask_kfr) + 6*(mask_ktl + mask_ktr)

    save_nifti(mask_knee, affine_l, 'knee_ref.nii.gz')
    
    pred_al, affine_l = get_data_seg(INPUT_PATH_KNOECHEL_L)
    pred_ar, affine_r = get_data_seg(INPUT_PATH_KNOECHEL_R)

    pred_afl = (pred_al == 1).astype(np.uint16)
    pred_afr = (pred_ar == 1).astype(np.uint16)

    pred_atl = (pred_al == 2).astype(np.uint16)
    pred_atr = (pred_ar == 2).astype(np.uint16)

    mask_ar, p1_ar, p2_ar = calc_ankle(get_largest_CC(pred_atr), get_largest_CC(pred_afr))
    mask_al, p1_al, p2_al = calc_ankle(get_largest_CC(pred_atl), get_largest_CC(pred_afl))

    tors_al = np.arctan((p2_al[1]-p1_al[1])/(p2_al[2]-p1_al[2]))*180/np.pi
    tors_ar = np.arctan((p2_ar[1]-p1_ar[1])/(p1_ar[2]-p2_ar[2]))*180/np.pi

    with open(OUTFILE, 'a') as f:
        print("Torsion Tibia Ankle: right: %4.2f°, left: %4.2f°" % (tors_ar, tors_al), file=f)
        print("Tibiatorsion: right: %4.2f°, left: %4.2f°" % (tors_ar+tors_ktr, tors_al+tors_ktl), file=f)

    mask_ankle = mask_al + mask_ar

    save_nifti(mask_ankle, affine_l, 'knoechel_ref.nii.gz')

    t = time.time() - time_start
    with open(OUTFILE, 'a') as f:
        print("T: %i\'%i\"" % (t/60, t%60), file=f)

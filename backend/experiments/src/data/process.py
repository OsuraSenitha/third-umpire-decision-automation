import numpy as np
import cv2 as cv
from .analyze import img2ColorMat
import os
import shutil
import json
from tqdm.auto import tqdm
from typing import Tuple

def getBoundingBoxesFromSegmentation(seg_img, labels):
    colorMat = img2ColorMat(seg_img)
    img_h, img_w, _ = seg_img.shape
    annotations = []
    for label_i, label in enumerate(labels):
        color = label["color"]
        thresh = np.isin(colorMat, color).astype(np.uint8)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            no_parent = hierarchy[0][:,3] == -1
            bounding_rects = []
            for cont, no_par in zip(contours, no_parent):
                if no_par:
                    box = cv.boundingRect(cont)
                    x,y,w,h = box
                    x_c, y_c = (x+w/2)/img_w, (y+h/2)/img_h
                    w_c, h_c = w/img_w, h/img_h
                    bounding_rects.append([label_i, x_c, y_c, w_c, h_c])
            annotations.extend(bounding_rects)

    return annotations
    
def splitForObjectDetect(src_dataset_path, train_weight, val_weight, dst_dataset_path):
    subdirs = ["images", "bboxes", "segmentations"]
    splits = ["train", "val"]

    for subdir in subdirs:
        for split in splits:
            path = f"{dst_dataset_path}/{subdir}/{split}"
            if not os.path.exists(path):
                os.makedirs(path)

    src_bbxs_dir = f"{src_dataset_path}/bboxes"
    src_imgs_dir = f"{src_dataset_path}/images"
    src_segs_dir = f"{src_dataset_path}/segmentations"

    dst_bbxs_dir = f"{dst_dataset_path}/bboxes"
    dst_imgs_dir = f"{dst_dataset_path}/images"
    dst_segs_dir = f"{dst_dataset_path}/segmentations"

    lbl_names = os.listdir(src_bbxs_dir)
    tot_weight = train_weight + val_weight
    for i, lbl_name in enumerate(lbl_names):
        dst_split = "val"
        if (i%tot_weight)-train_weight < 0:
            dst_split = "train"
        img_name = os.path.splitext(lbl_name)[0] + ".png"
        src_bbx_path = f"{src_bbxs_dir}/{lbl_name}"
        src_img_path = f"{src_imgs_dir}/{img_name}"
        src_seg_path = f"{src_segs_dir}/{img_name}"
        dst_bbx_path = f"{dst_bbxs_dir}/{dst_split}"
        dst_img_path = f"{dst_imgs_dir}/{dst_split}"
        dst_seg_path = f"{dst_segs_dir}/{dst_split}"

        shutil.copy(src_bbx_path, dst_bbx_path)
        shutil.copy(src_img_path, dst_img_path)
        if os.path.exists(src_seg_path): shutil.copy(src_seg_path, dst_seg_path)

    if len(os.listdir(dst_segs_dir)) == 0:
        shutil.rmtree(dst_segs_dir)

def cvtAnnotationsTXT2LST(txt_cntnt):
    lst = list(map(lambda line: [int(line.split()[0]), *list(map(float, line.split()[1:]))], txt_cntnt.strip().split("\n")))
    return lst

def cvtAnnotationsLST2TXT(lst_cntnt, round_deci):
    if round_deci:
        strn = "\n".join(list(map(lambda box: " ".join([str(int(box[0])), *list(map(lambda num: str(np.round(num, round_deci)).ljust(8, "0"), box[1:]))]), lst_cntnt)))
    else:
        strn = "\n".join(list(map(lambda box: " ".join([str(int(box[0])), *list(map(str, box[1:]))]), lst_cntnt)))
    return strn 
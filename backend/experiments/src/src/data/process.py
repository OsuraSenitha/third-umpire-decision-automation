import numpy as np
import cv2 as cv
from .analyze import img2ColorMat
import os
import shutil
import json
from tqdm.auto import tqdm

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
            # new_img = seg_img.copy()
            for cont, no_par in zip(contours, no_parent):
                if no_par:
                    box = cv.boundingRect(cont)
                    x,y,w,h = box
                    x_c, y_c = (x+w/2)/img_w, (y+h/2)/img_h
                    w_c, h_c = w/img_w, h/img_h
                    bounding_rects.append([label_i, x_c, y_c, w_c, h_c])
                    # new_img = cv.rectangle(seg_img,(x,y),(x+w,y+h),(0,255,0),2)
            annotations.extend(bounding_rects)

    return annotations

def segmentationDS2DetectionDS(data_path, label_names):
    annotations_path = f"{data_path}/annotations"
    classes = {}
    with open(f"{data_path}/classes/classes.json") as handler:
        classes = json.load(handler)
        classes = list(map(lambda obj: {"name":obj["name"], "color": obj["color"]}, classes))
    segments = list(filter(lambda name: "__fuse" in name, os.listdir(f"{data_path}/images")))
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    labels = list(filter(lambda cls: cls["name"] in label_names, classes))
    with tqdm(total=len(segments)) as pbar:
        for i, seg in enumerate(segments):
            img = cv.imread(f"{data_path}/images/{seg}")
            boxes = getBoundingBoxesFromSegmentation(img, labels)
            txt_cntnt = "\n".join(list(map(lambda line: " ".join(list(map(lambda num: str(np.round(num, 6)), line))), boxes)))
            img_f_name = seg[:-11]
            txt_f_name = os.path.splitext(img_f_name)[0]+".txt"
            with open(f"{annotations_path}/{txt_f_name}", "w") as handler:
                handler.write(txt_cntnt)
            
            pbar.update(1)
    
def splitForObjectDetect(src_data_path, train_weight, val_weight, dst_dataset_path):
    subdirs = ["images", "labels"]
    splits = ["train", "val"]

    for subdir in subdirs:
        for split in splits:
            path = f"{dst_dataset_path}/{subdir}/{split}"
            if not os.path.exists(path):
                os.makedirs(path)

    lbls_src_dir = f"{src_data_path}/annotations"
    imgs_src_dir = f"{src_data_path}/images"
    lbl_names = os.listdir(lbls_src_dir)
    tot_weight = train_weight + val_weight
    for i, lbl_name in enumerate(lbl_names):
        dst_split = "val"
        if (i%tot_weight)-train_weight < 0:
            dst_split = "train"
        img_name = os.path.splitext(lbl_name)[0] + ".png"
        shutil.copy(f"{lbls_src_dir}/{lbl_name}", f"{dst_dataset_path}/labels/{dst_split}")
        shutil.copy(f"{imgs_src_dir}/{img_name}", f"{dst_dataset_path}/images/{dst_split}")
        
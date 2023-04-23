import cv2 as cv
import numpy as np
import os
from typing import Tuple
from tqdm.auto import tqdm
import json
from .io import saveAnnotationsFile, readAnnotationsFile, readClassesFile
from .process import splitDataset
import yaml


def getBoundingBoxesFromSegmentation(seg_img, labels):
    annotations = []
    img_h, img_w, _ = seg_img.shape
    for label_i, label in enumerate(labels):
        color = label["color"]
        mask = (
            (seg_img[:, :, 0] == color[0])
            & (seg_img[:, :, 1] == color[1])
            & (seg_img[:, :, 2] == color[2])
        ).astype(np.uint8)
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            no_parent = hierarchy[0][:, 3] == -1
            no_parent_ids = np.where(no_parent)[0]
            contours_filtered = [contours[i] for i in no_parent_ids]
            for cont in contours_filtered:
                box = cv.boundingRect(cont)
                x, y, w, h = box
                x_c, y_c = (x + w / 2) / img_w, (y + h / 2) / img_h
                w_c, h_c = w / img_w, h / img_h
                annotations.append([label_i, x_c, y_c, w_c, h_c])

    return annotations


def getSegmentsFromPNG(
    img, color, normalize=True, epsilon: float = 1.3, threshold_area=100
):
    H, W, _ = img.shape

    mask = (
        (img[:, :, 0] == color[0])
        & (img[:, :, 1] == color[1])
        & (img[:, :, 2] == color[2])
    ).astype(np.uint8)
    contours_raw, hierarchy_raw = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # reduce the number of points
    contours_reduced = []
    for i in range(len(contours_raw)):
        reduced_contour = cv.approxPolyDP(contours_raw[i], epsilon, closed=True)
        contours_reduced.append(reduced_contour)

    if len(contours_reduced) > 0:
        # filter the contours by area
        contours, hierarchy = [], []
        for cnt, hir in zip(contours_reduced, hierarchy_raw[0]):
            area = cv.contourArea(cnt)
            if area > threshold_area:
                contours.append(cnt)
                hierarchy.append(hir)
        hierarchy = np.array(hierarchy)

        if len(contours) > 0:
            no_parent = hierarchy[:, 3] == -1
            no_parent_ids = np.where(no_parent)[0]
            contours = [contours[i].squeeze() for i in no_parent_ids]

            boundaries = []
            for cnt in contours:
                X, Y = cnt[:, 0], cnt[:, 1]

                if normalize:
                    X = X / W
                    Y = Y / H
                bnd = np.zeros(cnt.shape[0] * 2)
                bnd[0::2] = X
                bnd[1::2] = Y
                boundaries.append(bnd.tolist())

            return boundaries

    return []


def makeDarknetSegmentationLabel(seg, label_export_path, classes):
    lines = []
    for lbl, cls in enumerate(classes):
        color = cls["color"]
        boundaries = getSegmentsFromPNG(seg, color)
        pts = [(lbl, *bnd) for bnd in boundaries]
        lines.extend(pts)

    saveAnnotationsFile(lines, label_export_path, None)


def makeDarknetSegmentationDataset(
    src_path,
    context_classes,
    dataset_classes,
    export_path,
    pad_ratio=0.25,
    ds_name: str = "dataset",
    split_weights=None,
):
    src_images_dir = f"{src_path}/images"
    src_segmts_dir = f"{src_path}/segmentation-images"
    src_class_path = f"{src_path}/classes/classes.json"
    ds_file_path = f"{export_path}/{ds_name}.yaml"

    assert os.path.exists(src_images_dir)
    assert os.path.exists(src_segmts_dir)
    assert os.path.exists(src_class_path)

    classes = readClassesFile(
        src_class_path, required_classes=context_classes, format="bgr"
    )
    dataset_classes_obj = list(
        filter(lambda cls: cls["name"] in dataset_classes, classes)
    )
    ds_file_cntnt = {"nc": len(context_classes), "names": context_classes}

    dst_images_root = os.path.join(export_path, "images")
    dst_labels_root = os.path.join(export_path, "labels")
    dst_segmts_root = os.path.join(export_path, "segmentations")
    created_count = 0
    obj_nm_pad = 6

    # count the number of files
    file_count = 0
    for src_images_dir, _, img_lst in os.walk(src_images_dir):
        file_count += len(img_lst)

    print("---- Generatinig dataset ----")
    with tqdm(total=file_count) as pbar:
        for src_images_root, _, img_lst in os.walk(src_images_dir):
            if len(img_lst) > 0:
                trail = src_images_root[len(src_images_dir) :].strip("/")

                dst_segmts_path = os.path.join(dst_segmts_root, trail)
                dst_labels_path = os.path.join(dst_labels_root, trail)
                dst_images_path = os.path.join(dst_images_root, trail)
                os.makedirs(dst_labels_path, exist_ok=True)
                os.makedirs(dst_images_path, exist_ok=True)
                os.makedirs(dst_segmts_path, exist_ok=True)

                src_segmts_root = os.path.join(src_segmts_dir, trail).rstrip("/")

                for img_nm in img_lst:
                    img = cv.imread(f"{src_images_root}/{img_nm}")
                    seg = cv.imread(f"{src_segmts_root}/{img_nm}")

                    if seg is None:
                        pbar.update(1)
                        continue
                    assert img is not None

                    H, W, _ = img.shape

                    bbx = getBoundingBoxesFromSegmentation(seg, classes)
                    if len(bbx) == 0:
                        pbar.update(1)
                        continue

                    for line in bbx:
                        cls, *box = line
                        if context_classes[cls] in dataset_classes:
                            exp_obj_nm = str(created_count).rjust(obj_nm_pad, "0")
                            created_count += 1

                            x, y, w, h = box
                            x, y, w, h = x * W, y * H, w * W, h * H
                            w_pad, h_pad = (1 + pad_ratio) * w, (1 + pad_ratio) * h
                            size = max(w_pad, h_pad)
                            x1, y1 = int(x - size / 2), int(y - size / 2)
                            x2, y2 = int(x + size / 2), int(y + size / 2)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W, x2), min(H, y2)

                            img_trm = img[y1:y2, x1:x2]
                            seg_trm = seg[y1:y2, x1:x2]

                            if max(seg_trm.shape[:2]) > 100:
                                export_segmts_path = (
                                    f"{dst_segmts_path}/{exp_obj_nm}.txt"
                                )
                                export_labels_path = (
                                    f"{dst_labels_path}/{exp_obj_nm}.txt"
                                )
                                export_image_path = (
                                    f"{dst_images_path}/{exp_obj_nm}.png"
                                )
                                makeDarknetSegmentationLabel(
                                    seg_trm, export_segmts_path, classes
                                )
                                makeDarknetSegmentationLabel(
                                    seg_trm, export_labels_path, dataset_classes_obj
                                )
                                cv.imwrite(export_image_path, img_trm)

                    pbar.update(1)
    with open(ds_file_path, "w") as handler:
        yaml.dump(ds_file_cntnt, handler)

    if split_weights is not None:
        train_weight, val_weight = split_weights
        dst_path = f"{export_path}[splitted]"
        ds_file_path = f"{dst_path}/{ds_name}.yaml"
        ds_file_cntnt = {
            "train": f"./datasets/{ds_name}/images/train",
            "val": f"./datasets/{ds_name}/images/val",
            "nc": len(context_classes),
            "names": context_classes,
        }
        splitDataset(
            export_path,
            train_weight,
            val_weight,
            dst_path,
            subdirs=["images", "labels", "segmentations"],
            subdir_exts=["png", "txt", "txt"],
        )
        with open(ds_file_path, "w") as handler:
            yaml.dump(ds_file_cntnt, handler)


def segmentationDS2DetectionDS(
    data_path: str, label_names: Tuple[str] = ["Batsmen", "Ball", "Wicket"]
):
    bboxes_path = f"{data_path}/bboxes"
    classes_file_path = f"{data_path}/classes/classes.json"
    classes = readClassesFile(classes_file_path, label_names, "bgr")
    segments = list(
        filter(lambda name: "__fuse" in name, os.listdir(f"{data_path}/images"))
    )
    if not os.path.exists(bboxes_path):
        os.makedirs(bboxes_path)

    labels = list(filter(lambda cls: cls["name"] in label_names, classes))
    with tqdm(total=len(segments)) as pbar:
        for i, seg in enumerate(segments):
            img = cv.imread(f"{data_path}/images/{seg}")
            boxes = getBoundingBoxesFromSegmentation(img, labels)
            txt_cntnt = "\n".join(
                list(
                    map(
                        lambda line: " ".join(
                            list(map(lambda num: str(np.round(num, 6)), line))
                        ),
                        boxes,
                    )
                )
            )
            img_f_name = seg[:-11]
            txt_f_name = os.path.splitext(img_f_name)[0] + ".txt"
            with open(f"{bboxes_path}/{txt_f_name}", "w") as handler:
                handler.write(txt_cntnt)

            pbar.update(1)

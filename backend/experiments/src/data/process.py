import numpy as np
import os
import shutil
import cv2 as cv
from typing import Tuple, Union


def splitDataset(
    src_dataset_dir,
    train_weight,
    val_weight,
    dst_dataset_dir,
    subdirs=["bboxes", "images", "segmentations"],
    subdir_exts=["txt", "png", "png"],
):
    splits = ["train", "val"]

    for subdir in subdirs:
        for split in splits:
            path = f"{dst_dataset_dir}/{subdir}/{split}"
            if not os.path.exists(path):
                os.makedirs(path)

    src_dirs = list(map(lambda subdir: f"{src_dataset_dir}/{subdir}", subdirs))
    dst_dirs = list(map(lambda subdir: f"{dst_dataset_dir}/{subdir}", subdirs))

    obj_names = os.listdir(src_dirs[0])
    tot_weight = train_weight + val_weight
    for i, obj_name in enumerate(obj_names):
        obj_name = os.path.splitext(obj_name)[0]
        dst_split = "val"
        if (i % tot_weight) - train_weight < 0:
            dst_split = "train"

        for src_dir, dst_dir, ext in zip(src_dirs, dst_dirs, subdir_exts):
            src_path = f"{src_dir}/{obj_name}.{ext}"
            dst_dir = f"{dst_dir}/{dst_split}"
            shutil.copy(src_path, dst_dir)


def cvtAnnotationsTXT2LST(txt_cntnt):
    lst = list(
        map(
            lambda line: [int(line.split()[0]), *list(map(float, line.split()[1:]))],
            txt_cntnt.strip().split("\n"),
        )
    )
    return lst


def cvtAnnotationsLST2TXT(lst_cntnt, round_deci=None):
    if round_deci is not None:
        strn = "\n".join(
            list(
                map(
                    lambda box: " ".join(
                        [
                            str(int(box[0])),
                            *list(
                                map(
                                    lambda num: str(np.round(num, round_deci)).ljust(
                                        8, "0"
                                    ),
                                    box[1:],
                                )
                            ),
                        ]
                    ),
                    lst_cntnt,
                )
            )
        )
    else:
        strn = "\n".join(
            list(
                map(
                    lambda box: " ".join([str(int(box[0])), *list(map(str, box[1:]))]),
                    lst_cntnt,
                )
            )
        )
    return strn


def makeMaskFromSegments(shape: Tuple[int], segments: Union[str, Tuple]) -> np.ndarray:
    if type(segments) == str:
        segments = cvtAnnotationsTXT2LST(segments)
    mask = np.zeros(shape)
    H, W = shape
    new_segments = []
    for segment in segments:
        segment = np.array(segment[1:])
        segment[0::2] *= W
        segment[1::2] *= H
        segment = segment.reshape(-1, 2).astype(np.int64)
        new_segments.append(segment)

    cv.fillPoly(mask, new_segments, 1)
    mask = mask.astype(bool)

    return mask

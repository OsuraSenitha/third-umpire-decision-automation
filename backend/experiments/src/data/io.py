from typing import Union, Tuple
import numpy as np
import cv2 as cv
from .process import cvtAnnotationsTXT2LST, cvtAnnotationsLST2TXT

def readBBoxFile(file_path: str, tolist: bool=False) -> Union[str, Tuple[Tuple]]:
    txt_cntnt = ""
    with open(file_path) as handler:
        txt_cntnt = handler.read()
    output = txt_cntnt
    if tolist:
        output = cvtAnnotationsTXT2LST(txt_cntnt)
    
    return output

def saveBBoxFile(cntnt: Union[str, Tuple[Tuple]], file_path: str) -> None:
    if type(cntnt) != str:
        cntnt = cvtAnnotationsLST2TXT(cntnt)
    with open(file_path, "w") as handler:
        handler.write(cntnt)

def saveData(img: np.ndarray, seg: np.ndarray, bbx: Union[str, Tuple[Tuple]], ds_path: str, obj_name: str) -> None:
    dst_img_path = f"{ds_path}/images/{obj_name}.png"
    dst_seg_path = f"{ds_path}/segmentations/{obj_name}.png"
    dst_bbx_path = f"{ds_path}/bboxes/{obj_name}.txt"

    if type(bbx) != str:
        bbx = cvtAnnotationsLST2TXT(bbx)

    cv.imwrite(dst_img_path, img)
    cv.imwrite(dst_seg_path, seg)
    with open(dst_bbx_path, "w") as handler:
        handler.write(bbx)
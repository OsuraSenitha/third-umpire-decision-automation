from typing import Union, Tuple, Dict
import numpy as np
import json, yaml
import cv2 as cv
from .process import cvtAnnotationsTXT2LST, cvtAnnotationsLST2TXT


def colorHex2RGB(hex_color: str) -> Tuple[int]:
    if hex_color[0] == "#":
        hex_color = hex_color[1:]

    rgb = (hex_color[0:2], hex_color[2:4], hex_color[4:6])
    rgb = tuple(map(lambda h: int(h, base=16), rgb))

    return rgb


def readClassesFile(
    file_path,
    required_classes=[
        "Bowler",
        "Batsmen",
        "Wicket Keeper",
        "Umpire",
        "Fielder",
        "Ball",
        "Wicket",
    ],
    format="bgr",
) -> Dict[str, str]:
    classes_raw = {}
    with open(file_path) as handler:
        classes_raw = json.load(handler)

    classes = []
    if required_classes == "all":
        required_classes = list(map(lambda cls: cls["name"], classes_raw))
    for rc in required_classes:
        cls = tuple(filter(lambda cls: cls["name"] == rc, classes_raw))[0]
        color = np.array(colorHex2RGB(cls["color"]))
        if format == "bgr":
            color = color[::-1]
        cls = {"name": cls["name"], "color": color}
        classes.append(cls)

    return classes


def readDatasetConfig(path: str) -> Dict:
    content = {}
    with open(path) as handler:
        content = yaml.load(handler, Loader=yaml.FullLoader)
    return content


def saveDatasetConfig(config: Dict, path: str) -> None:
    with open(path, "w") as handler:
        yaml.dump(config, handler)


def readAnnotationsFile(
    file_path: str, tolist: bool = False
) -> Union[str, Tuple[Tuple]]:
    txt_cntnt = ""
    with open(file_path) as handler:
        txt_cntnt = handler.read()
    output = txt_cntnt
    if tolist:
        output = cvtAnnotationsTXT2LST(txt_cntnt)

    return output


def saveAnnotationsFile(
    cntnt: Union[str, Tuple[Tuple]], file_path: str, round_deci=None
) -> None:
    if type(cntnt) != str:
        cntnt = cvtAnnotationsLST2TXT(cntnt, round_deci)
    with open(file_path, "w") as handler:
        handler.write(cntnt)


def saveData(
    img: np.ndarray,
    bbx: Union[str, Tuple[Tuple]],
    seg: Union[str, Tuple[Tuple]],
    ds_path: str,
    obj_name: str,
) -> None:
    dst_img_path = f"{ds_path}/images/{obj_name}.png"
    dst_bbx_path = f"{ds_path}/bboxes/{obj_name}.txt"
    dst_seg_path = f"{ds_path}/segments/{obj_name}.txt"

    if type(bbx) != str:
        bbx = cvtAnnotationsLST2TXT(bbx)
    if type(seg) != str:
        seg = cvtAnnotationsLST2TXT(seg)

    cv.imwrite(dst_img_path, img)
    with open(dst_bbx_path, "w") as handler:
        handler.write(bbx)
    with open(dst_seg_path, "w") as handler:
        handler.write(seg)

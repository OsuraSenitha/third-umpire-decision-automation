from .visualize import drawRects
from .process import cvtAnnotationsTXT2LST, cvtAnnotationsLST2TXT
from .analyze import get_mask
from . import io
from . import augment
import numpy as np
from typing import Tuple, Union
import cv2 as cv
import os
import shutil
from tqdm.auto import tqdm

def DSSC_defined(img: np.ndarray, crop_data: Tuple) -> np.ndarray:
    [xmin, xmax, ymin, ymax] = crop_data
    cropped_img = img[ymin:ymax, xmin:xmax, :]

    return cropped_img

def DSSC(
        img: np.ndarray,
        bboxes: Union[str, Tuple[Tuple]],
        min_dim: int = 400,
        aspect_ratio_max: float = 1.5,
        visualize: bool = False
    ) -> Union[
        Tuple[np.ndarray, Tuple[Tuple], Tuple[int]],
        Tuple[np.ndarray, str, Tuple[int]]
    ]:
    """
    Performs the following transformation on the given image and annotations while keeping the objects intact
        1. Down scaling
        2. Shifting
        3. Cropping
    """

    boxes = bboxes
    return_string_annotations = False
    if type(bboxes) == str:
        return_string_annotations = True
        boxes = cvtAnnotationsTXT2LST(bboxes)
    W, H = img.shape[1], img.shape[0]

    # determine the possible extremes for cropping
    xmin, xmax, ymin, ymax = W, 0, H, 0
    for box in boxes:
        l, x, y, w, h = box
        x_s, y_s, w_s, h_s = x*W, y*H, w*W, h*H
        if xmax<x_s+w_s/2:
            xmax = int(x_s+w_s/2)
        if xmin>x_s-w_s/2:
            xmin = int(x_s-w_s/2)
        if ymax<y_s+h_s/2:
            ymax = int(y_s+h_s/2)
        if ymin>y_s-h_s/2:
            ymin = int(y_s-h_s/2)

    # randomly select values for croping while retaining the restrains
    xmin_crop_max = min(xmin, W-min_dim)
    xmin_crop_max = max(1, xmin_crop_max) # make sure low and high value in ranint doesnot get equal
    xmin_crop = np.random.randint(0, xmin_crop_max)
    xmax_crop_min = max(xmax, xmin_crop+min_dim)
    xmax_crop_min = min(W-1, xmax_crop_min) # make sure low and high value in ranint doesnot get equal
    xmax_crop = np.random.randint(xmax_crop_min, W)
    crop_W = xmax_crop - xmin_crop
    ymin_crop_min_dim_constraint = H-min_dim
    ymin_crop_aspectL_constraint = H-int(crop_W/aspect_ratio_max) # aspect lower
    # ymin_crop_aspectU_constraint = H-int(crop_width*aspect_ratio_max) # aspect upper
    ymin_crop_max = min(ymin, ymin_crop_min_dim_constraint, ymin_crop_aspectL_constraint)
    ymin_crop_max = max(1, ymin_crop_max) # make sure low and high value in ranint doesnot get equal
    ymin_crop = np.random.randint(0, ymin_crop_max)
    ymax_crop_min_dim_constraint = ymin_crop+min_dim
    ymax_crop_aspectL_constraint = ymin_crop+int(crop_W/aspect_ratio_max) # aspect lower
    # ymax_crop_aspectU_constraint = ymin_crop+int(crop_width*aspect_ratio_max) # aspect upper
    ymax_crop_min = max(ymax, ymax_crop_min_dim_constraint, ymax_crop_aspectL_constraint)
    ymax_crop_min = min(H-1, ymax_crop_min) # make sure low and high value in ranint doesnot get equal
    ymax_crop = np.random.randint(ymax_crop_min, H)
    crop_H = ymax_crop - ymin_crop

    # make sure the numbers are within the limits
    xmin_crop = max(0, xmin_crop)
    xmax_crop = min(W, xmax_crop)
    ymin_crop = max(0, ymin_crop)
    ymax_crop = min(H, ymax_crop)
    crop_data = [xmin_crop, xmax_crop, ymin_crop, ymax_crop]

    new_boxes = boxes
    if not visualize:
        new_boxes = []
        for box in boxes:
            l, x, y, w, h = box
            new_x = (x*W-xmin_crop)/crop_W
            new_y = (y*H-ymin_crop)/crop_H
            new_w = w*W/crop_W
            new_h = h*H/crop_H
            new_box = [l, new_x, new_y, new_w, new_h]
            new_boxes.append(new_box)
    if return_string_annotations:
        new_boxes = cvtAnnotationsLST2TXT(new_boxes)

    if visualize:
        drawn_img = drawRects(img, [[l, xmin, ymin, xmax, ymax]], False, True)
        drawn_img = drawRects(drawn_img, [[l, xmin_crop, ymin_crop, xmax_crop, ymax_crop]], False, True, (255, 0, 0))
        return drawn_img, new_boxes, crop_data
    else:
        cropped_img = DSSC_defined(img, crop_data)
        return cropped_img, new_boxes, crop_data

def horizontal_flip_defined(img: np.ndarray, flip: bool=True) -> np.ndarray:
    if flip:
        flip_img = img[:, ::-1, :]
        return flip_img
    else:
        return img

def horizontal_flip(
        img: np.ndarray, boxes: Union[str, Tuple[Tuple]]
    ) -> Tuple[np.ndarray, Union[str, Tuple[Tuple]], bool]:
    """
    Horizontally flips a given image and its annotations randomly
    """
    flip = bool(np.random.randint(0, 2))
    if flip:
        # flip the image horizontally
        flip_img = horizontal_flip_defined(img)
        # decode the annotations if they are strings
        return_string_annotations = False
        if type(boxes) == str:
            return_string_annotations = True
            boxes = cvtAnnotationsTXT2LST(boxes)
        # flip the annotations
        flip_boxes = []
        for box in boxes:
            l, x, y, w, h = box
            flip_box = (l, 1-x, y, w, h)
            flip_boxes.append(flip_box)
        # encode the boxes back to strings if they were given as strings
        if return_string_annotations:
            flip_boxes = cvtAnnotationsLST2TXT(flip_boxes)

        return flip_img, flip_boxes, flip
    else:
        return img, boxes, flip
    
def blur(img: np.ndarray, apply_blur_thresh:int = 300)->np.ndarray:
    """
    Randomly applies gaussian blur to a given image depending on its image dimension
    """
    add_blur = np.random.randint(0, 2)
    img_dim = min(img.shape[:2])
    if img_dim < apply_blur_thresh:
        add_blur = 0
    if add_blur:
        upper_lim = max(2, int(img_dim/125))
        base_k = np.random.randint(1, upper_lim)
        k_size = base_k*2+1
        blur_img = cv.GaussianBlur(img,(k_size,k_size), 0)
        return blur_img
    else:
        return img
    
def brightness_contrast(
        img: np.ndarray,
        max_bright: float = 1.5,
        min_bright: float = 0.5,
        max_contrt: float = 20,
        min_contrt: float = -20
    )->np.ndarray:
    """
    Randomly changes the brightness and contrast of a given image
    """
    rand = np.random.randn(2)
    alpha = np.clip(np.abs(rand[0])*2, min_bright, max_bright) # Brightness control
    beta = np.clip(rand[1]*10, min_contrt, max_contrt) # Contrast control
    adjusted_img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    return adjusted_img
    
def noise(img: np.ndarray, max_noise:float = 0.8)->np.ndarray:
    """
    Randomly adds gaussian noise to a given image randomly
    """
    add_noise = np.random.randint(0, 2)
    if add_noise:
        noise_scale = np.clip(np.random.randn(1), 0, max_noise)
        gauss_noise = (np.random.randn(*img.shape)*noise_scale).astype(np.uint8)
        gn_img = cv.add(img,gauss_noise)
        return gn_img
    else:
        return img

def rotate_hue(
        img:np.ndarray,
        seg:np.ndarray,
        human_colors:Tuple[str] = ['#0018fd', '#b07a53', '#8fdfa9', '#e20959', '#c2fe6b']
    ) -> np.ndarray:
    """
    Randomly shifts the hue of the humans in the image
    """
    # convert to HSV
    shift = np.random.uniform(-180, 180)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv)
    hnew = np.mod(h + shift, 180).astype(np.uint8)
    hsv_new = cv.merge([hnew,s,v])
    shifted_img = cv.cvtColor(hsv_new, cv.COLOR_HSV2BGR)

    mask = get_mask(seg, human_colors)

    adjusted_img = img.copy()
    adjusted_img[mask] = shifted_img[mask]

    return adjusted_img

def create_dataset(
        original_data_path: str,
        augment_data_path: str,
        total_raw_data_count: int,
        delete_nums: Tuple[int],
        emphasis_nums: Tuple[int],
        augment_rounds: int,
        emphasis_weight: int,
    ) -> None:
    # calculate the expected data counts
    usable_raw_data_count = total_raw_data_count-len(delete_nums)
    emphasis_count = len(emphasis_nums)
    non_emphasis_count = usable_raw_data_count - emphasis_count
    final_data_count = usable_raw_data_count + augment_rounds*(emphasis_count*emphasis_weight + non_emphasis_count)
    final_data_count, emphasis_count*emphasis_weight*augment_rounds

    src_images_path = f"{original_data_path}/images"
    src_bboxes_path = f"{original_data_path}/bboxes"
    image_names = tuple(filter(lambda name: "___" not in name, os.listdir(src_images_path)))

    # validate the completeness of the raw dataset
    print("---- Validating Raw Data ----")
    with tqdm(total=usable_raw_data_count) as pbar:
        for img_name in image_names:
            if all(tuple(map(lambda num: str(num) not in img_name, delete_nums))):
                img_src_path = f"{src_images_path}/{img_name}"
                seg_src_path = f"{src_images_path}/{img_name}___fuse.png"
                bbx_src_path = f"{src_bboxes_path}/{os.path.splitext(img_name)[0]}.txt"
                assert os.path.exists(img_src_path)
                assert os.path.exists(seg_src_path)
                assert os.path.exists(bbx_src_path)
                pbar.update(1)

    # Create the directories
    dst_imgs_path = f"{augment_data_path}/images"
    dst_segs_path = f"{augment_data_path}/segmentations"
    dst_bbxs_path = f"{augment_data_path}/bboxes"
    for dir in [dst_imgs_path, dst_segs_path, dst_bbxs_path]:
        os.makedirs(dir, exist_ok=True)

    # Create new dataset
    print("---- Creating New Data ----")
    with tqdm(total=final_data_count) as pbar:
        for img_name in image_names:
            if all(tuple(map(lambda num: f"({num})" not in img_name, delete_nums))):

                obj_name = os.path.splitext(img_name)[0]

                src_img_path = f"{src_images_path}/{img_name}"
                src_seg_path = f"{src_images_path}/{img_name}___fuse.png"
                src_bbx_path = f"{src_bboxes_path}/{obj_name}.txt"
                src_img = cv.imread(src_img_path)
                src_seg = cv.imread(src_seg_path)
                src_bbx = io.readBBoxFile(src_bbx_path, True)

                # copy the raw image itself first
                dst_img_path = f"{dst_imgs_path}/{obj_name}-0.png"
                dst_seg_path = f"{dst_segs_path}/{obj_name}-0.png"
                dst_bbx_path = f"{dst_bbxs_path}/{obj_name}-0.txt"
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_seg_path, dst_seg_path)
                shutil.copy(src_bbx_path, dst_bbx_path)
                pbar.update(1)
                # raw_data_count + augment_rounds*(emphasis_count*emphasis_ratio + non_emphasis_count)
                new_img_count = augment_rounds
                if any(tuple(map(lambda num: f"({num})" in img_name, emphasis_nums))):
                    new_img_count = augment_rounds*emphasis_weight

                for i in range(1, new_img_count+1):

                    dst_img, dst_bbx, crop_data = augment.DSSC(src_img, src_bbx)
                    dst_seg = augment.DSSC_defined(src_seg, crop_data)
                    
                    dst_img, dst_bbx, flipped = augment.horizontal_flip(dst_img, dst_bbx)
                    dst_seg = augment.horizontal_flip_defined(dst_seg, flipped)

                    dst_img = augment.rotate_hue(dst_img, dst_seg)
                    dst_img = augment.brightness_contrast(dst_img)
                    dst_img = augment.blur(dst_img)
                    dst_img = augment.noise(dst_img)

                    io.saveData(dst_img, dst_seg, dst_bbx, augment_data_path, f"{obj_name}-{i}")

                    pbar.update(1)
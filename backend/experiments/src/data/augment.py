from .visualize import drawRects
from .process import cvtAnnotationsTXT2LST, cvtAnnotationsLST2TXT, makeMaskFromSegments
from .analyze import getMask
from . import io
from . import augment
import numpy as np
from typing import Tuple, Union
import cv2 as cv
import os
import shutil
from tqdm.auto import tqdm


def DSSC(
    img: np.ndarray,
    bboxes: Union[str, Tuple[Tuple]],
    sgmnts: Union[str, Tuple[Tuple]],
    interested_seg_lbls: Tuple[int],
    min_dim: int = 100,
    aspect_ratio_max: float = 1.5,
    visualize: bool = False,
) -> Union[
    Tuple[np.ndarray, Tuple[Tuple], Tuple[int]], Tuple[np.ndarray, str, Tuple[int]]
]:
    """
    Performs the following transformation on the given image and annotations while keeping the objects intact
        1. Down scaling
        2. Shifting
        3. Cropping
    """

    return_string_annotations = False
    if type(bboxes) == str:
        return_string_annotations = True
        bboxes = cvtAnnotationsTXT2LST(bboxes)
        sgmnts = cvtAnnotationsTXT2LST(sgmnts)
    H, W, _ = img.shape

    # determine the possible extremes for cropping
    xmin, xmax, ymin, ymax = W, 0, H, 0
    for label in sgmnts:
        l, *pts = label
        if l in interested_seg_lbls:
            X = np.array(pts[0::2])
            Y = np.array(pts[1::2])
            X = X * W
            Y = Y * H
            xmin = min(X.min(), xmin)
            ymin = min(Y.min(), ymin)
            xmax = max(X.max(), xmax)
            ymax = max(Y.max(), ymax)

    # randomly select values for croping while retaining the restrains
    xmin_crop_max = min(xmin, W - min_dim)
    xmin_crop_max = max(
        1, xmin_crop_max
    )  # make sure low and high value in ranint doesnot get equal
    xmin_crop = np.random.randint(0, xmin_crop_max)
    xmax_crop_min = max(xmax, xmin_crop + min_dim)
    xmax_crop_min = min(
        W - 1, xmax_crop_min
    )  # make sure low and high value in ranint doesnot get equal
    xmax_crop = np.random.randint(xmax_crop_min, W)
    crop_W = xmax_crop - xmin_crop
    ymin_crop_min_dim_constraint = H - min_dim
    ymin_crop_aspectL_constraint = H - int(crop_W / aspect_ratio_max)  # aspect lower
    # ymin_crop_aspectU_constraint = H-int(crop_width*aspect_ratio_max) # aspect upper
    ymin_crop_max = min(
        ymin, ymin_crop_min_dim_constraint, ymin_crop_aspectL_constraint
    )
    ymin_crop_max = max(
        1, ymin_crop_max
    )  # make sure low and high value in ranint doesnot get equal
    ymin_crop = np.random.randint(0, ymin_crop_max)
    ymax_crop_min_dim_constraint = ymin_crop + min_dim
    ymax_crop_aspectL_constraint = ymin_crop + int(
        crop_W / aspect_ratio_max
    )  # aspect lower
    # ymax_crop_aspectU_constraint = ymin_crop+int(crop_width*aspect_ratio_max) # aspect upper
    ymax_crop_min = max(
        ymax, ymax_crop_min_dim_constraint, ymax_crop_aspectL_constraint
    )
    ymax_crop_min = min(
        H - 1, ymax_crop_min
    )  # make sure low and high value in ranint doesnot get equal
    ymax_crop = np.random.randint(ymax_crop_min, H)
    crop_H = ymax_crop - ymin_crop

    # make sure the numbers are within the limits
    xmin_crop = max(0, xmin_crop)
    xmax_crop = min(W, xmax_crop)
    ymin_crop = max(0, ymin_crop)
    ymax_crop = min(H, ymax_crop)
    crop_data = [xmin_crop, xmax_crop, ymin_crop, ymax_crop]

    new_bboxes = bboxes
    new_sgmnts = sgmnts
    if not visualize:
        new_bboxes = []
        for label in bboxes:
            l, x, y, w, h = label
            new_x = (x * W - xmin_crop) / crop_W
            new_y = (y * H - ymin_crop) / crop_H
            new_w = w * W / crop_W
            new_h = h * H / crop_H
            new_lbl = [l, new_x, new_y, new_w, new_h]
            new_bboxes.append(new_lbl)
        new_sgmnts = []
        for label in sgmnts:
            l, *pts = label
            X = np.array(pts[0::2])
            Y = np.array(pts[1::2])
            X = (X * W - xmin_crop) / crop_W
            Y = (Y * H - ymin_crop) / crop_H
            n = X.size
            min_val = np.zeros(n)
            max_val = np.ones(n)
            X = np.min([X, max_val], axis=0)
            X = np.max([X, min_val], axis=0)
            Y = np.min([Y, max_val], axis=0)
            Y = np.max([Y, min_val], axis=0)

            new_lbl = np.zeros(len(label))
            new_lbl[0] = l
            new_lbl[1::2] = X
            new_lbl[2::2] = Y
            new_lbl = new_lbl.tolist()
            new_sgmnts.append(new_lbl)

        cropped_img = img[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]
        if return_string_annotations:
            new_bboxes = cvtAnnotationsLST2TXT(new_bboxes)
            new_sgmnts = cvtAnnotationsLST2TXT(new_sgmnts)

        return cropped_img, new_bboxes, new_sgmnts, crop_data
    else:
        drawn_img = drawRects(img, [[l, xmin, ymin, xmax, ymax]], False, True)
        drawn_img = drawRects(
            drawn_img,
            [[l, xmin_crop, ymin_crop, xmax_crop, ymax_crop]],
            False,
            True,
            (255, 0, 0),
        )
        if return_string_annotations:
            new_bboxes = cvtAnnotationsLST2TXT(new_bboxes)
            new_sgmnts = cvtAnnotationsLST2TXT(new_sgmnts)

        return drawn_img, new_bboxes, new_sgmnts, crop_data


def horizontal_flip(
    img: np.ndarray, bbx: Union[str, Tuple[Tuple]], seg: Union[str, Tuple[Tuple]]
) -> Tuple[np.ndarray, Union[str, Tuple[Tuple]], bool]:
    """
    Horizontally flips a given image and its annotations randomly
    """
    flip = bool(np.random.randint(0, 2))
    if flip:
        # flip the image horizontally
        flip_img = img[:, ::-1, :]
        # decode the annotations if they are strings
        return_string_annotations = False
        if type(bbx) == str:
            return_string_annotations = True
            bbx = cvtAnnotationsTXT2LST(bbx)
            seg = cvtAnnotationsTXT2LST(seg)
        # flip the annotations
        bbx = np.array(bbx)
        bbx[:, 1] = 1 - bbx[:, 1]
        flip_bbxs = bbx.tolist()

        for lbl in seg:
            lbl[1::2] = list(map(lambda val: 1 - val, lbl[1::2]))
        flip_segs = seg

        # encode the boxes back to strings if they were given as strings
        if return_string_annotations:
            flip_bbxs = cvtAnnotationsLST2TXT(flip_bbxs)
            flip_segs = cvtAnnotationsLST2TXT(flip_segs)

        return flip_img, flip_bbxs, flip_segs
    else:
        return img, bbx, seg


def blur(img: np.ndarray, apply_blur_thresh: int = 300) -> np.ndarray:
    """
    Randomly applies gaussian blur to a given image depending on its image dimension
    """
    add_blur = np.random.randint(0, 2)
    img_dim = min(img.shape[:2])
    if img_dim < apply_blur_thresh:
        add_blur = 0
    if add_blur:
        upper_lim = max(2, int(img_dim / 125))
        base_k = np.random.randint(1, upper_lim)
        k_size = base_k * 2 + 1
        blur_img = cv.GaussianBlur(img, (k_size, k_size), 0)
        return blur_img
    else:
        return img


def brightness_contrast(
    img: np.ndarray,
    max_bright: float = 1.5,
    min_bright: float = 0.5,
    max_contrt: float = 20,
    min_contrt: float = -20,
) -> np.ndarray:
    """
    Randomly changes the brightness and contrast of a given image
    """
    rand = np.random.randn(2)
    alpha = np.clip(np.abs(rand[0]) * 2, min_bright, max_bright)  # Brightness control
    beta = np.clip(rand[1] * 10, min_contrt, max_contrt)  # Contrast control
    adjusted_img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    return adjusted_img


def noise(
    img: np.ndarray, max_noise_scale: float = 10, noise_thresh_dim=400
) -> np.ndarray:
    """
    Randomly adds gaussian noise to a given image randomly
    """
    H, W, _ = img.shape
    if max(H, W) > noise_thresh_dim:
        add_noise = np.random.randint(0, 2)
        if add_noise:
            max_noise = max_noise_scale / min(H, W) * 1080
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
            noise_scale = np.random.uniform(max_noise)
            gauss_noise = np.clip(
                np.random.randn(*img.shape[:-1]) * noise_scale, -max_noise, max_noise
            ).astype(np.float32)
            hsv[:, :, 2] = cv.add(hsv[:, :, 2], gauss_noise)
            hsv = hsv.astype(np.uint8)
            gn_img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            return gn_img

    return img


def rotate_hue(
    img: np.ndarray,
    human_seg: np.ndarray,
) -> np.ndarray:
    """
    Randomly shifts the hue of the humans in the image
    """

    # convert to HSV
    shift = np.random.uniform(-180, 180)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    hnew = np.mod(h + shift, 180).astype(np.uint8)
    hsv_new = cv.merge([hnew, s, v])
    shifted_img = cv.cvtColor(hsv_new, cv.COLOR_HSV2BGR)

    mask = makeMaskFromSegments(img.shape[:2], human_seg)

    adjusted_img = img.copy()
    adjusted_img[mask] = shifted_img[mask]

    return adjusted_img


def create_dataset(
    original_data_path: str,
    augment_data_path: str,
    emphasis_img_names: Tuple[str],
    augment_rounds: int,
    emphasis_weight: int,
    interested_seg_lbls: Tuple[int],
    human_nms: Tuple[str] = ["Bowler", "Batsmen", "Wicket Keeper", "Umpire", "Fielder"],
) -> None:
    images_subdir = "images"
    bboxes_subdir = "bboxes"
    segmts_subdir = "segments"

    src_images_dir = f"{original_data_path}/{images_subdir}"
    src_bboxes_dir = f"{original_data_path}/{bboxes_subdir}"
    src_segmts_dir = f"{original_data_path}/{segmts_subdir}"

    src_bbx_config_path = f"{original_data_path}/bboxes.yaml"
    src_seg_config_path = f"{original_data_path}/segments.yaml"
    dst_bbx_config_path = f"{augment_data_path}/bboxes.yaml"
    dst_seg_config_path = f"{augment_data_path}/segments.yaml"

    # count the number of images
    total_raw_data_count = len(os.listdir(src_images_dir))

    # calculate the expected data counts
    emphasis_count = len(emphasis_img_names)
    non_emphasis_count = total_raw_data_count - emphasis_count
    final_data_count = total_raw_data_count + augment_rounds * (
        emphasis_count * emphasis_weight + non_emphasis_count
    )
    final_data_count, emphasis_count * emphasis_weight * augment_rounds

    # validate the completeness of the raw dataset
    print("---- Validating Raw Data ----")
    with tqdm(total=total_raw_data_count) as pbar:
        for img_nm in os.listdir(src_images_dir):
            obj_nm = os.path.splitext(img_nm)[0]
            lbl_nm = f"{obj_nm}.txt"
            bbx_path = f"{src_bboxes_dir}/{lbl_nm}"
            seg_path = f"{src_segmts_dir}/{lbl_nm}"
            assert os.path.exists(bbx_path)
            assert os.path.exists(seg_path)
            pbar.update(1)

    # Create the directories
    dst_imgs_dir = f"{augment_data_path}/{images_subdir}"
    dst_bbxs_dir = f"{augment_data_path}/{bboxes_subdir}"
    dst_segs_dir = f"{augment_data_path}/{segmts_subdir}"
    for dir in [dst_imgs_dir, dst_bbxs_dir, dst_segs_dir]:
        os.makedirs(dir, exist_ok=True)

    # copy config files
    shutil.copy(src_bbx_config_path, dst_bbx_config_path)
    shutil.copy(src_seg_config_path, dst_seg_config_path)

    # Create new dataset
    print("---- Creating New Data ----")
    seg_config = io.readDatasetConfig(src_seg_config_path)
    seg_class_names = seg_config["names"]
    human_seg_lbls = [seg_class_names.index(nm) for nm in human_nms]
    with tqdm(total=final_data_count) as pbar:
        for img_nm in os.listdir(src_images_dir):
            obj_nm = os.path.splitext(img_nm)[0]
            lbl_nm = obj_nm + ".txt"

            new_img_count = augment_rounds
            if img_nm in emphasis_img_names:
                new_img_count = augment_rounds * emphasis_weight

            src_img_path = f"{src_images_dir}/{img_nm}"
            src_bbx_path = f"{src_bboxes_dir}/{lbl_nm}"
            src_seg_path = f"{src_segmts_dir}/{lbl_nm}"
            src_img = cv.imread(src_img_path)
            src_bbx = io.readAnnotationsFile(src_bbx_path)
            src_seg = io.readAnnotationsFile(src_seg_path)
            if src_bbx.strip() == "":
                pbar.update(new_img_count + 1)
                continue
            src_bbx = cvtAnnotationsTXT2LST(src_bbx)
            src_seg = cvtAnnotationsTXT2LST(src_seg)

            # copy the raw image itself first
            dst_img_path = f"{dst_imgs_dir}/{obj_nm}-0.png"
            dst_bbx_path = f"{dst_bbxs_dir}/{obj_nm}-0.txt"
            dst_seg_path = f"{dst_segs_dir}/{obj_nm}-0.txt"
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_bbx_path, dst_bbx_path)
            shutil.copy(src_seg_path, dst_seg_path)
            pbar.update(1)

            for i in range(1, new_img_count + 1):
                dst_img, dst_bbx, dst_seg, _ = augment.DSSC(
                    src_img, src_bbx, src_seg, interested_seg_lbls
                )
                dst_img, dst_bbx, dst_seg = augment.horizontal_flip(
                    dst_img, dst_bbx, dst_seg
                )
                human_segs = list(
                    filter(lambda seg_line: seg_line[0] in human_seg_lbls, dst_seg)
                )
                dst_img = augment.rotate_hue(dst_img, human_segs)
                dst_img = brightness_contrast(dst_img)
                dst_img = blur(dst_img)
                dst_img = noise(dst_img)

                dst_bbx = cvtAnnotationsLST2TXT(dst_bbx)
                dst_seg = cvtAnnotationsLST2TXT(dst_seg)
                io.saveData(
                    dst_img, dst_bbx, dst_seg, augment_data_path, f"{obj_nm}-{i}"
                )

                pbar.update(1)

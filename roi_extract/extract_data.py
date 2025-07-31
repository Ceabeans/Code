import random
import shutil
import sys
import os
import time

import cv2
import numpy as np
from tqdm import tqdm
from base import config_toml, current_dir_root, mylogger
from palm_roi_ext.instance import AutoRoIExtract

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

"""
Extract ROI (Region of Interest) from images in a folder using a palm instance.
If the thumb in the image is pointing to the right, perform a horizontal flip (need_flip_horizontal=True).
"""


def ex_tract_data(need_flip_horizontal=False):
    data_origin_path = config_toml["DATAEXTRACT"]["data_origin_path"]
    data_roi_path = config_toml["DATAEXTRACT"]["data_roi_path"]
    data_save_path = os.path.join(current_dir_root, data_origin_path)
    data_roi_path = os.path.join(current_dir_root, data_roi_path)
    if not os.path.exists(data_roi_path):
        os.makedirs(data_roi_path)

    roi_extract = AutoRoIExtract(method='bezier')
    img_dir_paths = os.listdir(data_save_path)
    mylogger.info(f"number：{len(img_dir_paths)}")
    start = time.time()
    for index, img_dir_path in enumerate(tqdm(img_dir_paths, desc="Processing directories")):
        img_dir_path_abs = os.path.join(data_save_path, img_dir_path)
        if img_dir_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', 'tiff')):
            try:
                # img = cv2.imread(img_dir_path_abs)    
                img = cv2.imdecode(np.fromfile(
                    img_dir_path_abs, dtype=np.uint8), -1)
                if need_flip_horizontal:
                    img = cv2.flip(img, 1)
                draw_img, roi_img = roi_extract.roi_extract(img)
                mylogger.info(f"size：{roi_img.shape}")
                file_name = os.path.basename(img_dir_path_abs).split(".")[0]
                # cv2.imwrite(os.path.join(data_roi_path,
                #             f"{file_name}.bmp"), roi_img)    
                cv2.imencode('.bmp', roi_img)[1].tofile(
                    os.path.join(data_roi_path, f"{file_name}.bmp"))
            except Exception as e:
                mylogger.error(
                    f"Error processing file {img_dir_path_abs}: {e}")
    end = time.time()
    mylogger.info(f"Time elapsed: {(end - start):.2f} seconds")


if __name__ == '__main__':
    # left hand
    ex_tract_data(need_flip_horizontal=False)
    # right hand
    # ex_tract_data(need_flip_horizontal=True)

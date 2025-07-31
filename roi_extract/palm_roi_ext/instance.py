
from palm_roi_ext.hand_key_points.key_point import HandKeyPointDetect
from palm_roi_ext.rotate import HandRotate
from palm_roi_ext.extract import ROIExtract
from base import ShowImage, config_toml
import numpy as np
import cv2
import sys
import os
from typing import Literal
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))


class AutoRoIExtract(ShowImage):
    """
    mothod : {'mpd', 'bezier'}
    """

    def __init__(self, method: Literal["mpd", "bezier"] = 'bezier'):
        self.mothod = method
        # 1. 手部关键点识别
        self.key_points_instance = HandKeyPointDetect()
        # 2. 图像旋转
        self.rotate_instance = HandRotate()
        # 3. ROI区域提取
        self.roi_extract_instance = ROIExtract(method)

    # 由手部关键点计算gap_point(A,B)
    def get_gap_point(self, key_points):
        if self.mothod == 'bezier':
            A = np.mean(key_points[[5, 9]], axis=0)
            B = np.mean(key_points[[13, 17]], axis=0)
        elif self.mothod == 'mpd':
            A = key_points[9]
            B = key_points[13]
        return np.vstack([A, B])

    def roi_extract_test(self, img):
        # 1. 手部关键点识别
        key_points = self.key_points_instance.get_hand_key_point(img)
        gap_points = self.get_gap_point(key_points)
        # 展示关键点识别出来的效果
        self.key_points_instance.show_key_point(img, key_points, gap_points)
        # 2. 图像旋转
        img, angle, gap_points = self.rotate_instance.rotate_angle_img(
            gap_points, img)
        self.show_image("rotate", img)
        # 3. ROI区域提取
        draw_img, roi_img = self.roi_extract_instance.extract_roi(
            gap_points, img)
        self.show_image("extract", draw_img)
        self.show_image("roi", roi_img)

    def roi_extract(self, img):
        # 1. 手部关键点识别
        key_points = self.key_points_instance.get_hand_key_point(img)
        gap_points = self.get_gap_point(key_points)
        # 2. 图像旋转
        img, angle, gap_points = self.rotate_instance.rotate_angle_img(
            gap_points, img)
        # 3. ROI区域提取
        draw_img, roi_img = self.roi_extract_instance.extract_roi(
            gap_points, img)
        return draw_img, roi_img


if __name__ == '__main__':

    img = cv2.imread(r"../../data/palm.JPG")
    roi_extract = AutoRoIExtract()
    # draw_img,roi,roi_circle = roi_extract.roi_extract(img)
    # roi_extract.show_image("extract",draw_img)
    # roi_extract.show_image("roi",roi)
    # roi_extract.show_image("roi_circle",roi_circle)
    roi_extract.roi_extract_test(img)

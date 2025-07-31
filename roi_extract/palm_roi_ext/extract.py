from base import ShowImage
import numpy as np
import cv2
import sys
import os
from typing import Literal
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))


class ROIExtract(ShowImage):
    def __init__(self, method: Literal["mpd", "bezier"] = 'bezier'):
        self.method = method
        if self.method == 'bezier':
            self.square_side_coef = 7/6
            self.oc_coef = 3/4
        elif self.method == 'mpd':
            self.square_side_coef = 5/2
            self.oc_coef = 3/2

    def extract_roi(self, gap_points, image):
        assert gap_points.shape == (2, 2)

        (A_x, A_y), (B_x, B_y) = gap_points
        square_side = self.square_side_coef * abs(B_x-A_x)
        # 计算ROI区域的中心点
        center = [(A_x + B_x) / 2, A_y + self.oc_coef * abs(B_x - A_x)]
        # 计算ROI区域的坐标
        roi_top_left = (
            int(center[0] - square_side / 2), int(center[1] - square_side / 2))
        roi_bottom_right = (
            int(center[0] + square_side / 2), int(center[1] + square_side / 2))

        # 裁剪ROI区域
        # img(h, w, c)
        cropped_roi = image[roi_top_left[1]:roi_bottom_right[1],
                            roi_top_left[0]:roi_bottom_right[0]]
        cropped_roi = cv2.resize(cropped_roi, (256, 256))
        # 绘制ROI区域
        image_with_roi = cv2.rectangle(
            image.copy(), roi_top_left, roi_bottom_right, (0, 0, 255), 2)
        # 绘制中心点
        image_with_roi = cv2.circle(image_with_roi, (int(
            center[0]), int(center[1])), 5, (255, 50, 60), -1)
        
        return image_with_roi, cropped_roi

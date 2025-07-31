from base import ShowImage
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))


class HandRotate(ShowImage):
    def __calculate_angle_Aetween_vectors(self,vector_1: np.array, vector_2: np.array):
        """
        计算vector_1到vector_2的旋转角度,正值为逆时针,负值为顺时针
        
        返回:float: 两个向量之间的夹角，单位为度。
        """

        angle = np.arctan2(np.linalg.det(
            [vector_1, vector_2]), np.dot(vector_1, vector_2))
        angle_degrees = np.degrees(angle)
        return angle_degrees

    def rotate_angle_img(self, gap_points, image):
        """
        :param gap_points: 手部关键点
        :return: 返回旋转后的图像、旋转角度和旋转后的关键点
        """
        if gap_points.size == 0:
            return None
        # 获取A, B
        A, B = gap_points

        (h, w) = image.shape[:2]
        # 计算向量
        A_x = A[0]
        A_y = -A[1]
        B_x = B[0]
        B_y = -B[1]

        # 计算向量AB
        vector_AB = (B_x - A_x, B_y - A_y)
        vector_OX = (w, 0)

        # 计算向量与x轴的夹角
        angle = self.__calculate_angle_Aetween_vectors(vector_AB, vector_OX)

        # 旋转图片以使AB与x轴平行
        center = (w / 2, h / 2)
        # 旋转角度为向量AB与x轴的夹角
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        # 旋转关键点
        rotated_points = []
        for point in gap_points:
            # 将点转换为齐次坐标
            homogeneous_point = np.array([point[0], point[1], 1])
            # 应用旋转矩阵
            transformed_point = matrix @ homogeneous_point
            # 转换回二维坐标
            rotated_points.append(transformed_point[:2])

        return rotated, angle, np.array(rotated_points)

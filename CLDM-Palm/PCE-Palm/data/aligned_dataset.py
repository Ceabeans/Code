# import os.path
# from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


# class AlignedDataset(BaseDataset):
#     """A dataset class for paired image dataset.

#     It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
#     During test time, you need to prepare a directory '/path/to/data/test'.
#     """

#     def __init__(self, opt):
#         """Initialize this dataset class.

#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
#         self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
#         assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
#         self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
#         self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

#     def __getitem__(self, index):
#         """Return a data point and its metadata information.

#         Parameters:
#             index - - a random integer for data indexing

#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor) - - an image in the input domain
#             B (tensor) - - its corresponding image in the target domain
#             A_paths (str) - - image paths
#             B_paths (str) - - image paths (same as A_paths)
#         """
#         # read a image given a random integer index
#         AB_path = self.AB_paths[index]
#         AB = Image.open(AB_path).convert('RGB')
#         # split AB image into A and B
#         w, h = AB.size
#         w2 = int(w / 2)

#         if self.opt.single_test:
#             A = AB
#             B = AB.copy()
#         else:
#             A = AB.crop((0, 0, w2, h))
#             B = AB.crop((w2, 0, w, h))



#         # apply the same transform to both A and B
#         transform_params = get_params(self.opt, A.size)
#         A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
#         B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

#         A = A_transform(A)
#         B = B_transform(B)

#         return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return len(self.AB_paths)

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    """用于配对图像数据集的类。

    假设目录结构为 '/path/to/data/train' 下有两个文件夹 'trainA' 和 'trainB'，两个子文件夹中的图像文件名相同。
    """

    def __init__(self, opt):
        """初始化数据集类。

        参数：
            opt (Option class) -- 存储所有实验标志；需要是 BaseOptions 的子类
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # 获取A类图像的目录
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # 获取B类图像的目录

        # 获取A和B文件夹中按名称排序的图像路径
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        assert len(self.A_paths) == len(self.B_paths), "A和B文件夹中的文件数量不一致！"
        assert self.opt.load_size >= self.opt.crop_size, "crop_size应小于加载图像的尺寸"

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数：
            index - - 用于数据索引的随机整数

        返回包含A, B, A_paths 和 B_paths 的字典：
            A (tensor) - - 输入域中的图像
            B (tensor) - - 目标域中的对应图像
            A_paths (str) - - A图像路径
            B_paths (str) - - B图像路径
        """
        # 读取索引位置的图像
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # 对A和B应用相同的变换
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """返回数据集中图像的总数。"""
        return len(self.A_paths)

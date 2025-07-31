# import os
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from itertools import islice
# from util import html
# import datetime

# # options
# opt = TestOptions().parse()
# opt.num_threads = 1   # test code only supports num_threads=1
# opt.batch_size = 1   # test code only supports batch_size=1
# opt.serial_batches = True  # no shuffle

# # create dataset
# dataset = create_dataset(opt)
# model = create_model(opt)
# model.setup(opt)
# # model.eval()
# print('Loading model %s' % opt.model)

# # create website
# web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
# webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# if __name__ == '__main__':
#     # test stage
#     for i, data in enumerate(islice(dataset, opt.num_test)):
#         model.set_input(data)
#         images = []
#         names = []
        
#         print('process input image %3.3d/%3.3d' % (i, min(len(dataset), opt.num_test//opt.batch_size)))
#         if not opt.sync:
#             z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
#         for nn in range(1, opt.n_samples + 1):
#             encode = nn == 0 and not opt.no_encode
#             real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
#             if nn == 0:
#                 images = [real_A, real_B, fake_B]
#                 names = ['input', 'ground truth', 'encoded']
#             else:
#                 images.append(fake_B)
#                 names.append('random_sample%2.2d' % nn)

        
#         img_path = os.path.basename(data['A_paths'][0]).split('.')[0]
#         save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

#     webpage.save()

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import torch

# options
opt = TestOptions().parse()
opt.dataroot = '/root/onethingai-fs/pce_Validation'
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1    # test code only supports batch_size=1
opt.n_samples = 1     # 每张bezier生成的伪掌纹数
opt.serial_batches = True  # no shuffle
opt.load_size = 256
opt.dataset_mode = 'single'
opt.no_flip = True
opt.use_dropout = False
opt.checkpoints_dir = './checkpoints/pcetopalm/'

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

opt.num_test = min(opt.num_test, len(dataset))

# 指定根输出目录
output_dir = '/root/onethingai-fs/PCE_Validation'  # 修改为你想要的根输出目录

if __name__ == '__main__':
    for i, data in enumerate(islice(dataset, opt.num_test)):
        # 获取输入图片的子文件夹结构
        input_subfolder = os.path.relpath(data['A_paths'][0], start=opt.dataroot)
        input_subfolder = os.path.dirname(input_subfolder)

        # 构建输出文件夹路径
        output_subfolder = os.path.join(output_dir, input_subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # 生成文件名列表
        img_name = os.path.basename(data['A_paths'][0]).split('.')[0]
        custom_img_path = os.path.join(output_subfolder, img_name)

        # 检查是否已经处理过该项
        output_exists = True
        for nn in range(opt.n_samples):
            output_file = f"{custom_img_path}_random_sample{nn:02d}.png"
            if not os.path.exists(output_file):
                output_exists = False
                break

        if output_exists:
            print(f"Skipping already processed image {i+1}/{opt.num_test}: {img_name}")
            continue

        # 移除 test_mode 参数
        model.set_input(data,test_mode=True)
        print('Processing input image %3.3d/%3.3d' % (i+1, opt.num_test))

        z_samples = model.get_z_random(opt.n_samples, opt.nz)
        images = []
        names = []

        for nn in range(opt.n_samples):
            real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=False)
            images.append(fake_B)
            names.append(f"random_sample{nn:02d}")

        # 保存图像
        save_images(webpage, images, names, custom_img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

    webpage.save()

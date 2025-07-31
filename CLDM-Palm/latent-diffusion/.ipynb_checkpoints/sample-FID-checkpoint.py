import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from DDIM_C import GaussianDiffusion
from ldm.modules.diffusionmodules.openaimodel import UNetModel

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {device_name}")
else:
    print("No GPU available, using CPU.")

# Step 1: 实例化 VQModel
ckpt_path = 'vqmodel_checkpoint.ckpt'
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

class VQModel(torch.nn.Module):
    def __init__(self, ddconfig, embed_dim=3, n_embed=8192):
        super().__init__()
        from taming.modules.diffusionmodules.model import Decoder
        from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def decode(self, x, force_not_quantize=False):
        if not force_not_quantize:
            quant, _, _ = self.quantize(x)
        else:
            quant = x
        dec = self.decoder(self.post_quant_conv(quant))
        return dec

vq_model = VQModel(
    ddconfig={
        'double_z': False,
        'z_channels': 3,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0
    },
    embed_dim=3,
    n_embed=8192
)

# 加载 VQModel 权重
state_dict = checkpoint.get('model_state_dict', checkpoint)
filtered_state_dict = {k: v for k, v in state_dict.items() if k in vq_model.state_dict()}
vq_model.load_state_dict(filtered_state_dict, strict=False)
vq_model.to(device).eval()
for param in vq_model.parameters():
    param.requires_grad = False

# Step 2: 实例化 UNetModel
unet_config = {
    "image_size": 64,
    "in_channels": 6,
    "out_channels": 3,
    "model_channels": 224,
    "attention_resolutions": [8, 4, 2],
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 3, 4],
    "num_head_channels": 32,
}

unet_model = UNetModel(**unet_config)
checkpoint_path_u = 'ddim_c.ckpt'
pretrained_dict = torch.load(checkpoint_path_u, map_location=device,weights_only=True)
unet_model.load_state_dict(pretrained_dict)
unet_model.to(device).eval()
for param in unet_model.parameters():
    param.requires_grad = False

# Step 3: 实例化 GaussianDiffusion
timesteps = 1000
gaussian_diffusion = GaussianDiffusion(
    timesteps=timesteps,
    beta_schedule='linear',
    linear_start=0.0015,
    linear_end=0.0155
)

# Step 4: 图像采样与生成
input_base_path = '/root/onethingai-fs/BEZIER-40w'
output_base_path = '/root/onethingai-fs/C_LDM_DDIM20_Validation'
os.makedirs(output_base_path, exist_ok=True)
image_size = 64
ddimstep = 20
batch_size = 25  # 每次处理的图像数量

def process_images(input_folder, output_folder):
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    os.makedirs(output_folder, exist_ok=True)

    # 分批处理图像
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        images = []
        
        for file in batch_files:
            image_path = os.path.join(input_folder, file)
            image = Image.open(image_path).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
            image_tensor = (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0) * 2 - 1
            images.append(image_tensor)
        
        images_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            sample_tensor = gaussian_diffusion.ddim_sample(
                unet_model, 
                image_size=image_size, 
                batch_size=len(batch_files), 
                channels=3, 
                ddim_timesteps=ddimstep,
                cd=images_tensor
            )
            generated_images = vq_model.decode(sample_tensor)
            output_tensor = generated_images.clamp(-1, 1)

        for j, output_image in enumerate(output_tensor):
            output_image = ((output_image + 1) / 2 * 255).clamp(0, 255).byte()
            output_image_pil = Image.fromarray(output_image.permute(1, 2, 0).cpu().numpy())
            output_image_pil.save(os.path.join(output_folder, batch_files[j]))

def process_all_folders():
    total_folders = 4000
    progress_bar = tqdm(total=total_folders, desc="Processing folders")
    
    for folder_idx in range(total_folders):
        folder_name = f"{folder_idx:04d}"
        input_folder = os.path.join(input_base_path, folder_name)
        output_folder = os.path.join(output_base_path, folder_name)
        if os.path.isdir(input_folder):
            process_images(input_folder, output_folder)
        progress_bar.update(1)
    
    progress_bar.close()

if __name__ == "__main__":
    process_all_folders()

# import os
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from DDIM_C import GaussianDiffusion
# from ldm.modules.diffusionmodules.openaimodel import UNetModel

# # 检查是否有 GPU 可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     device_name = torch.cuda.get_device_name(0)
#     print(f"GPU Name: {device_name}")
# else:
#     print("No GPU available, using CPU.")

# # Step 1: 实例化 VQModel
# ckpt_path = 'vqmodel_checkpoint.ckpt'
# checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

# class VQModel(torch.nn.Module):
#     def __init__(self, ddconfig, embed_dim=3, n_embed=8192):
#         super().__init__()
#         from taming.modules.diffusionmodules.model import Decoder
#         from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
#         self.decoder = Decoder(**ddconfig)
#         self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
#         self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

#     def decode(self, x, force_not_quantize=False):
#         if not force_not_quantize:
#             quant, _, _ = self.quantize(x)
#         else:
#             quant = x
#         dec = self.decoder(self.post_quant_conv(quant))
#         return dec

# vq_model = VQModel(
#     ddconfig={
#         'double_z': False,
#         'z_channels': 3,
#         'resolution': 256,
#         'in_channels': 3,
#         'out_ch': 3,
#         'ch': 128,
#         'ch_mult': [1, 2, 4],
#         'num_res_blocks': 2,
#         'attn_resolutions': [],
#         'dropout': 0.0
#     },
#     embed_dim=3,
#     n_embed=8192
# )

# # 加载 VQModel 权重
# state_dict = checkpoint.get('model_state_dict', checkpoint)
# filtered_state_dict = {k: v for k, v in state_dict.items() if k in vq_model.state_dict()}
# vq_model.load_state_dict(filtered_state_dict, strict=False)
# vq_model.to(device).eval()
# for param in vq_model.parameters():
#     param.requires_grad = False

# # Step 2: 实例化 UNetModel
# unet_config = {
#     "image_size": 64,
#     "in_channels": 6,
#     "out_channels": 3,
#     "model_channels": 224,
#     "attention_resolutions": [8, 4, 2],
#     "num_res_blocks": 2,
#     "channel_mult": [1, 2, 3, 4],
#     "num_head_channels": 32,
# }

# unet_model = UNetModel(**unet_config)
# checkpoint_path_u = 'ddim_c.ckpt'
# pretrained_dict = torch.load(checkpoint_path_u, map_location=device, weights_only=True)
# unet_model.load_state_dict(pretrained_dict)
# unet_model.to(device).eval()
# for param in unet_model.parameters():
#     param.requires_grad = False

# # Step 3: 实例化 GaussianDiffusion
# timesteps = 1000
# gaussian_diffusion = GaussianDiffusion(
#     timesteps=timesteps,
#     beta_schedule='linear',
#     linear_start=0.0015,
#     linear_end=0.0155
# )

# # Step 4: 定义 Dataset 和 DataLoader
# class ImageDataset(Dataset):
#     def __init__(self, image_files, input_folder, image_size=64):
#         self.image_files = image_files
#         self.input_folder = input_folder
#         self.image_size = image_size

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.input_folder, self.image_files[idx])
#         image = Image.open(img_name).convert('RGB').resize((self.image_size, self.image_size), Image.LANCZOS)
#         image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1
#         return image, self.image_files[idx]

# # 处理图像
# input_base_path = '/root/onethingai-fs/BEZIER-40w'
# output_base_path = '/root/onethingai-fs/C_LDM_DDIM20_Validation'
# os.makedirs(output_base_path, exist_ok=True)
# image_size = 64
# ddimstep = 20
# batch_size = 25  # 每次处理的图像数量

# def process_images_with_loader(input_folder, output_folder):
#     image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
#     os.makedirs(output_folder, exist_ok=True)

#     dataset = ImageDataset(image_files, input_folder, image_size)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

#     for images, filenames in dataloader:
#         images = images.to(device)
        
#         with torch.no_grad():
#             sample_tensor = gaussian_diffusion.ddim_sample(
#                 unet_model, 
#                 image_size=image_size, 
#                 batch_size=len(filenames), 
#                 channels=3, 
#                 ddim_timesteps=ddimstep,
#                 cd=images
#             )
#             generated_images = vq_model.decode(sample_tensor).clamp(-1, 1)

#             # 将生成的图像范围从 [-1, 1] 映射到 [0, 255] 并转换为 uint8
#             output_images = ((generated_images + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
#             output_images = output_images.transpose(0, 2, 3, 1)  # 转换为 [batch, height, width, channels]

#         # 使用 NumPy 进行批量保存
#         for j, output_image in enumerate(output_images):
#             Image.fromarray(output_image).save(os.path.join(output_folder, filenames[j]))

# def process_all_folders():
#     total_folders = 4000
#     progress_bar = tqdm(total=total_folders, desc="Processing folders")
    
#     for folder_idx in range(total_folders):
#         folder_name = f"{folder_idx:04d}"
#         input_folder = os.path.join(input_base_path, folder_name)
#         output_folder = os.path.join(output_base_path, folder_name)
#         if os.path.isdir(input_folder):
#             process_images_with_loader(input_folder, output_folder)
#         progress_bar.update(1)
    
#     progress_bar.close()

# if __name__ == "__main__":
#     process_all_folders()

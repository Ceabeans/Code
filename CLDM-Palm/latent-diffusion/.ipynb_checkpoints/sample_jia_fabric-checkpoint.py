import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from DDIM_C import GaussianDiffusion
# pip install lightning
from lightning import Fabric
import os
from tqdm import tqdm


class VQDecoderInterface(torch.nn.Module):
    def __init__(self, ddconfig, embed_dim=3, n_embed=8192, ckpt_path=None):
        super().__init__()
        from taming.modules.diffusionmodules.model import Decoder
        from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1)
        self.load_para(ckpt_path)

    # 加载 VQModel 权重
    def load_para(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        filtered_state_dict = {k: v for k,
                               v in state_dict.items() if k in self.state_dict()}
        self.load_state_dict(filtered_state_dict, strict=False)

    def forward(self, x, force_not_quantize=False):
        if not force_not_quantize:
            quant, _, _ = self.quantize(x)
        else:
            quant = x
        dec = self.decoder(self.post_quant_conv(quant))
        return dec


class SampleLite(Fabric):

    def run(self):
        self.seed_everything(1234)

        # --------------------config------------------------------
        # input_base_path = '/root/onethingai-fs/BEZIER-40w'
        # output_base_path = '/root/onethingai-fs/C_LDM_DDIM20_Validation'
        input_base_path = './bezier_synthetic/'
        output_base_path = './sample'
        image_size = 64
        total_folders = 2
        n_samples = 20   # 每个id生成的图像数量
        batch_size = 20  # 每次处理的图像数量
        ddimstep = 20

        # --------------------dataset------------------------------
        trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        dataset = ImageFolder(input_base_path, transform=trans)
        dataloader = DataLoader(dataset, batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)

        # --------------------model-----------------------------
        decoder = VQDecoderInterface(
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
            n_embed=8192,
            ckpt_path='vqmodel_checkpoint.ckpt',
        )

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
        unet_model.load_state_dict(torch.load(
            checkpoint_path_u, weights_only=True))

        # --------------------Diffusion-----------------------------
        gaussian_diffusion = GaussianDiffusion(
            timesteps=1000,
            beta_schedule='linear',
            linear_start=0.0015,
            linear_end=0.0155
        )

        # --------------------fabric-----------------------------
        dataloader = self.setup_dataloaders(dataloader)
        decoder = self.setup(decoder).eval()
        unet_model = self.setup(unet_model).eval()

        # --------------------sample-----------------------------
        os.makedirs(output_base_path, exist_ok=True)

        with torch.no_grad():
            grid = []
            n_batches = n_samples//batch_size
            for i, (x, id) in enumerate(tqdm(dataloader)):
                x = gaussian_diffusion.ddim_sample(
                    unet_model,
                    image_size=image_size,
                    batch_size=batch_size,
                    channels=3,
                    ddim_timesteps=ddimstep,
                    cd=x,
                )
                x = decoder(x).clamp(-1, 1).cpu()
                grid.append(x)

                if (i+1) % n_batches == 0:
                    grid = torch.cat(grid, dim=0)
                    grid = make_grid(grid/2+0.5, nrow=10, padding=0)
                    save_image(grid, os.path.join(
                        output_base_path, f"{id[0]:04d}.png"))
                    grid = []


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')  # ['high','medium']
    evaler = SampleLite(
        accelerator='cuda',
        devices=-1,
        strategy='auto',
        # precision='16-mixed',
    )
    evaler.run()

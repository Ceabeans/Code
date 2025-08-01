# CLDM-Plam
# Palmprint Code Overview and Guide

![Samples of CLDM-Palm](Sample.png)

## 1. Extracting Palmprint ROI (Region of Interest) — (`roi_extract` folder)
The main script is `extract_data.py`. Modify the path to the original palmprint images and the output path for the palmprint ROI. Run the script to obtain the palmprint ROIs.

## 2. Generating Required Bézier Curve Images — (`bezier_generate` folder)

### Script: `BEZIER.py` (for FID comparison)
- By default, generates 3739 IDs × 10 Bézier curve images (binary black and white, single channel, 256×256, PNG format) with preset parameters.
- **Notes:**
  1. To change the output path, simply modify `parser.add_argument('--output', type=str, default='D:/1-zhangwen-data/BEZIER')`. By default, 3739 folders are generated in the `BEZIER` directory, each containing 10 images.
  2. To change the number of generated images, edit it in `def parse_args():`. (Additionally, ensure that the number of threads can be evenly divided by the number of IDs.) At line 303, adjust the digit length in `filename = join(args.output, '%.4d' % i, '%.2d.png' % s)` for the folder name (default: 4 digits) and each file (default: 2 digits). Also, update the process and ID counts in the print statement at line 310.
  3. Finally, click the run button before `if __name__ == '__main__':` to execute.

### Script: `BEZIER-40w.py` (for recognition)
- By default, generates 4000 IDs × 100 Bézier curve images (binary black and white, single channel, 256×256, PNG format) with preset parameters. The output path should be modified as above.
- **Differences from `BEZIER.py`:**
  1. Parameter adjustment: reduced intra-class perturbation.
  2. Data augmentation: Each ID class includes rotation and translation (the relative position of the lines remains unchanged) to increase the robustness of the Bézier curves.
- **Data Augmentation:** After generating the images, run `bezier-zengqiang.ipynb`. Set `base_path` as above. **Note:** Enhanced images will overwrite the originals! This script also provides a checking function; after augmentation, run the second code block to check if the images are binary, single channel, 256×256, and PNG format.

### Script: `BEZIER-PCE.py` (for PCE-Palm, comparative experiments)
- By default, generates 4000 IDs × 100 PCE-version Bézier curve images (binary black and white, single channel, 256×256, PNG format) with preset parameters. Modify the output path as above.

## 3. Training and Sample Collection for CLDM-Palm — (`CLDM-Palm` folder)
1. **Extract 37,390 Real Palmprint PCE Images:**  
   Modify the input and output paths in `PCEM_numpy.py` under the `PCE-Palm` folder and run the script.
2. **Convert PCE Images to Pseudo-Bézier Images:**  
   Run `train.py` in the `CUT` subfolder under `PCE-Palm`. Refer to [https://github.com/Ukuer/PCE-Palm](https://github.com/Ukuer/PCE-Palm) for parameter and path settings.
3. **Map Real Palmprints to Latent Space:**  
  First, you need to download and add the vq-f4 pre-trained model [https://ommer-lab.com/files/latent-diffusion/vq-f4.zip] parameters from LDM to this directory. Run `VQVAE-GAN_Finetune.ipynb` in the `latent-diffusion` folder. 
4. **Train the Latent Diffusion Model:**  
   Run `C-LDM.ipynb` in the `latent-diffusion` folder.
5. **Sampling:**  
   Adjust sampling steps, output path, etc., as needed and run `sample.py`.

## 4. Evaluating the Quality of Generated Palmprints (`fid` folder)
Modify the comparison image path and run `fid.ipynb`.

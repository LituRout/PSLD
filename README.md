# Solving Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models

The repository contains reproducible `PyTorch` source code of our paper [Solving Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models](link). We present the first framework to solve general inverse problems leveraging pre-trained \textit{latent} diffusion models.  Previously proposed algorithms (such as DPS and DDRM) only apply to *pixel-space* diffusion models.  We theoretically analyze our algorithm showing provable sample recovery in a linear model setting. The algorithmic insight obtained from our analysis extends to more general settings often considered in practice. Experimentally, we outperform previously proposed posterior sampling algorithms in a wide variety of problems including random inpainting, block inpainting, denoising, deblurring, destriping, and super-resolution.

## Comparison with state-of-the-art commercial services leveraging Stable Diffusion

<p align="center"><img src="pics/image.png" width="700" /></p>


## Prerequisites
The implementation is GPU-based. Single GPU (A100) is sufficient to run each experiment. Tested with 
`torch==1.12.0 torchvision==0.13.1a0`. To reproduce the reported results, consider using the exact version of `PyTorch` and its required dependencies as other versions might be incompatible. Make sure to install all the required packages for [`/diffusion-posterior-sampling/`](https://github.com/LituRout/PSLD/tree/main/diffusion-posterior-sampling) and [`/stable-diffusion/`](https://github.com/LituRout/PSLD/tree/main/stable-diffusion). Check if the DPS sampler and Stable Diffusion sampler is working before proceeding to the next steps.

## Repository structure
All the experiments are issued in the form of pretty self-explanatory `python` codes. To execute each code, we provide shell scripts inside `stable-diffusion/run/` folder. 

### Main Experiments
Execute the following commands inside the `stable-diffuson` folder.

**Posterior Sampling**

- `sh run/inverse.sh` for super-resolution (4x) task. 
- `sh run/inverse_rip.sh` for random inpainting task.
- `sh run/inverse_gb.sh` for Gaussian deblur task
- `sh run/inverse_mb.sh` for motion deblur task.
- `sh run/inverse_bip.sh` for box inpainting task.

## Evaluation
**Results on Super-resolution**
<p align="center"><img src="pics/image-sr.png" width="700" /></p>

**Results on Random and Box Inpainting**
<p align="center"><img src="pics/image-rip-bip.png" width="700" /></p>


**Results on Gaussian Deblur**
<p align="center"><img src="pics/image-gb.png" width="700" /></p>

**Results on Motion Deblur**
<p align="center"><img src="pics/image-mb.png" width="700" /></p>


## Credits
- [FFHQ (256x256)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for in-distribution samples;
- [ImageNet (256x256)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for out-of-distribution samples;
- [FID repository](https://github.com/mseitzer/pytorch-fid) to compute **FID** score;
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the generative foundation model;
- [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) for the DPS baseline and measurement operators;
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) for the generative latent diffusion model;
- [RePaint](https://github.com/andreas128/RePaint) for measurement operators.




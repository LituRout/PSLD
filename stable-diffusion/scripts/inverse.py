import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import pdb


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_false',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=1000,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    ## 
    parser.add_argument(
        "--dps_path",
        type=str,
        default='../diffusion-posterior-sampling/',
        help="DPS codebase path",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default='configs/inpainting_config.yaml',
        help="task config yml file",
    )
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default='configs/diffusion_config.yaml',
        help="diffusion config yml file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default='configs/model_config.yaml',
        help="model config yml file",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-1,
        help="inpainting error",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1,
        help="measurement error",
    )
    parser.add_argument(
        "--inpainting",
        type=int,
        default=0,
        help="inpainting",
    )
    parser.add_argument(
        "--general_inverse",
        type=int,
        default=1,
        help="general inverse",
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default='00014.png',
        help='input image',
    )
    parser.add_argument(
        "--skip_low_res",
        action='store_true',
        help='downsample result to 256',
    )
    parser.add_argument(
        "--ffhq256",
        action='store_true',
        help='load SD weights trained on FFHQ',
    )
    ##

    opt = parser.parse_args()
    # pdb.set_trace()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        
    ## 
    if opt.ffhq256:
        print("Using FFHQ 256 finetuned model...")
        opt.config = "models/ldm/ffhq256/config.yaml"
        opt.ckpt = "models/ldm/ffhq256/model.ckpt"
    ##
    
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        # pdb.set_trace()
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    #########################################################
    ## DPS configs
    #########################################################
    sys.path.append(opt.dps_path)

    import yaml
    from guided_diffusion.measurements import get_noise, get_operator
    from util.img_utils import clear_color, mask_generator
    import torch.nn.functional as f
    import matplotlib.pyplot as plt


    def load_yaml(file_path: str) -> dict:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    
    
    model_config=opt.dps_path+opt.model_config
    diffusion_config=opt.dps_path+opt.diffusion_config
    task_config=opt.dps_path+opt.task_config

    # pdb.set_trace()

    # Load configurations
    model_config = load_yaml(model_config)
    diffusion_config = load_yaml(diffusion_config)
    task_config = load_yaml(task_config)

    task_config['data']['root'] = opt.dps_path + 'data/samples/'
    img = plt.imread(task_config['data']['root']+opt.file_id)
    # img = next(iter(loader))

    img = img - img.min()
    img = img / img.max()
    img = torch.FloatTensor(img)
    img = torch.unsqueeze(img, dim=0).permute(0,3,1,2)
    img = img[:,:3,:,:].cuda()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
        **measure_config['mask_opt']
        )
    
    img = f.interpolate(img, opt.H)
    x_checked_image_torch = img[:,:3,:,:].cuda()

    org_image = torch.clone(x_checked_image_torch[0].detach())
    org_image = (org_image - 0.5)/0.5
    org_image = org_image[None,:,:,:].cuda()

    # Exception) In case of inpainging,
    if measure_config['operator'] ['name'] == 'inpainting':
        mask = mask_gen(org_image) # dps mask
        # mask = torch.ones_like(org_image) # no mask
        
        mask = mask[:, 0, :, :].unsqueeze(dim=0)
        # Forward measurement model (Ax + n)
        y = operator.forward(org_image, mask=mask)
        y_n = noiser(y)

    else: 
        # Forward measurement model (Ax + n)
        y = operator.forward(org_image)
        y_n = noiser(y)
        mask = None
    

    #########################################################

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.ffhq256:
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        ip_mask = mask,
                                                        measurements = y_n,
                                                        operator = operator,
                                                        gamma = opt.gamma,
                                                        inpainting = opt.inpainting,
                                                        omega = opt.omega,
                                                        general_inverse=opt.general_inverse,
                                                        noiser=noiser,
                                                        ffhq256=opt.ffhq256)
                    else:
                        # pdb.set_trace()
                        if opt.scale != 1.0 :
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        ip_mask = mask,
                                                        measurements = y_n,
                                                        operator = operator,
                                                        gamma = opt.gamma,
                                                        inpainting = opt.inpainting,
                                                        omega = opt.omega,
                                                        general_inverse=opt.general_inverse,
                                                        noiser=noiser)


                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    
                    
                    if not opt.skip_save:
                        mask = mask.cpu().numpy()
                        
                        for x_sample in x_checked_image_torch:
                            if opt.inpainting:  # Inpainting gluing logic as in SD inpaint.py
                                image = torch.clamp((org_image+1.0)/2.0, min=0.0, max=1.0)
                                image = image.cpu().numpy()
                                
                                inpainted = (1-mask)*image+mask*x_sample.cpu().numpy()
                                inpainted = inpainted.transpose(0,2,3,1)[0]*255
                                Image.fromarray(inpainted.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
                            else:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"temp_{base_count:05}.png"))

                            base_count += 1

                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)
                    
                    # pdb.set_trace()
                    if not opt.skip_low_res:
                        if not opt.skip_save:
                            inpainted_image_low_res = f.interpolate(x_checked_image_torch.type(torch.float32), size=(opt.H//2, opt.W//2))
                            for x_sample in inpainted_image_low_res:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}_low_res.png"))
                                base_count += 1

                        

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                # img = put_watermark(img, wm_encoder)
                img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

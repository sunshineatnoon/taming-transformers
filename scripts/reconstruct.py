import sys
sys.path.append(".")

import torch
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import io
import PIL
import yaml
import torch
import requests
import numpy as np
from imageio import imsave
from scipy.io import loadmat
from omegaconf import OmegaConf
import torchvision.transforms as T
import torchvision.utils as vutils
from taming.models.vqgan import VQModel
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import glob
import os

logit_laplace_eps: float = 0.1

colors = loadmat('/home/xtli/Dropbox/parametric_image/data/color150.mat')['colors']
colors = np.concatenate((colors, colors, colors, colors, colors, colors, colors, colors))

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def colorEncode(labelmap, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def map_pixels(x: torch.Tensor) -> torch.Tensor:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != torch.float:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  idx = indices.view(1, 1, 20, 20)
  idx = F.interpolate(idx.float(), size = (320, 320), mode = 'nearest')
  color = colorEncode(idx.squeeze().cpu().numpy())
  imsave('idx.png', color)
  print(f"VQGAN: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

config1024 = load_config("2021-06-09T12-36-40_custom_simple_vqgan/configs/2021-06-09T12-36-40-project.yaml", display=False)
model1024 = load_vqgan(config1024, ckpt_path="2021-06-09T12-36-40_custom_simple_vqgan/checkpoints/last.ckpt").to(DEVICE)

def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

def reconstruction_pipeline(size=320):
  img_list = glob.glob('/home/xtli/DATA/BSR_processed/train/*_0_img.jpg')
  os.makedirs("vqgan-1024-256-simple", exist_ok=True)
  for img_path in img_list:
    img_name = img_path.split('/')[-1]
    img = PIL.Image.open(img_path)
    x = preprocess(img, target_image_size=size)
    x = x.to(DEVICE)
    print(f"input is of size: {x.shape}")
    x2 = reconstruct_with_vqgan(preprocess_vqgan(x), model1024)
    out_path = os.path.join("vqgan-1024-256-simple/{}.jpg".format(img_name))
    vutils.save_image(x2, out_path, normalize=True)

reconstruction_pipeline()
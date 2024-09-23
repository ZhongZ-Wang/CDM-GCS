import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch, os, shutil, h5py
from modules import UNet_conditional
from cdm import Diffusion


device = "cuda"
model = UNet_conditional(num_classes=850, device=device)
diffusion = Diffusion(img_size=64, device=device)
ckpt = torch.load("../Pretrained_model/ckpt.pt")
model.load_state_dict(ckpt)

n_posterior = 100
true_obs_data = np.loadtxt('../Data_2D/single_test/dobs_norm.txt')
true_obs_data = true_obs_data.reshape(1, -1)
obs = np.repeat(true_obs_data, n_posterior, axis=0)

y = torch.FloatTensor(obs).to(device)
x = diffusion.sample(model, len(y), y, cfg_scale=3)

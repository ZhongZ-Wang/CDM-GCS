import matplotlib.pyplot as plt
import numpy as np
import torch, os, shutil, h5py
from modules import UNet_conditional
from cdm import Diffusion


device = "cuda"
model = UNet_conditional(num_classes=990, device=device)
diffusion = Diffusion(img_size=64, device=device)
ckpt = torch.load("../Pretrained_model/{}/ckpt.pt".format(dir))
model.load_state_dict(ckpt)

if os.path.exists('./perm/'):
    shutil.rmtree('./perm/')
    os.makedirs('./perm/')
else:
    os.makedirs('./perm/')

n_posterior = 100
true_obs_data = np.loadtxt('../Data_3D/single_test/dobs_norm.txt')
true_obs_data = true_obs_data.reshape(1, -1)
obs = np.repeat(true_obs_data, n_posterior, axis=0)

y = torch.FloatTensor(obs).to(device)
x, x_array = diffusion.sample(model, len(y), y, cfg_scale=3)

for i in range(len(x)):
    sampled_perm = x[i].clamp(-1, 1).cpu().numpy()
    plt.imshow(sampled_perm[0], vmin=-1, vmax=1, cmap='RdBu_r')
    plt.savefig('./perm/sampled_perm{}.png'.format(i + 1))
    np.savetxt('./perm/sampled_perm{}.txt'.format(i + 1), sampled_perm.flatten())

x_array = x_array.clamp(-1, 1).cpu().numpy()
hf = h5py.File('denoising{}.h5'.format(n_posterior), 'w')
hf.create_dataset('x_array', data=x_array[:, :20], dtype='f', compression='gzip')
hf.close()
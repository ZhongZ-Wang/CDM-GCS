import numpy as np
import h5py


def load_data(ndata):

    fname1 = '../Data_3D/train_set/perm_norm_n{}.h5'.format(ndata)
    hf = h5py.File(fname1, 'r')
    perm = hf['perm_norm'][:]
    hf.close()

    fname2 = '../Data_3D/train_set/obs_norm_n{}.h5'.format(ndata)
    hf = h5py.File(fname2, 'r')
    obs_norm = hf['obs_norm'][:]
    hf.close()
    obs_data = obs_norm.reshape(ndata, -1)

    return perm, obs_data


# perm, obs_data = load_data(1500)
# print(perm.shape, obs_data.shape)
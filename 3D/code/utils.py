import os
import torch
from torch.utils.data import DataLoader
from load_data import load_data


def l2_loss(pred, true):
    loss = torch.sum((pred-true)**2, dim=[1, 2, 3])
    return torch.mean(loss)


def get_train_data(args):
    perm, obs_data = load_data(args.train_number)
    perm = torch.as_tensor(perm)
    obs_data = torch.as_tensor(obs_data)
    dataset = torch.utils.data.TensorDataset(perm, obs_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)

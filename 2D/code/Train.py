import copy
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import optim

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_train_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    l = len(dataloader)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("logs", datetime.now().strftime('%Y%m%d_%H%M') + '_ndata{}'.format(args.train_number)))

    ema = EMA(args.ema)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
               labels = None
            predicted_noise = model(x_t, t, labels)

            if args.mse:
                loss = mse(noise, predicted_noise)
            else:
                loss = l2_loss(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    torch.save(model.state_dict(), os.path.join("./models", args.run_name, f"ckpt.pt"))
    # torch.save(ema_model.state_dict(), os.path.join("./models", args.run_name, f"ema_ckpt.pt"))
    # torch.save(optimizer.state_dict(), os.path.join("./models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 16
    args.ema = 0.998
    args.epochs = 600  # 300
    args.image_size = 64
    args.num_classes = 850
    args.train_number = 900
    args.device = "cuda"
    args.lr = 2e-4
    args.mse = True
    args.run_name = 'DDPM_conditional_' + datetime.now().strftime('%Y%m%d_%H%M') + \
                    '_batchsize{}'.format(args.batch_size) + '_epochs{}'.format(args.epochs)
    train(args)


if __name__ == '__main__':
    launch()
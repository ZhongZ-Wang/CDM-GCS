import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=400, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 32)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 16)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128, 8)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64, 16)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32, 32)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32, 64)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        if num_classes is not None:
            # self.label_emb = nn.Embedding(num_classes, time_dim)
            self.fc = nn.Sequential(
                nn.Linear(num_classes, 1024),
                nn.SiLU(),
                nn.Linear(1024, 256),
            )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            y = self.fc(y)
            t += y  # （1）
            # (2) t = torch.cat((t, y), 1)
        # print(x.shape)
        x1 = self.inc(x)  # (N, 32, 64, 64)
        # print(x1.shape)
        x2 = self.down1(x1, t)
        # print(x2.shape)
        x2 = self.sa1(x2)
        # print(x2.shape)
        x3 = self.down2(x2, t)
        # print(x3.shape)
        x3 = self.sa2(x3)
        # print(x3.shape)
        x4 = self.down3(x3, t)
        # print(x4.shape)
        x4 = self.sa3(x4)
        # print(x4.shape)

        x4 = self.bot1(x4)
        # print(x4.shape)
        x4 = self.bot2(x4)
        # print(x4.shape)
        x4 = self.bot3(x4)
        # print(x4.shape)

        x = self.up1(x4, x3, t)
        # print('-----')
        # print(x.shape)
        x = self.sa4(x)
        # print(x.shape)
        x = self.up2(x, x2, t)
        # print(x.shape)
        x = self.sa5(x)
        # print(x.shape)
        x = self.up3(x, x1, t)
        # print(x.shape)
        x = self.sa6(x)
        # print(x.shape)
        output = self.outc(x)
        # print(output.shape)
        return output


# if __name__ == '__main__':
#     # net = UNet(device="cpu")
#     # from torchviz import make_dot
#     net = UNet_conditional(num_classes=850, device="cpu")
#     print(sum([p.numel() for p in net.parameters()]))
#     x = torch.randn(3, 1, 64, 64)
#     t = x.new_tensor([500] * x.shape[0]).long()
#     y = torch.randn(3, 850)  # .long()
#     output = net(x, t, y)
#     # print(net(x, t, y).shape)
#     g = make_dot(output)
#     g.render('model', view=False)
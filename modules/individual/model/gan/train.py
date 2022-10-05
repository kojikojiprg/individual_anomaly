import time
from logging import Logger
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .discriminator import Discriminator
from .generator import Generator


def train(
    G: Generator,
    D: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    config: SimpleNamespace,
    device: str,
    logger: Logger,
):
    d_z = config.model.G.d_z
    n_epochs = config.epochs

    g_optim = torch.optim.Adam(
        G.parameters(),
        config.optim.lr_g,
        (config.optim.beta1, config.optim.beta2),
    )
    d_optim = torch.optim.Adam(
        D.parameters(),
        config.optim.lr_d,
        (config.optim.beta1, config.optim.beta2),
    )

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    G.train()
    D.train()
    torch.backends.cudnn.benchmark = True

    logger.info("=> start training")
    for epoch in range(n_epochs):
        d_losses = []
        g_losses = []

        t_epoch_start = time.time()
        for frame_nums, pids, keypoints in dataloader:
            # learn Discriminator
            keypoints = keypoints.to(device)
            mini_batch_size = keypoints.size()[0]
            label_real = torch.full((mini_batch_size,), 1, dtype=torch.float32).to(
                device
            )
            label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float32).to(
                device
            )

            d_out_real, _, _, _ = D(keypoints)
            # print(d_out_real)

            z = torch.randn(mini_batch_size, d_z).to(device)
            fake_keypoints, _, _ = G(z)
            d_out_fake, _, _, _ = D(fake_keypoints)

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            # d_loss_real = criterion(d_out_real.view(-1), label_real)
            # d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # learn Generator
            z = torch.randn(mini_batch_size, d_z).to(device)
            fake_keypoints, _, _ = G(z)
            d_out_fake, _, _, _ = D(fake_keypoints)

            g_loss = -d_out_fake.mean()
            # g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # sum losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        t_epoch_finish = time.time()
        d_loss_mean = np.average(d_losses)
        g_loss_mean = np.average(g_losses)
        logger.info("-------------")
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs} | "
            + f"D_loss:{d_loss_mean:.5e} | "
            + f"G_loss:{g_loss_mean:.5e}"
        )
        logger.info("time: {:.3f} sec.".format(t_epoch_finish - t_epoch_start))
    logger.info("=> finish training")

    return G, D

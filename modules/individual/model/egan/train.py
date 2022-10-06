import time
from logging import Logger
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator


def train(
    G: Generator,
    D: Discriminator,
    E: Encoder,
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
    e_optim = torch.optim.Adam(
        E.parameters(),
        config.optim.lr_e,
        (config.optim.beta1, config.optim.beta2),
    )

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    G.train()
    D.train()
    E.train()
    torch.backends.cudnn.benchmark = True

    logger.info("=> start training")
    for epoch in range(n_epochs):
        d_losses = []
        g_losses = []
        e_losses = []

        t_epoch_start = time.time()
        for frame_nums, pids, keypoints in dataloader:
            keypoints = keypoints.to(device)
            mini_batch_size = keypoints.size()[0]
            label_real = torch.full((mini_batch_size,), 1, dtype=torch.float32).to(
                device
            )
            label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float32).to(
                device
            )

            # train Discriminator
            z_out_real, _, _, _ = E(keypoints)
            d_out_real, _ = D(keypoints, z_out_real)

            z = torch.randn(mini_batch_size, d_z).to(device)
            fake_keypoints, _, _ = G(z)
            d_out_fake, _ = D(fake_keypoints, z)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # train Generator
            z = torch.randn(mini_batch_size, d_z).to(device)
            fake_keypoints, _, _ = G(z)
            d_out_fake, _ = D(fake_keypoints, z)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # train Encoder
            z_out_real, _, _, _ = E(keypoints)
            d_out_real, _ = D(keypoints, z_out_real)

            e_loss = criterion(d_out_real.view(-1), label_fake)

            e_optim.zero_grad()
            e_loss.backward()
            e_optim.step()

            # append losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            e_losses.append(e_loss.item())

        t_epoch_finish = time.time()
        d_loss_mean = np.average(d_losses)
        g_loss_mean = np.average(g_losses)
        e_loss_mean = np.average(e_losses)
        logger.info("-------------")
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs} | "
            + f"G_loss:{g_loss_mean:.5e} | "
            + f"D_loss:{d_loss_mean:.5e} | "
            + f"E_loss:{e_loss_mean:.5e}"
        )
        logger.info("time: {:.3f} sec.".format(t_epoch_finish - t_epoch_start))
    logger.info("=> finish training")

    return G, D, E

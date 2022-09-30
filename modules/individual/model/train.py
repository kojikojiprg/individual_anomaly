import time
from logging import Logger
from types import SimpleNamespace

import numpy as np
import torch

from . import Discriminator, Generator


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

    G.train()
    D.train()
    torch.backends.cudnn.benchmark = True

    logger.info("=> start training")
    for epoch in range(n_epochs):
        t_epoch_start = time.time()
        g_losses = []
        d_losses = []

        for pids, keypoints in dataloader:
            # learn Discriminator
            keypoints = keypoints.to(device)
            mini_batch_size = keypoints.size()[0]

            d_out_real, _, _, _ = D(keypoints)

            z = torch.randn(mini_batch_size, d_z).to(device)
            fake_keypoints, _, _ = G(z)
            d_out_fake, _, _, _ = D(fake_keypoints)

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # learn Generator
            z = torch.randn(mini_batch_size, d_z).to(device)
            fake_keypoints, _, _ = G(z)
            d_out_fake, _, _, _ = D(fake_keypoints)

            g_loss = -d_out_fake.mean()

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # sum losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        t_epoch_finish = time.time()
        logger.info("-------------")
        logger.info(
            "Epoch {}/{} | D_loss:{:.5e} | G_loss:{:.5e}".format(
                epoch, n_epochs, np.mean(d_losses), np.mean(g_losses)
            )
        )
        logger.info("time: {:.3f} sec.".format(t_epoch_finish - t_epoch_start))
    logger.info("=> finish training")

    return G, D

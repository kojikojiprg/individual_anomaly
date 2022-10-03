from typing import Any, Dict, List

import torch

from ..format import Format
from .layers.discriminator import Discriminator
from .layers.generator import Generator


def test_generator(
    G: Generator, num_individual: int, d_z: int, device: str
) -> List[Dict[str, Any]]:
    z = torch.randn((num_individual, d_z)).to(device)

    results = []
    G.eval()
    with torch.no_grad():
        fake_keypoints, weights_spat, weights_temp = G(z)

    for i, (kps, w_spat, w_temp) in enumerate(
        zip(fake_keypoints, weights_spat, weights_temp)
    ):
        kps = kps.cpu().numpy()
        w_spat = w_spat.cpu().numpy()
        w_temp = w_temp.cpu().numpy()

        for frame_num in range(kps):
            results.append(
                {
                    Format.frame_num: frame_num,
                    Format.id: i,
                    Format.keypoints: kps,
                    Format.w_spat: w_spat,
                    Format.w_temp: w_temp,
                }
            )

    return results


def test_discriminator(D: Discriminator):
    D.eval()

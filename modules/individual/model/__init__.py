from .discriminator import Discriminator
from .generator import Generator
from .train import train


class IndividualGAN:
    def __init__(self, config, device):
        self._config = config
        self._device = device
        self._G = Generator(config.model.G, device).to(device)
        self._D = Discriminator(config.model.D, device).to(device)

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    def load_checkpoints(self, checkpoints_path):
        pass

    def train(self, dataloader):
        self._G, self._D = train(
            self._G, self._D, dataloader, self._config, self._device
        )

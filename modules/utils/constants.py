from dataclasses import dataclass


@dataclass(frozen=True)
class Stages:
    train: str = "train"
    test: str = "test"
    inference: str = "inference"

import os
import pickle


def load(path):
    with open(path, "br") as f:
        data = pickle.load(f)
    return data


def dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "bw") as f:
        pickle.dump(data, f)
